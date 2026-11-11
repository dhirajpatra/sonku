"""The conversation loop.

 known face arrives  -> greet by name, open a session (no wake word needed)
 unknown face        -> stay quiet until the wake word, then offer to learn their name + face
 session open        -> every utterance is a turn until SESSION_TIMEOUT_SEC of silence
`handle_utterance` holds all the logic and takes plain text, so it is testable without hardware.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.core import intents
from app.core.phrases import phrase, phrase_lang
from app.llm import prompts
from app.llm.router import LLMUnavailable
from app.utils.lang import detect_lang, normalize_lang

log = logging.getLogger("sonku.orchestrator")


@dataclass
class Session:
    person_id: int | None = None
    name: str | None = None
    lang: str = "en"
    can_enroll: bool = False
    started: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    history: list = field(default_factory=list)  # [(role, text)]
    stage: str = "chat"  # chat | await_name | await_consent
    pending_name: str | None = None
    intro_offered: bool = False
    name_attempts: int = 0
    consent_attempts: int = 0


class Orchestrator(threading.Thread):
    def __init__(self, rt):
        super().__init__(daemon=True, name="orchestrator")
        self.rt = rt
        self.s: Settings = rt.s
        self.session: Session | None = None
        self.state = "idle"
        self.paused = False
        self.metrics: dict = {}
        self._stop_evt = threading.Event()
        self._turn_lock = threading.RLock()
        self._handled_seq = 0

    # ------------------------------------------------------------------ thread
    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        log.info("orchestrator started")
        while not self._stop_evt.is_set():
            try:
                if self.paused or self.rt.recorder is None:
                    self.state = "paused" if self.paused else "idle"
                    time.sleep(0.5)
                    continue
                self._tick()
            except Exception:
                log.exception("orchestrator loop error")
                time.sleep(1.0)

    def _tick(self) -> None:
        self._check_session_expiry()
        self._maybe_greet()
        self.state = "listening" if self.session else "idle"
        audio = self.rt.recorder.listen(wait_sec=2.0)
        if audio is None:
            return
        self.state = "transcribing"
        t0 = time.time()
        text, lang = self.rt.stt.transcribe(audio, self.session.lang if self.session else None)
        stt_ms = int((time.time() - t0) * 1000)
        if not text:
            return
        log.info("[heard %s] %s", lang, text)
        self.handle_utterance(text, lang, stt_ms=stt_ms)

    # ---------------------------------------------------------------- sessions
    def _end_session(self, reason: str = "") -> None:
        if self.session:
            log.info("session ended (%s)", reason or "done")
        self.session = None

    def _check_session_expiry(self) -> None:
        sess = self.session
        if not sess:
            return
        now = time.time()
        if now - sess.last_activity > self.s.session_timeout_sec:
            self._end_session("timeout")
            return
        w = self.rt.watcher
        if w and sess.person_id:
            p = w.snapshot()
            if p.status == "none" and now - p.last_face_ts > self.s.absence_sec + 5:
                self._end_session("person left")

    def _maybe_greet(self) -> None:
        w = self.rt.watcher
        if not w:
            return
        p = w.snapshot()
        if p.status != "known" or p.seq == self._handled_seq:
            return
        self._handled_seq = p.seq
        with self._turn_lock:
            if self.session and self.session.person_id == p.person_id:
                return
            person = self.rt.store.get_person(p.person_id)
            if not person:
                return
            if self.session:
                self._end_session("another person arrived")
            self.session = Session(person_id=person["id"], name=person["name"], lang=person["language"])
            recently = person["last_greeted"] and time.time() - person["last_greeted"] < self.s.greet_cooldown_min * 60
            if self.s.greeting_enabled and not recently:
                self.rt.store.mark_greeted(person["id"])
                lang = person["language"]
                self._say(phrase("greet_known", phrase_lang("greet_known", lang), name=person["name"]), lang)
            log.info("session opened for %s (greeted=%s)", person["name"], not recently)

    def _open_session_from_wake(self) -> None:
        p = self.rt.watcher.snapshot() if self.rt.watcher else None
        if p and p.status == "known":
            person = self.rt.store.get_person(p.person_id)
            if person:
                self.session = Session(person_id=person["id"], name=person["name"], lang=person["language"])
                return
        can_enroll = bool(p and p.face_present and p.status in ("unknown", "checking") and self.rt.faces)
        self.session = Session(can_enroll=can_enroll)

    # -------------------------------------------------------------------- turns
    def handle_utterance(self, text: str, lang: str = "en", stt_ms: int = 0) -> str | None:
        with self._turn_lock:
            if self.session is None:
                heard, rest = intents.find_wake_word(text, self.s.wake_word_list)
                if not heard:
                    return None
                self._open_session_from_wake()
                sess = self.session
                if not rest:
                    if sess.can_enroll:
                        sess.stage, sess.intro_offered = "await_name", True
                        self._say(phrase("meet_ask_name", phrase_lang("meet_ask_name", lang)), lang)
                    else:
                        self._say(phrase("listening", phrase_lang("listening", lang)), lang)
                    return None
                text = rest
            sess = self.session
            sess.last_activity = time.time()
            lang = self._pick_lang(sess, text, lang)
            if sess.stage == "await_name":
                return self._stage_name(sess, text, lang)
            if sess.stage == "await_consent":
                return self._stage_consent(sess, text, lang)
            if intents.is_forget(text):
                return self._forget(sess, lang)
            if intents.is_goodbye(text):
                self._say(phrase("bye", phrase_lang("bye", lang)), lang)
                self._end_session("goodbye")
                return None
            reply = self._chat(sess, text, lang, stt_ms=stt_ms)
            if sess.can_enroll and not sess.intro_offered and sess.person_id is None:
                sess.intro_offered, sess.stage = True, "await_name"
                self._say(phrase("ask_name", phrase_lang("ask_name", lang)), lang)
            sess.last_activity = time.time()
            return reply

    def _pick_lang(self, sess: Session, text: str, stt_lang: str) -> str:
        lang = normalize_lang(stt_lang if stt_lang != "en" else detect_lang(text, "en"))
        if sess.person_id and lang != sess.lang and len(text.split()) >= 2:
            self.rt.store.set_language(sess.person_id, lang)  # remember each person's language for the next greeting
        sess.lang = lang
        return lang

    def _stage_name(self, sess: Session, text: str, lang: str) -> None:
        name = self.rt.decision.extract_name(text)
        if not name:
            sess.name_attempts += 1
            if sess.name_attempts >= 2:
                sess.stage = "chat"
                self._say(phrase("skip_name", phrase_lang("skip_name", lang)), lang)
            else:
                self._say(phrase("ask_name_again", phrase_lang("ask_name_again", lang)), lang)
            return None
        sess.pending_name = name
        if self.s.require_face_consent:
            sess.stage = "await_consent"
            self._say(phrase("ask_consent", phrase_lang("ask_consent", lang), name=name), lang)
            return None
        self._enroll(sess, name, lang)
        return None

    def _stage_consent(self, sess: Session, text: str, lang: str) -> None:
        if intents.is_yes(text):
            self._enroll(sess, sess.pending_name or "Friend", lang)
        elif intents.is_no(text) or sess.consent_attempts >= 1:
            sess.stage = "chat"
            self._say(phrase("declined", phrase_lang("declined", lang)), lang)
        else:
            sess.consent_attempts += 1
            self._say(phrase("ask_consent", phrase_lang("ask_consent", lang), name=sess.pending_name or ""), lang)
        return None

    def _enroll(self, sess: Session, name: str, lang: str) -> None:
        sess.stage = "chat"
        w, faces, store = self.rt.watcher, self.rt.faces, self.rt.store
        embs = w.capture_embeddings(3) if w else []
        if not embs:
            self._say(phrase("enroll_fail", phrase_lang("enroll_fail", lang)), lang)
            return
        known, _score = faces.match(embs[0], store.gallery())
        if known:  # never create a duplicate profile for someone already stored
            person = store.get_person(known)
            sess.person_id, sess.name = known, person["name"]
            self._say(phrase("already_known", phrase_lang("already_known", lang), name=person["name"]), lang)
            return
        pid = store.add_person(name, lang)
        for e in embs:
            store.add_face(pid, e)
        store.mark_greeted(pid)
        sess.person_id, sess.name = pid, name
        log.info("enrolled %s with %d face templates", name, len(embs))
        self._say(phrase("enrolled", phrase_lang("enrolled", lang), name=name), lang)

    def _forget(self, sess: Session, lang: str) -> None:
        if sess.person_id:
            self.rt.store.delete_person(sess.person_id)
            if self.rt.watcher:
                self.rt.watcher.reset_identity()
            log.info("person %s asked to be forgotten and was deleted", sess.person_id)
            self._say(phrase("forgot", phrase_lang("forgot", lang)), lang)
        else:
            self._say(phrase("forgot_none", phrase_lang("forgot_none", lang)), lang)
        self._end_session("forgotten")

    # --------------------------------------------------------------------- chat
    def _speech_ready(self) -> bool:
        return bool(self.rt.tts and self.rt.tts.ready)

    def _say(self, text: str, lang: str = "en") -> None:
        self.state = "speaking"
        if self._speech_ready():
            self.rt.tts.speak_stream([text], lang)
        else:
            log.info("[Sonku %s] (no speaker) %s", lang, text)

    def _chat(self, sess: Session, text: str, lang: str, stt_ms: int = 0, speak: bool = True, persist: bool = True) -> str:
        t0 = time.time()
        self.state = "thinking"
        image = None
        if self.rt.camera and self.rt.decision.needs_vision(text):
            image = self.rt.camera.jpeg_b64()
        summary = ""
        if sess.person_id:
            person = self.rt.store.get_person(sess.person_id)
            summary = person["summary"] if person else ""
        system = prompts.system_prompt(lang, sess.name, summary)
        history = sess.history[-self.s.history_turns * 2:]
        marks: dict = {}

        def sentences():
            for sent in self.rt.router.reply_stream(system, history, text, image):
                marks.setdefault("first", time.time())
                yield sent

        try:
            if speak and self._speech_ready():
                self.state = "speaking"
                reply = self.rt.tts.speak_stream(sentences(), lang)
            else:
                reply = " ".join(sentences())
        except LLMUnavailable:
            reply = phrase("llm_down", phrase_lang("llm_down", lang))
            if speak:
                self._say(reply, phrase_lang("llm_down", lang))
        total = int((time.time() - t0) * 1000)
        first_audio = self.rt.tts.first_audio_ts if (speak and self.rt.tts) else 0.0
        self.metrics = {
            "stt_ms": stt_ms,
            "llm_first_sentence_ms": int((marks["first"] - t0) * 1000) if "first" in marks else None,
            "first_audio_ms": int((first_audio - t0) * 1000) if first_audio else None,
            "total_ms": total,
            "backend": self.rt.router.last_backend,
            "vision": bool(image),
        }
        log.info("turn timing: %s", self.metrics)
        if reply and persist:
            sess.history += [("user", text), ("assistant", reply)]
            self.rt.store.add_interaction(sess.person_id, lang, text, reply, self.rt.router.last_backend, total, bool(image))
        self.state = "listening" if self.session else "idle"
        return reply

    # ------------------------------------------------------------ admin helpers
    def text_turn(self, text: str, speak: bool = False) -> dict:
        """Run one turn from text (admin panel test). Uses a throwaway session and stores nothing."""
        with self._turn_lock:
            lang = normalize_lang(detect_lang(text, "en"))
            reply = self._chat(Session(lang=lang), text, lang, speak=speak, persist=False)
            return {"reply": reply, "lang": lang, "metrics": self.metrics}

    def say(self, text: str, lang: str = "en") -> None:
        with self._turn_lock:
            self._say(text, normalize_lang(lang))

    def end_session(self) -> None:
        with self._turn_lock:
            self._end_session("admin")

    def describe(self) -> dict:
        s = self.session
        return {
            "state": self.state,
            "paused": self.paused,
            "session": None if not s else {
                "person_id": s.person_id, "name": s.name, "lang": s.lang, "stage": s.stage,
                "idle_sec": round(time.time() - s.last_activity), "turns": len(s.history) // 2,
            },
            "last_turn": self.metrics,
        }
