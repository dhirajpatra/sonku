"""The conversation loop.

 known people arrive -> greet them by name (everyone in view), open a session, no wake word needed
 unknown face        -> offer to take a photo / ask the name (UNKNOWN_FACE_MODE), or wait for the wake word
 quiet after greeting-> start talking by itself: comment on what it sees and offer help
 session open        -> every utterance is a turn until SESSION_TIMEOUT_SEC of silence
`handle_utterance` holds all the conversation logic and takes plain text, so it is testable without hardware.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import Settings
from app.core import intents
from app.core.phrases import join_names, phrase, phrase_lang
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
    # chat | await_name | await_consent | await_photo_consent | await_name_final
    stage: str = "chat"
    pending_name: str | None = None
    pending_embs: list = field(default_factory=list)  # photo templates held in memory until a name is given
    intro_offered: bool = False
    name_attempts: int = 0
    consent_attempts: int = 0
    stage_seen: str = "chat"
    stage_since: float = field(default_factory=time.time)


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
        self._announced: dict[int, float] = {}  # known people already greeted while in view
        self._gone_since: dict[int, float] = {}
        self._last_user_speech = 0.0
        self._last_robot_speech = 0.0
        self._last_stranger_prompt = 0.0
        self._want_opener: dict | None = None
        self._opener: dict | None = None

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
        self.rt.audio.refresh()  # picks up Bluetooth/USB devices connected or removed while running
        self._check_session_expiry()
        self._social_tick()
        self.state = "listening" if self.session else "idle"
        audio = self.rt.recorder.listen(wait_sec=2.0)
        if audio is None:
            return
        self._last_user_speech = time.time()
        self.state = "transcribing"
        hint = self.session.lang if self.session else None
        t0 = time.time()
        text, lang = self.rt.stt.transcribe(audio, hint)
        stt_ms = int((time.time() - t0) * 1000)
        total_sec = len(audio) / 16000
        # The sentence sounds unfinished ("... and", "wait, let me think"): keep listening for the rest
        for _ in range(3):
            if (not text or total_sec >= self.s.max_utterance_sec - 1 or not intents.looks_incomplete(text)
                    or not self._engaged(text)):
                break
            log.info("[heard so far] %s  (waiting up to %.0fs for the rest)", text, self.s.patience_sec)
            self.state = "listening"
            more = self.rt.recorder.listen(wait_sec=self.s.patience_sec, max_sec=self.s.max_utterance_sec - total_sec)
            if more is None:
                break
            self._last_user_speech = time.time()
            audio = np.concatenate([audio, np.zeros(int(0.3 * 16000), dtype=np.int16), more])
            total_sec = len(audio) / 16000
            self.state = "transcribing"
            t1 = time.time()
            new_text, new_lang = self.rt.stt.transcribe(audio, hint)
            stt_ms += int((time.time() - t1) * 1000)
            if not new_text:
                break
            text, lang = new_text, new_lang
        if not text:
            log.info("[mic] captured %.1fs of sound (loudest %d, threshold %d) but speech-to-text returned nothing",
                     len(audio) / 16000, self.rt.recorder.last_peak, self.rt.recorder.last_thr)
            return
        log.info("[heard %s] %s", lang, text)
        self.handle_utterance(text, lang, stt_ms=stt_ms)

    def _engaged(self, text: str) -> bool:
        """Only wait for more speech when someone is actually talking to the robot."""
        return (self.session is not None or intents.find_wake_word(text, self.s.wake_word_list)[0]
                or self._known_person_nearby() is not None)

    # ------------------------------------------------------------------ who is in view
    @staticmethod
    def _people_of(p) -> list[dict]:
        if p.people:
            return list(p.people)
        if p.status == "known" and p.person_id is not None:
            return [{"id": p.person_id, "name": p.name}]
        return []

    def _scene_text(self, p) -> str:
        parts = [f"{x['name']} (someone you know)" for x in self._people_of(p)]
        if p.unknown:
            parts.append(f"{p.unknown} person{'s' if p.unknown > 1 else ''} you do not know yet")
        return ", ".join(parts)

    def _known_person_nearby(self) -> dict | None:
        w = self.rt.watcher
        p = w.snapshot() if w else None
        if not p or not p.face_present:
            return None
        people = self._people_of(p)
        return self.rt.store.get_person(people[0]["id"]) if people else None

    # ---------------------------------------------------------------- sessions
    def _end_session(self, reason: str = "") -> None:
        if self.session:
            log.info("session ended (%s)", reason or "done")
            maint = getattr(self.rt, "maint", None)
            if maint and self.session.person_id:
                maint.kick()  # write a fresh summary soon if this person has enough new turns
        self.session = None
        self._want_opener = None

    def _resume_history(self, pid: int) -> list:
        """Earlier turns with this person, so a new conversation continues where the last one stopped."""
        if self.s.resume_turns <= 0:
            return []
        entry = self.rt.cache.get(pid)  # served from memory; the cache refreshes itself after new turns
        return list(entry.history) if entry and entry.history else []

    def _new_session(self, person: dict) -> Session:
        return Session(person_id=person["id"], name=person["name"], lang=person["language"],
                       history=self._resume_history(person["id"]))

    def _check_session_expiry(self) -> None:
        sess = self.session
        if not sess:
            return
        now = time.time()
        if sess.stage != sess.stage_seen:
            sess.stage_seen, sess.stage_since = sess.stage, now
        if sess.stage != "chat" and now - sess.stage_since > self.s.stage_timeout_sec:
            log.info("no answer to the %s question; carrying on without it", sess.stage)
            sess.stage, sess.stage_seen, sess.pending_embs, sess.pending_name = "chat", "chat", [], None
        limit = self.s.session_timeout_sec * (1 if sess.stage == "chat" else 2)  # more time to answer a question
        if now - sess.last_activity > limit:
            self._end_session("timeout")
            return
        w = self.rt.watcher
        if w and sess.person_id:
            p = w.snapshot()
            if p.status == "none" and now - p.last_face_ts > self.s.session_absent_sec:
                self._end_session("person left")

    # ------------------------------------------------------------------ social behaviour
    def _social_tick(self) -> None:
        w = self.rt.watcher
        if not w:
            return
        p = w.snapshot()
        self._maybe_greet(p)
        self._maybe_meet_stranger(p)
        self._maybe_proactive(p)

    def _maybe_greet(self, p=None) -> None:
        """Greet known people when they come into view: one hello for everyone who just arrived."""
        w = self.rt.watcher
        if not w:
            return
        p = p or w.snapshot()
        now = time.time()
        people = self._people_of(p)
        present = {x["id"] for x in people}
        for pid in list(self._announced):  # forget people who stayed away a while, so they are greeted on return
            if pid in present:
                self._gone_since.pop(pid, None)
            elif now - self._gone_since.setdefault(pid, now) > self.s.absence_sec + 20:
                self._announced.pop(pid, None)
                self._gone_since.pop(pid, None)
        newcomers = [x for x in people if x["id"] not in self._announced]
        if not newcomers:
            return
        for x in newcomers:
            self._announced[x["id"]] = now
        with self._turn_lock:
            primary = self.rt.store.get_person(people[0]["id"])  # the largest known face: probably the closest person
            if not primary:
                return
            sess = self.session
            if sess and sess.person_id is None and sess.stage == "chat":
                # a conversation started with the wake word before the face was recognised carries on as this person
                sess.person_id, sess.name = primary["id"], primary["name"]
                if not sess.history:
                    sess.history = self._resume_history(primary["id"])
                log.info("conversation continues with %s", primary["name"])
            elif sess and sess.person_id not in present:
                self._end_session("another person arrived")
                self.session = self._new_session(primary)
            elif not sess:
                self.session = self._new_session(primary)
            to_greet = []
            for x in newcomers:
                person = self.rt.store.get_person(x["id"])
                recently = person and person["last_greeted"] and now - person["last_greeted"] < self.s.greet_cooldown_min * 60
                if person and not recently:
                    to_greet.append(person)
            if self.s.greeting_enabled and to_greet:
                for person in to_greet:
                    self.rt.store.mark_greeted(person["id"])
                lang = to_greet[0]["language"]
                if len(to_greet) == 1:
                    text = phrase("greet_known", phrase_lang("greet_known", lang), name=to_greet[0]["name"])
                else:
                    gl = phrase_lang("greet_many", lang)
                    text = phrase("greet_many", gl, names=join_names([x["name"] for x in to_greet], gl))
                self._say(text, lang)
                self._want_opener = {"ts": time.time()}
            log.info("people in view: %s (greeted=%s)", [x["name"] for x in people], [x["name"] for x in to_greet])

    def _maybe_meet_stranger(self, p) -> None:
        """A face nobody recognises has stayed in view: offer to remember it (UNKNOWN_FACE_MODE)."""
        mode = self.s.unknown_face_mode
        if mode == "wake_word" or not (self.rt.watcher and self.rt.faces):
            return
        now = time.time()
        if p.unknown <= 0 or not p.unknown_since or now - p.unknown_since < self.s.unknown_confirm_sec:
            return
        if now - self._last_stranger_prompt < self.s.unknown_prompt_cooldown_min * 60:
            return
        if now - self._last_user_speech < 4.0:  # somebody is talking right now
            return
        with self._turn_lock:
            sess = self.session
            if sess and sess.stage != "chat":
                return
            self._last_stranger_prompt = now
            group = bool(self._people_of(p))
            if not sess:
                sess = self.session = Session(can_enroll=True)
            lang = sess.lang
            sess.last_activity = now
            if mode == "capture_first":
                embs = self.rt.watcher.capture_embeddings(3, unknown_only=True)
                if not embs:  # they moved away: try again soon
                    self._last_stranger_prompt = now - self.s.unknown_prompt_cooldown_min * 60 + 30
                    return
                sess.pending_embs, sess.stage, sess.name_attempts = embs, "await_name_final", 0
                self._say(phrase("stranger_photo_taken", phrase_lang("stranger_photo_taken", lang)), lang)
            elif mode == "ask_name":
                sess.stage, sess.intro_offered, sess.can_enroll, sess.name_attempts = "await_name", True, True, 0
                self._say(phrase("stranger_ask_name", phrase_lang("stranger_ask_name", lang)), lang)
            else:  # ask_photo
                key = "stranger_ask_photo_group" if group else "stranger_ask_photo"
                sess.stage, sess.consent_attempts = "await_photo_consent", 0
                self._say(phrase(key, phrase_lang(key, lang)), lang)
            log.info("stranger in view: asked (%s mode)", mode)

    def _maybe_proactive(self, p) -> None:
        """After a greeting, if the person stays quiet, start talking: comment on the scene and offer help."""
        if not self.s.proactive_enabled:
            return
        now = time.time()
        op = self._opener
        if op is not None:
            if op["text"] is None:
                return  # still being composed in the background
            self._opener = None
            sess = self.session
            if self._last_user_speech > op["ts"] or not sess or sess.stage != "chat":
                return  # the person started talking meanwhile: the remark is stale
            with self._turn_lock:
                sess.history.append(("assistant", op["text"]))
                self._say(op["text"], sess.lang)
                sess.last_activity = time.time()
            return
        want = self._want_opener
        sess = self.session
        if not want or not sess or sess.stage != "chat":
            return
        if self._last_user_speech > want["ts"]:
            self._want_opener = None  # they already started talking
            return
        if now - want["ts"] < self.s.proactive_idle_sec or now - self._last_robot_speech < 2.0:
            return
        self._want_opener = None
        self._opener = {"ts": now, "text": None}
        threading.Thread(target=self._compose_opener, args=(self._opener, sess, self._scene_text(p)),
                         daemon=True, name="opener").start()

    def _compose_opener(self, holder: dict, sess: Session, scene: str) -> None:
        text = ""
        try:
            image = self.rt.camera.jpeg_b64() if self.rt.camera else None
            entry = self.rt.cache.get(sess.person_id)
            system = (prompts.system_prompt(sess.lang, sess.name, entry.summary if entry else "",
                                            has_image=bool(image), scene=scene) + "\n" + prompts.OPENER_RULES)
            text = " ".join(self.rt.router.reply_stream(system, list(sess.history)[-4:], "(Start the conversation now.)",
                                                        image, effort="minimal"))
        except LLMUnavailable:
            pass
        except Exception:
            log.exception("could not compose an opener")
        holder["text"] = text.strip() or phrase("offer_help", phrase_lang("offer_help", sess.lang))

    def _open_session_from_wake(self) -> None:
        p = self.rt.watcher.snapshot() if self.rt.watcher else None
        if p:
            people = self._people_of(p)
            person = self.rt.store.get_person(people[0]["id"]) if people else None
            if person:
                self.session = self._new_session(person)
                return
        can_enroll = bool(p and p.face_present and p.status in ("unknown", "checking") and self.rt.faces)
        self.session = Session(can_enroll=can_enroll)

    # -------------------------------------------------------------------- turns
    def handle_utterance(self, text: str, lang: str = "en", stt_ms: int = 0) -> str | None:
        with self._turn_lock:
            self._last_user_speech = time.time()
            if self.session is None:
                heard, rest = intents.find_wake_word(text, self.s.wake_word_list)
                if heard:
                    self._open_session_from_wake()
                else:
                    person = self._known_person_nearby()
                    if not person:
                        return None  # nobody known in front of the robot: only the wake word starts a conversation
                    self.session = self._new_session(person)
                    rest = text
                    log.info("conversation resumed with %s (known face, no wake word)", person["name"])
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
            self._want_opener = None
            lang = self._pick_lang(sess, text, lang)
            if sess.stage == "await_name":
                return self._stage_name(sess, text, lang)
            if sess.stage == "await_consent":
                return self._stage_consent(sess, text, lang)
            if sess.stage == "await_photo_consent":
                return self._stage_photo_consent(sess, text, lang)
            if sess.stage == "await_name_final":
                return self._stage_name_final(sess, text, lang)
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

    # -- wake-word enrolment: name -> permission -> photo ---------------------------
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
        if self.s.require_face_consent and not intents.implies_consent(text):
            sess.stage = "await_consent"
            self._say(phrase("ask_consent", phrase_lang("ask_consent", lang), name=name), lang)
            return None
        self._enroll(sess, name, lang)
        return None

    def _stage_consent(self, sess: Session, text: str, lang: str) -> None:
        if intents.is_yes(text) or intents.implies_consent(text):
            self._enroll(sess, sess.pending_name or "Friend", lang)
        elif intents.is_no(text) or sess.consent_attempts >= 1:
            sess.stage = "chat"
            self._say(phrase("declined", phrase_lang("declined", lang)), lang)
        else:
            sess.consent_attempts += 1
            self._say(phrase("consent_retry", phrase_lang("consent_retry", lang)), lang)
        return None

    def _enroll(self, sess: Session, name: str, lang: str) -> None:
        sess.stage = "chat"
        w = self.rt.watcher
        embs = w.capture_embeddings(3, unknown_only=True) if w else []
        if not embs:
            self._say(phrase("enroll_fail", phrase_lang("enroll_fail", lang)), lang)
            return
        self._save_person(sess, name, embs, lang)

    # -- stranger enrolment: photo permission -> photo -> name ---------------------
    def _stage_photo_consent(self, sess: Session, text: str, lang: str) -> None:
        if intents.is_yes(text) or intents.implies_consent(text):
            w = self.rt.watcher
            embs = w.capture_embeddings(3, unknown_only=True) if w else []
            if not embs:
                sess.stage = "chat"
                self._say(phrase("enroll_fail", phrase_lang("enroll_fail", lang)), lang)
                return None
            sess.pending_embs, sess.stage, sess.name_attempts = embs, "await_name_final", 0
            self._say(phrase("photo_taken_ask_name", phrase_lang("photo_taken_ask_name", lang)), lang)
        elif intents.is_no(text) or sess.consent_attempts >= 1:
            sess.stage = "chat"
            self._say(phrase("declined", phrase_lang("declined", lang)), lang)
        else:
            sess.consent_attempts += 1
            self._say(phrase("photo_consent_retry", phrase_lang("photo_consent_retry", lang)), lang)
        return None

    def _stage_name_final(self, sess: Session, text: str, lang: str) -> None:
        if intents.is_no(text) and not intents.extract_name(text):
            sess.pending_embs, sess.stage = [], "chat"
            self._say(phrase("declined", phrase_lang("declined", lang)), lang)
            return None
        name = self.rt.decision.extract_name(text)
        if not name:
            sess.name_attempts += 1
            if sess.name_attempts >= 2:
                sess.pending_embs, sess.stage = [], "chat"  # never stored: the photo is simply dropped
                self._say(phrase("name_final_skip", phrase_lang("name_final_skip", lang)), lang)
            else:
                self._say(phrase("ask_name_again", phrase_lang("ask_name_again", lang)), lang)
            return None
        embs, sess.pending_embs = sess.pending_embs, []
        sess.stage = "chat"
        self._save_person(sess, name, embs, lang)
        return None

    def _save_person(self, sess: Session, name: str, embs: list, lang: str) -> None:
        faces, store = self.rt.faces, self.rt.store
        known, _score = faces.match(embs[0], store.gallery())
        if known:  # never create a duplicate profile for someone already stored
            person = store.get_person(known)
            if sess.person_id is None:
                sess.person_id, sess.name = known, person["name"]
            self._say(phrase("already_known", phrase_lang("already_known", lang), name=person["name"]), lang)
            return
        pid = store.add_person(name, lang)
        for e in embs:
            store.add_face(pid, e)
        store.mark_greeted(pid)
        self._announced[pid] = time.time()  # the watcher will confirm them soon: no second greeting
        if sess.person_id is None:  # a newcomer talking to Sonku; a known person's conversation stays theirs
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
        self._last_robot_speech = time.time()
        if self.session:  # the idle timer starts when the robot stops talking, not when the person did
            self.session.last_activity = time.time()

    def _chat(self, sess: Session, text: str, lang: str, stt_ms: int = 0, speak: bool = True, persist: bool = True) -> str:
        t0 = time.time()
        self.state = "thinking"
        image = None
        t_dec = time.time()
        if self.rt.camera and self.rt.decision.needs_vision(text):
            image = self.rt.camera.jpeg_b64()
        decision_ms = int((time.time() - t_dec) * 1000)
        entry = self.rt.cache.get(sess.person_id)  # compact notes, no database read when the person is hot
        w = self.rt.watcher
        scene = self._scene_text(w.snapshot()) if w else ""
        system = prompts.system_prompt(lang, sess.name, entry.summary if entry else "", has_image=bool(image), scene=scene)
        history = list(sess.history)
        effort = self.rt.decision.reasoning_effort(text, bool(image))
        marks: dict = {}

        def sentences():
            for sent in self.rt.router.reply_stream(system, history, text, image, effort=effort):
                marks.setdefault("first", time.time())
                yield sent

        try:
            if speak and self._speech_ready():
                self.state = "speaking"
                reply = self.rt.tts.speak_stream(sentences(), lang, cue=True)
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
            "decision_ms": decision_ms,
            "llm_first_sentence_ms": int((marks["first"] - t0) * 1000) if "first" in marks else None,
            "first_audio_ms": int((first_audio - t0) * 1000) if first_audio else None,
            "total_ms": total,
            "backend": self.rt.router.last_backend,
            "vision": bool(image),
            "effort": effort or self.s.meta_reasoning_effort,
        }
        log.info("turn timing: %s", self.metrics)
        if reply and persist:
            sess.history += [("user", text), ("assistant", reply)]
            limit = self.s.history_turns * 2
            if len(sess.history) > limit:  # drop the oldest half in one go: a sliding window would change the prefix every turn
                del sess.history[: limit // 2]
            self.rt.store.add_interaction(sess.person_id, lang, text, reply, self.rt.router.last_backend, total, bool(image))
        self._last_robot_speech = time.time()
        sess.last_activity = time.time()
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
