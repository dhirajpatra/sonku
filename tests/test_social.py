import time

import numpy as np

from app.audio.tts import CUES, TTS
from app.llm.meta_client import MetaClient
from app.vision.faces import FaceEngine
from app.vision.presence import Presence, PresenceWatcher
from tests.test_features import OK, R
from tests.test_orchestrator import FakeFaces, FakeWatcher, stub_llm


class GridFaces(FaceEngine):
    """A face engine that 'sees' whatever rows the test puts in front of it."""

    def __init__(self, s):
        super().__init__(s)
        self.rows, self.embs = [], {}

    def detect(self, frame):
        return list(self.rows)

    def embed(self, frame, row):
        return self.embs[int(row[14])]


def row(i, w=100):
    r = np.zeros(15, dtype=np.float32)
    r[2] = r[3] = w
    r[14] = i
    return r


def vectors(n=3):
    rng = np.random.RandomState(7)
    return {i: (lambda v: v / np.linalg.norm(v))(rng.randn(128).astype(np.float32)) for i in range(1, n + 1)}


def recorder(rt):
    said = []
    rt.orch._say = lambda text, lang="en": said.append(text)
    return said


# ---------------------------------------------------------------- several people in view
def test_everyone_in_view_is_recognised(rt):
    vec = vectors()
    a, b = rt.store.add_person("Asha"), rt.store.add_person("Ravi")
    rt.store.add_face(a, vec[1])
    rt.store.add_face(b, vec[2])
    f = GridFaces(rt.s)
    f.embs, f.rows = vec, [row(1, 120), row(2, 90), row(3, 80)]  # Asha, Ravi and a stranger
    w = PresenceWatcher(rt.s, None, f, rt.store)
    frame, t = np.zeros((20, 20, 3), np.uint8), time.time()
    w._step(frame, t)
    w._step(frame, t + 2.1)
    p = w.snapshot()
    assert [x["name"] for x in p.people] == ["Asha", "Ravi"]  # largest (closest) face first
    assert p.faces == 3 and p.unknown == 1 and p.seq == 2 and p.name == "Asha" and p.unknown_since > 0


def test_enrolment_photo_uses_the_unknown_face_not_the_known_one(rt):
    vec = vectors()
    a = rt.store.add_person("Asha")
    rt.store.add_face(a, vec[1])

    class Cam:
        def read(self):
            return np.zeros((20, 20, 3), np.uint8)

    f = GridFaces(rt.s)
    f.embs, f.rows = vec, [row(1, 150), row(3, 80)]  # the known person is the bigger face
    w = PresenceWatcher(rt.s, Cam(), f, rt.store)
    embs = w.capture_embeddings(2, spacing=0, timeout=2, unknown_only=True)
    assert len(embs) == 2 and all(np.allclose(e, vec[3]) for e in embs)


def test_two_people_are_greeted_together_and_latecomers_alone(rt):
    a, b, c = (rt.store.add_person(n) for n in ("Asha", "Ravi", "Meera"))
    p = Presence(face_present=True, status="known", person_id=a, name="Asha", faces=2, seq=2,
                 people=[{"id": a, "name": "Asha"}, {"id": b, "name": "Ravi"}])
    rt.watcher = FakeWatcher(p)
    said = recorder(rt)
    rt.orch._maybe_greet()
    assert said == ["Hello Asha and Ravi! Nice to see you all."] and rt.orch.session.person_id == a
    rt.orch._maybe_greet()
    assert len(said) == 1  # same people: no second hello
    p.people.append({"id": c, "name": "Meera"})
    rt.orch._maybe_greet()
    assert said[-1] == "Hello Meera! Nice to see you again." and rt.orch.session.person_id == a


# ---------------------------------------------------------------- strangers
def stranger(rt, mode):
    rt.s.unknown_face_mode = mode
    p = Presence(face_present=True, status="unknown", faces=1, unknown=1, unknown_since=time.time() - 10)
    rt.watcher, rt.faces = FakeWatcher(p), FakeFaces()
    stub_llm(rt)
    return p, recorder(rt)


def test_ask_photo_mode_asks_then_photographs_then_asks_the_name(rt):
    p, said = stranger(rt, "ask_photo")
    rt.orch._maybe_meet_stranger(p)
    assert "take your photo" in said[0] and rt.orch.session.stage == "await_photo_consent" and rt.store.list_people() == []
    rt.orch.handle_utterance("yes", "en")
    assert rt.orch.session.stage == "await_name_final" and rt.store.list_people() == []  # photo is only held in memory
    rt.orch.handle_utterance("my name is Meera", "en")
    people = rt.store.list_people()
    assert [x["name"] for x in people] == ["Meera"] and people[0]["face_count"] == 3
    assert rt.orch.session.person_id == people[0]["id"]


def test_capture_first_mode_takes_the_photo_then_asks_the_name(rt):
    p, said = stranger(rt, "capture_first")
    rt.orch._maybe_meet_stranger(p)
    assert "just taken a quick photo" in said[0] and rt.orch.session.stage == "await_name_final"
    assert rt.store.list_people() == []  # nothing is saved until a name is given
    rt.orch.handle_utterance("I'm Ravi", "en")
    assert [x["name"] for x in rt.store.list_people()] == ["Ravi"]


def test_declined_or_unnamed_photo_is_dropped_and_stranger_not_nagged(rt):
    p, said = stranger(rt, "capture_first")
    rt.orch._maybe_meet_stranger(p)
    rt.orch.handle_utterance("no", "en")
    assert rt.store.list_people() == [] and rt.orch.session.stage == "chat" and not rt.orch.session.pending_embs
    rt.orch._maybe_meet_stranger(p)
    assert len(said) == 2  # photo notice + "declined" only: the cooldown stops a second prompt
    rt.orch._maybe_meet_stranger(p)
    assert len(said) == 2  # nothing new: they spoke a moment ago, so Sonku does not interrupt
    rt.orch._last_stranger_prompt = rt.orch._last_user_speech = 0.0
    rt.orch._maybe_meet_stranger(p)
    rt.orch.handle_utterance("blah blah blah", "en")
    rt.orch.handle_utterance("blah blah blah", "en")
    assert rt.store.list_people() == [] and rt.orch.session.stage == "chat"


def test_ask_name_mode_and_wake_word_mode(rt):
    p, said = stranger(rt, "ask_name")
    rt.orch._maybe_meet_stranger(p)
    assert "Who are you" in said[0] and rt.orch.session.stage == "await_name"
    rt.orch._end_session()
    p, said = stranger(rt, "wake_word")
    rt.orch._maybe_meet_stranger(p)
    assert said == [] and rt.orch.session is None


def test_stranger_must_stay_in_view_before_being_asked(rt):
    p, said = stranger(rt, "ask_photo")
    p.unknown_since = time.time() - 0.5
    rt.orch._maybe_meet_stranger(p)
    assert said == []


def test_unanswered_photo_question_is_dropped(rt):
    p, said = stranger(rt, "ask_photo")
    rt.orch._maybe_meet_stranger(p)
    sess = rt.orch.session
    rt.orch._check_session_expiry()  # notes when the question was asked
    sess.stage_since -= 1000
    rt.orch._check_session_expiry()
    assert rt.orch.session is sess and sess.stage == "chat"


# ---------------------------------------------------------------- talking first, like family
def known_in_view(rt, unknown=1):
    class Cam:
        def jpeg_b64(self):
            return "IMG"

    rt.camera = Cam()
    pid = rt.store.add_person("Asha", "en")
    p = Presence(face_present=True, status="known", person_id=pid, name="Asha", faces=1 + unknown, unknown=unknown, seq=1,
                 people=[{"id": pid, "name": "Asha"}])
    rt.watcher = FakeWatcher(p)
    return p


def wait_for_opener(rt):
    for _ in range(100):
        if rt.orch._opener and rt.orch._opener["text"] is not None:
            return
        time.sleep(0.05)
    raise AssertionError("opener never finished")


def test_quiet_person_gets_a_friendly_opener_about_what_the_camera_sees(rt):
    p = known_in_view(rt)
    said = recorder(rt)
    calls = stub_llm(rt, reply="I can see someone sitting at the dining table. Who is she?")
    rt.orch._maybe_greet()
    assert said == ["Hello Asha! Nice to see you again."] and rt.orch._want_opener
    rt.orch._maybe_proactive(p)
    assert rt.orch._opener is None  # too soon: give the person a moment to speak first
    rt.orch._want_opener["ts"] -= 20
    rt.orch._maybe_proactive(p)
    wait_for_opener(rt)
    rt.orch._maybe_proactive(p)
    assert said[-1] == "I can see someone sitting at the dining table. Who is she?"
    assert calls[0][3] == "IMG" and "1 person you do not know yet" in calls[0][0] and "Asha (someone you know)" in calls[0][0]
    assert rt.orch.session.history[-1] == ("assistant", said[-1])


def test_opener_is_dropped_if_the_person_spoke_first(rt):
    p = known_in_view(rt, unknown=0)
    said = recorder(rt)
    stub_llm(rt, reply="Anything I can help with?")
    rt.orch._maybe_greet()
    rt.orch._want_opener["ts"] -= 20
    rt.orch._maybe_proactive(p)
    wait_for_opener(rt)
    rt.orch._last_user_speech = time.time() + 5  # they started talking while it was being composed
    rt.orch._maybe_proactive(p)
    assert said == ["Hello Asha! Nice to see you again."]
    rt.s.proactive_enabled = False
    rt.orch._want_opener = {"ts": 0}
    rt.orch._maybe_proactive(p)
    assert rt.orch._opener is None


# ---------------------------------------------------------------- speed
def test_thinking_cue_only_when_the_answer_is_slow(rt):
    rt.s.thinking_cue_after_sec = 0.2
    tts, played = TTS(rt.s), []
    tts._synth = lambda text, lang: text
    tts._play = lambda path: played.append(path)

    def slow():
        time.sleep(0.8)
        yield "Real answer."

    def fast():
        yield "Real answer."

    assert tts.speak_stream(slow(), "en", cue=True) == "Real answer."
    assert played[0] in CUES["en"] and played[1] == "Real answer."
    played.clear()
    tts.speak_stream(fast(), "en", cue=True)
    assert played == ["Real answer."]
    played.clear()
    tts.speak_stream(slow(), "en", cue=False)
    assert played == ["Real answer."]


def test_light_thinking_only_for_short_casual_turns(rt):
    d = rt.decision
    assert d.reasoning_effort("how are you today", False) == "minimal"
    assert d.reasoning_effort("explain why the sky is blue", False) is None
    assert d.reasoning_effort("how do I reset my router password", False) is None
    assert d.reasoning_effort("tell me about the history of the roman empire in great detail please", False) is None
    assert d.reasoning_effort("what is this", True) is None  # photos keep full effort
    rt.s.adaptive_effort = False
    assert d.reasoning_effort("hello", False) is None


def test_meta_effort_can_be_chosen_per_call(rt):
    rt.s.meta_api_key = "key"
    c, sent = MetaClient(rt.s), []
    c.http.post = lambda url, headers=None, json=None, timeout=None: sent.append(json) or R(data=OK)
    c.complete("S", [], "q", effort="minimal")
    c.complete("S", [], "q", effort="none")  # not a real level: the configured one is kept
    assert sent[0]["reasoning"] == {"effort": "minimal"} and sent[1]["reasoning"] == {"effort": rt.s.meta_reasoning_effort}
