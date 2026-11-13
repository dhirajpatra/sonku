import time

import numpy as np

from app.core.orchestrator import Session
from app.vision.presence import Presence, PresenceWatcher
from tests.test_orchestrator import FakeWatcher, stub_llm


def test_known_person_without_wake_word_resumes_earlier_conversation(rt):
    pid = rt.store.add_person("Dheeraj")
    rt.store.add_interaction(pid, "en", "encourage my son to talk", "I will encourage him.", "meta_api", 10)
    rt.watcher = FakeWatcher(Presence(face_present=True, status="known", person_id=pid, name="Dheeraj", seq=1))
    calls = stub_llm(rt)
    assert rt.orch.handle_utterance("do you remember what I asked", "en") == "Sure thing."
    assert rt.orch.session.person_id == pid
    assert calls[0][1][0] == ("user", "encourage my son to talk")  # earlier turn reloaded as history


def test_unknown_face_still_needs_the_wake_word(rt):
    rt.watcher = FakeWatcher(Presence(face_present=True, status="unknown"))
    calls = stub_llm(rt)
    assert rt.orch.handle_utterance("what is the weather", "en") is None and not calls


def test_session_survives_short_absence_but_not_long(rt):
    pid = rt.store.add_person("Asha")
    rt.watcher = FakeWatcher(Presence(face_present=False, status="none", last_face_ts=time.time() - 20))
    rt.orch.session = Session(person_id=pid, name="Asha")
    rt.orch._check_session_expiry()
    assert rt.orch.session is not None
    rt.watcher.p.last_face_ts = time.time() - 100
    rt.orch._check_session_expiry()
    assert rt.orch.session is None


def test_presence_reset_keeps_last_face_time(rt):
    class NoFaces:
        def detect(self, frame):
            return []

    w = PresenceWatcher(rt.s, None, NoFaces(), rt.store)
    seen = time.time() - 20
    w._p = Presence(face_present=True, status="known", person_id=1, name="A", last_face_ts=seen)
    w._step(np.zeros((20, 20, 3), dtype=np.uint8), time.time())
    snap = w.snapshot()
    assert snap.status == "none" and abs(snap.last_face_ts - seen) < 1  # was reset to 0 before the fix


def test_anonymous_conversation_is_adopted_when_face_recognised(rt):
    pid = rt.store.add_person("Asha")
    stub_llm(rt)
    rt.orch.handle_utterance("sonku tell me a joke", "en")
    assert rt.orch.session.person_id is None
    rt.watcher = FakeWatcher(Presence(face_present=True, status="known", person_id=pid, name="Asha", seq=1))
    rt.orch._maybe_greet()
    assert rt.orch.session.person_id == pid and len(rt.orch.session.history) == 2


def test_new_looks_are_learned_up_to_the_limit(rt):
    w = PresenceWatcher(rt.s, None, None, rt.store)
    pid = rt.store.add_person("Asha")
    rt.store.add_face(pid, np.ones(128))
    e = np.ones(128) / np.sqrt(128)
    w._learn_template(pid, e, 0.9, 1000.0)
    assert rt.store.face_count(pid) == 1  # a strong match has nothing new to teach
    w._learn_template(pid, e, 0.45, 1000.0)
    assert rt.store.face_count(pid) == 2
    w._learn_template(pid, e, 0.45, 1010.0)
    assert rt.store.face_count(pid) == 2  # rate limited
    rt.s.faces_per_person = 2
    w._learn_template(pid, e, 0.45, 5000.0)
    assert rt.store.face_count(pid) == 2  # full


def test_summary_written_automatically_and_recent_turns_kept(rt):
    pid = rt.store.add_person("Asha")
    for i in range(35):
        rt.store.add_interaction(pid, "en", f"u{i}", f"r{i}", "m", 1)
    rt.router.summarize = lambda prompt, **k: "Asha likes tea."
    rt.maint.run_once()
    assert rt.store.get_person(pid)["summary"] == "Asha likes tea."
    assert rt.store.count_interactions(pid) == rt.s.summary_keep_recent
    assert rt.store.recent_interactions(pid, 2)[-1]["user_text"] == "u34"  # newest turns survive


def test_small_decision_model_only_asked_when_sentence_points_at_something(rt):
    rt.s.decision_use_llm = True
    asked = []
    rt.decision.ollama.complete = lambda *a, **k: asked.append(1) or "NO"
    assert not rt.decision.needs_vision("tell me the syllabus for fifth semester")
    assert not asked
    rt.decision.needs_vision("is that a good idea today")
    assert asked
