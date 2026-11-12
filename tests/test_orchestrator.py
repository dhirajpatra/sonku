import numpy as np

from app.vision.presence import Presence


class FakeWatcher:
    def __init__(self, presence):
        self.p = presence
        self.reset = False

    def snapshot(self):
        return self.p

    def capture_embeddings(self, n=3):
        return [np.ones(128, dtype=np.float32) / np.sqrt(128) for _ in range(n)]

    def reset_identity(self):
        self.reset = True


class FakeFaces:
    def match(self, emb, gallery):
        return (gallery[0][0], 0.9) if gallery else (None, 0.0)


def stub_llm(rt, reply="Sure thing."):
    calls = []

    def fake(system, history, text, image=None):
        calls.append((system, list(history), text, image))
        rt.router.last_backend = "stub"
        yield reply

    rt.router.reply_stream = fake
    return calls


def test_ignored_without_wake_word_or_session(rt):
    calls = stub_llm(rt)
    assert rt.orch.handle_utterance("what is the weather", "en") is None
    assert rt.orch.session is None and not calls


def test_wake_word_with_question_answers_and_stores(rt):
    calls = stub_llm(rt)
    assert rt.orch.handle_utterance("hi sonku what is two plus two", "en") == "Sure thing."
    assert calls[0][2] == "what is two plus two"
    assert rt.store.count_interactions() == 1 and rt.orch.session is not None
    rt.orch.handle_utterance("and three plus three", "en")  # follow-up needs no wake word
    assert len(calls[1][1]) == 2  # previous turn passed as history


def test_unknown_face_enrolment_with_consent(rt):
    rt.watcher = FakeWatcher(Presence(face_present=True, status="unknown"))
    rt.faces = FakeFaces()
    stub_llm(rt)
    rt.orch.handle_utterance("hi sonku", "en")
    assert rt.orch.session.stage == "await_name"
    rt.orch.handle_utterance("my name is Asha", "en")
    assert rt.orch.session.stage == "await_consent" and rt.store.list_people() == []
    rt.orch.handle_utterance("yes please", "en")
    people = rt.store.list_people()
    assert [p["name"] for p in people] == ["Asha"] and people[0]["face_count"] == 3
    assert rt.orch.session.person_id == people[0]["id"]


def test_declined_consent_stores_nothing(rt):
    rt.watcher = FakeWatcher(Presence(face_present=True, status="unknown"))
    rt.faces = FakeFaces()
    stub_llm(rt)
    rt.orch.handle_utterance("sonku", "en")
    rt.orch.handle_utterance("I'm Ravi", "en")
    rt.orch.handle_utterance("no", "en")
    assert rt.store.list_people() == [] and rt.orch.session.stage == "chat"


def test_known_person_greeted_once_without_wake_word(rt):
    pid = rt.store.add_person("Asha", "hi")
    rt.watcher = FakeWatcher(Presence(face_present=True, status="known", person_id=pid, name="Asha", seq=1))
    rt.orch._maybe_greet()
    assert rt.orch.session and rt.orch.session.name == "Asha"
    assert rt.store.get_person(pid)["last_greeted"] is not None
    rt.orch._end_session()
    rt.orch._maybe_greet()  # same arrival (same seq): no second greeting
    assert rt.orch.session is None


def test_forget_me_deletes_everything(rt):
    pid = rt.store.add_person("Asha")
    rt.store.add_face(pid, np.ones(128))
    rt.watcher = FakeWatcher(Presence(face_present=True, status="known", person_id=pid, name="Asha", seq=1))
    stub_llm(rt)
    rt.orch.handle_utterance("sonku how are you", "en")
    rt.orch.handle_utterance("please forget me", "en")
    assert rt.store.list_people() == [] and rt.store.gallery() == [] and rt.store.count_interactions() == 0
    assert rt.watcher.reset and rt.orch.session is None


def test_vision_only_when_asked(rt):
    class Cam:
        def jpeg_b64(self):
            return "IMG"

    rt.camera = Cam()
    calls = stub_llm(rt)
    rt.orch.handle_utterance("sonku what am I holding", "en")
    rt.orch.handle_utterance("tell me a joke", "en")
    assert calls[0][3] == "IMG" and calls[1][3] is None


def test_llm_failure_speaks_fallback(rt):
    from app.llm.router import LLMUnavailable

    def broken(*a, **k):
        raise LLMUnavailable("down")
        yield  # pragma: no cover

    rt.router.reply_stream = broken
    reply = rt.orch.handle_utterance("sonku hello there friend", "en")
    assert "can't reach my brain" in reply


def test_explicit_request_to_remember_skips_consent_question(rt):
    rt.watcher = FakeWatcher(Presence(face_present=True, status="unknown"))
    rt.faces = FakeFaces()
    stub_llm(rt)
    rt.orch.handle_utterance("hi sonku", "en")
    rt.orch.handle_utterance("my name is Dheeraj and please remember my face", "en")
    people = rt.store.list_people()
    assert [p["name"] for p in people] == ["Dheeraj"] and people[0]["face_count"] == 3
    assert rt.orch.session.stage == "chat"


def test_consent_question_gets_extra_time_and_unclear_answer_is_retried(rt):
    import time
    rt.watcher = FakeWatcher(Presence(face_present=True, status="unknown"))
    rt.faces = FakeFaces()
    stub_llm(rt)
    rt.orch.handle_utterance("sonku", "en")
    rt.orch.handle_utterance("I'm Ravi", "en")
    sess = rt.orch.session
    sess.last_activity = time.time() - (rt.s.session_timeout_sec + 10)  # past the chat timeout, inside the doubled one
    rt.orch._check_session_expiry()
    assert rt.orch.session is sess
    rt.orch.handle_utterance("has it taken", "en")
    assert rt.orch.session.stage == "await_consent" and rt.store.list_people() == []
    rt.orch.handle_utterance("yes", "en")
    assert [p["name"] for p in rt.store.list_people()] == ["Ravi"]


def test_take_my_photo_counts_as_yes_at_consent_stage(rt):
    rt.watcher = FakeWatcher(Presence(face_present=True, status="unknown"))
    rt.faces = FakeFaces()
    stub_llm(rt)
    rt.orch.handle_utterance("sonku", "en")
    rt.orch.handle_utterance("I'm Meera", "en")
    assert rt.orch.session.stage == "await_consent"
    rt.orch.handle_utterance("yes, remember me", "en")
    assert [p["name"] for p in rt.store.list_people()] == ["Meera"]
