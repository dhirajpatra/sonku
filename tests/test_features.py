import json

import cv2
import numpy as np

from app.core import intents as I
from app.core.orchestrator import Session
from app.llm import prompts
from app.llm.meta_client import MetaClient
from app.llm.ollama_client import OllamaClient
from app.utils.text import drop_unfinished, split_long
from app.vision.camera import Camera
from app.vision.presence import Presence
from tests.test_orchestrator import FakeWatcher, stub_llm


# ---------------------------------------------------------------- listening
def test_unfinished_sentences_are_recognised():
    for text in ["my son is very shy and", "wait let me think", "I want to talk about the", "hold on", "he does not talk to"]:
        assert I.looks_incomplete(text), text
    for text in ["tell me a joke", "I don't think so", "what is two plus two", "he is very shy."]:
        assert not I.looks_incomplete(text), text


class FakeRecorder:
    last_peak = last_thr = 0

    def __init__(self, clips):
        self.clips, self.calls = list(clips), []

    def listen(self, wait_sec=2.0, max_sec=None):
        self.calls.append((wait_sec, max_sec))
        return self.clips.pop(0) if self.clips else None


class FakeSTT:
    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, audio, hint=None):
        return self.texts.pop(0), "en"


def test_waits_for_the_rest_of_an_unfinished_sentence(rt):
    clip = (np.ones(16000) * 1000).astype(np.int16)
    rt.recorder = FakeRecorder([clip, clip])
    rt.stt = FakeSTT(["my son does not talk to other kids and", "my son does not talk to other kids and he is very shy"])
    rt.orch.session = Session()
    calls = stub_llm(rt)
    rt.orch._tick()
    assert calls[0][2] == "my son does not talk to other kids and he is very shy"  # one turn, whole sentence
    assert rt.recorder.calls[1][0] == rt.s.patience_sec  # second listen used the long patience window


def test_finished_sentence_is_not_delayed(rt):
    clip = (np.ones(16000) * 1000).astype(np.int16)
    rt.recorder = FakeRecorder([clip])
    rt.stt = FakeSTT(["tell me a joke"])
    rt.orch.session = Session()
    stub_llm(rt)
    rt.orch._tick()
    assert len(rt.recorder.calls) == 1


# ---------------------------------------------------------------- full-length answers
def test_cut_off_replies_never_end_mid_sentence():
    assert drop_unfinished("One. Two. Thr") == "One. Two."
    assert drop_unfinished("Only a fragment") == "Only a fragment"
    assert drop_unfinished("Done.") == "Done."


def test_long_sentences_are_split_for_speech(rt):
    long = ("This is a long clause, " * 30).strip() + "."
    parts = split_long(long)
    assert len(parts) > 1 and all(len(p) <= 180 for p in parts) and " ".join(parts).split() == long.split()
    from app.audio.tts import TTS
    tts, seen = TTS(rt.s), []
    tts._synth = lambda text, lang: seen.append(text)  # returns None: nothing to play
    assert tts.speak_stream([long], "en") == long and seen == parts


class R:
    def __init__(self, status=200, data=None, text=""):
        self.status_code, self._d, self.text = status, data or {}, text

    def json(self):
        return self._d


OK = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello there."}]}],
      "usage": {"input_tokens": 10, "output_tokens": 5}}


def meta(rt, responses):
    rt.s.meta_api_key = "key"
    c, sent = MetaClient(rt.s), []
    queue = list(responses)

    def post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        return queue.pop(0)

    c.http.post = post
    return c, sent


def test_meta_uses_developer_message_and_image_block(rt):
    c, sent = meta(rt, [R(data=OK)])
    assert c.complete("RULES", [("user", "hi"), ("assistant", "hello")], "what is this", image_b64="AAAA") == "Hello there."
    msgs = sent[0]["input"]
    assert [m["role"] for m in msgs] == ["developer", "user", "assistant", "user"] and msgs[0]["content"] == "RULES"
    last = msgs[-1]["content"]
    assert [b["type"] for b in last] == ["input_text", "input_image"] and last[1]["image_url"] == "data:image/jpeg;base64,AAAA"
    assert sent[0]["reasoning"] == {"effort": rt.s.meta_reasoning_effort}


def test_meta_falls_back_to_one_text_prompt_if_layout_rejected(rt):
    c, sent = meta(rt, [R(400, text="bad input"), R(data=OK), R(data=OK)])
    assert c.complete("RULES", [("user", "hi"), ("assistant", "hello")], "next") == "Hello there."
    assert len(sent[1]["input"]) == 1 and "Conversation so far:" in sent[1]["input"][0]["content"][0]["text"]
    c.complete("RULES", [], "again")
    assert len(sent) == 3 and len(sent[2]["input"]) == 1  # remembered: no second failed attempt


def test_meta_reply_cut_by_token_limit_drops_the_fragment(rt):
    cut = dict(OK, status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    cut["output"] = [{"type": "message", "content": [{"type": "output_text", "text": "First sentence. Second sent"}]}]
    c, _ = meta(rt, [R(data=cut)])
    assert c.complete("S", [], "q") == "First sentence." and c.totals["truncated"] == 1


def test_ollama_reply_cut_by_token_limit_drops_the_fragment(rt):
    class Stream:
        status_code = 200

        def __init__(self, reason):
            self.lines = [json.dumps({"message": {"content": "First sentence. Second sen"}, "done": False}),
                          json.dumps({"message": {"content": ""}, "done": True, "done_reason": reason})]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_lines(self):
            return iter(self.lines)

    c = OllamaClient(rt.s)
    c.http.post = lambda *a, **k: Stream("length")
    assert list(c.stream("s", [], "q")) == ["First sentence."]
    c.http.post = lambda *a, **k: Stream("stop")
    assert list(c.stream("s", [], "q")) == ["First sentence.", "Second sen"]


def test_reasoning_effort_only_accepts_real_levels(monkeypatch):
    from app.config import Settings
    for given, expected in [("none", "low"), ("HIGH", "high"), ("minimal", "minimal"), ("turbo", "low")]:
        monkeypatch.setenv("META_REASONING_EFFORT", given)
        assert Settings.from_env().meta_reasoning_effort == expected


# ---------------------------------------------------------------- images
def test_photo_is_shrunk_and_prompt_mentions_it(rt):
    ok, buf = cv2.imencode(".jpg", np.zeros((10, 10, 3), np.uint8))
    big = np.zeros((1000, 2000, 3), np.uint8)
    out = cv2.imdecode(np.frombuffer(Camera(rt.s).jpeg(big, max_side=1024), np.uint8), cv2.IMREAD_COLOR)
    assert out.shape[:2] == (512, 1024)
    small = cv2.imdecode(np.frombuffer(Camera(rt.s).jpeg(np.zeros((480, 640, 3), np.uint8), max_side=1024), np.uint8), cv2.IMREAD_COLOR)
    assert small.shape[:2] == (480, 640)  # never upscaled
    assert "fresh photo" in prompts.system_prompt("en", None, "", has_image=True)
    assert "fresh photo" not in prompts.system_prompt("en", None, "")


def test_vision_turn_system_prompt_mentions_the_attached_photo(rt):
    class Cam:
        def jpeg_b64(self):
            return "IMG"

    rt.camera = Cam()
    calls = stub_llm(rt)
    rt.orch.handle_utterance("sonku what am I holding", "en")
    assert "fresh photo" in calls[0][0]


# ---------------------------------------------------------------- person cache
def test_cache_keeps_at_least_three_people_and_evicts_least_recent(rt):
    rt.s.person_cache_size = 1  # asking for less than the minimum still gives 3
    ids = [rt.store.add_person(n) for n in "ABCD"]
    for pid in ids[:3]:
        rt.cache.get(pid)
    assert rt.cache.capacity == 3 and rt.cache.stats()["size"] == 3
    rt.cache.get(ids[0])  # touch A: B is now the least recently used
    rt.cache.get(ids[3])  # D arrives, B leaves
    st = rt.cache.stats()
    assert set(st["people"]) == {"A", "C", "D"} and st["hits"] == 1 and st["misses"] == 4


def test_cache_refreshes_when_the_store_changes(rt):
    pid = rt.store.add_person("Asha")
    rt.store.set_summary(pid, "Likes tea. " * 100)
    e = rt.cache.get(pid)
    assert len(e.summary) <= 600 and e.summary.endswith(".")  # compact, cut at a sentence end
    rt.store.set_summary(pid, "Likes coffee.")
    assert rt.cache.get(pid).summary == "Likes coffee."
    rt.store.add_interaction(pid, "en", "hi", "hello", "m", 1)
    assert rt.cache.get(pid).history == [("user", "hi"), ("assistant", "hello")]
    rt.store.rename_person(pid, "Asha P")
    assert rt.cache.get(pid).name == "Asha P"
    rt.store.delete_person(pid)
    assert rt.cache.get(pid) is None


def test_cache_fast_path_recognises_hot_people_only(rt):
    rng = np.random.RandomState(1)
    emb, other = rng.randn(128).astype(np.float32), rng.randn(128).astype(np.float32)
    emb, other = emb / np.linalg.norm(emb), other / np.linalg.norm(other)
    pid = rt.store.add_person("Asha")
    rt.store.add_face(pid, emb)
    assert rt.cache.match(emb)[0] is None  # not hot yet: the full gallery handles it
    rt.cache.get(pid)
    assert rt.cache.match(emb)[0] == pid and rt.cache.match(other)[0] is None


def test_cache_is_warmed_with_recent_people(rt):
    ids = [rt.store.add_person(n) for n in "ABCDE"]
    rt.store.touch_seen(ids[1])
    rt.cache.warm()
    assert rt.cache.stats()["size"] == 5 and rt.cache.stats()["people"][0] == "B"  # most recently seen first


def test_summary_reaches_the_prompt_and_stays_fresh(rt):
    pid = rt.store.add_person("Asha")
    rt.store.set_summary(pid, "Asha likes tea.")
    rt.watcher = FakeWatcher(Presence(face_present=True, status="known", person_id=pid, name="Asha", seq=1))
    calls = stub_llm(rt)
    rt.orch.handle_utterance("tell me a joke", "en")
    assert "Asha likes tea." in calls[0][0]
    rt.store.set_summary(pid, "Asha likes coffee.")
    rt.orch.handle_utterance("another one", "en")
    assert "Asha likes coffee." in calls[1][0]
