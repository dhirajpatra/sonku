import base64

import pytest
from fastapi.testclient import TestClient

from app.llm.meta_client import MetaError
from app.llm.router import LLMUnavailable
from app.web.server import create_app

AUTH = ("admin", "secret")
H = {"X-Sonku": "1"}


def test_meta_failure_falls_back_to_ollama_then_breaker(rt):
    rt.s.llm_backend = "meta_api"

    def meta_fail(*a, **k):
        raise MetaError("timeout")

    rt.router.meta.complete = meta_fail
    rt.router.ollama.stream = lambda *a, **k: iter(["From ollama."])
    assert list(rt.router.reply_stream("sys", [], "hi")) == ["From ollama."]
    assert rt.router.last_backend == "ollama"
    list(rt.router.reply_stream("sys", [], "hi"))
    assert rt.router.status()["meta_skipped_for_sec"] > 0  # breaker open after two failures


def test_meta_success_is_split_into_sentences(rt):
    rt.s.llm_backend = "meta_api"
    rt.router.meta.complete = lambda *a, **k: "Hello there, friend. How are you today?"
    assert list(rt.router.reply_stream("s", [], "hi")) == ["Hello there, friend.", "How are you today?"]
    assert rt.router.last_backend == "meta_api"


def test_both_backends_down_raises(rt):
    from app.llm.ollama_client import OllamaError

    rt.s.llm_backend = "ollama"

    def boom(*a, **k):
        raise OllamaError("refused")
        yield

    rt.router.ollama.stream = boom
    with pytest.raises(LLMUnavailable):
        list(rt.router.reply_stream("s", [], "hi"))


@pytest.fixture()
def client(rt):
    with TestClient(create_app(rt)) as c:
        yield c


def test_auth_and_csrf(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", auth=("admin", "wrong")).status_code == 401
    assert client.get("/", auth=AUTH).status_code == 200
    assert client.delete("/api/people?confirm=yes", auth=AUTH).status_code == 403  # no X-Sonku header


def test_people_memory_settings_flow(client, rt):
    pid = rt.store.add_person("Asha")
    rt.store.add_interaction(pid, "en", "hi", "hello", "stub", 5)
    people = client.get("/api/people", auth=AUTH).json()
    assert people[0]["name"] == "Asha" and people[0]["interaction_count"] == 1
    assert client.patch(f"/api/people/{pid}", auth=AUTH, headers=H, json={"name": "Asha P"}).status_code == 200
    mem = client.get("/api/memory", auth=AUTH).json()
    assert mem["status"] == "ok" and mem["people"] == 1
    r = client.put("/api/settings", auth=AUTH, headers=H, json={"greet_cooldown_min": 2, "default_voice": "satvik"})
    assert r.status_code == 200 and rt.s.default_voice == "satvik" and rt.store.get_settings()["greet_cooldown_min"] == "2.0"
    assert client.put("/api/settings", auth=AUTH, headers=H, json={"meta_api_key": "x"}).status_code == 422
    assert client.delete("/api/people?confirm=yes", auth=AUTH, headers=H).json()["deleted"] == 1
    assert client.get("/api/people", auth=AUTH).json() == []


def test_chat_logs_and_status(client, rt):
    rt.router.reply_stream = lambda *a, **k: iter(["Pong."])
    out = client.post("/api/chat", auth=AUTH, headers=H, json={"text": "ping"}).json()
    assert out["reply"] == "Pong."
    st = client.get("/api/status", auth=AUTH).json()
    assert st["health"]["camera"] == "headless" and "memory" in st
    assert client.post("/api/memory/maintenance", auth=AUTH, headers=H).status_code == 200
    assert client.get("/api/camera/snapshot", auth=AUTH).status_code == 503


def test_settings_persist_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SONKU_HEADLESS", "true")
    from app.config import Settings
    from app.runtime import Runtime

    r1 = Runtime(Settings.from_env())
    r1.store.save_setting("default_voice", "satvik")
    r1.store.close()
    r2 = Runtime(Settings.from_env())
    r2.start()
    assert r2.s.default_voice == "satvik"
    r2.stop()
