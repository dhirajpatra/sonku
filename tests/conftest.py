import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def rt(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SONKU_HEADLESS", "true")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.setenv("DECISION_USE_LLM", "false")
    from app.config import Settings
    from app.runtime import Runtime

    runtime = Runtime(Settings.from_env())
    yield runtime
    runtime.store.close()
