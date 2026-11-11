import numpy as np


def test_face_overwrite_and_cascade(rt):
    st = rt.store
    pid = st.add_person("Asha", "hi")
    for _ in range(5):
        st.add_face(pid, np.random.rand(128))
    assert st.prune_faces(pid, 3) == 2 and len(st.gallery()) == 3
    for i in range(15):
        st.add_interaction(pid, "hi", f"u{i}", f"r{i}", "ollama", 10)
    old = st.older_interactions(pid, 10)
    assert [o["user_text"] for o in old] == ["u0", "u1", "u2", "u3", "u4"]  # oldest rows are the ones compressed
    st.delete_person(pid)
    assert st.count_interactions() == 0 and st.gallery() == []


def test_old_faces_purged(rt):
    st = rt.store
    pid = st.add_person("Old")
    st.add_face(pid, np.ones(128))
    st._run("UPDATE faces SET created_at = created_at - ?", (100 * 86400,))
    assert st.purge_old_faces(90) == 1 and st.purge_old_faces(0) == 0


def test_usage_levels(rt):
    rt.s.max_memory_gb = 1e-9  # ~1 byte: the data dir is certainly over the limit
    assert rt.store.usage()["status"] == "full"
    rt.s.max_memory_gb = 5.0
    assert rt.store.usage()["status"] == "ok"


def test_runtime_settings_validation(rt):
    assert rt.s.set_value("greet_cooldown_min", "3") == 3.0
    for key, bad in [("llm_backend", "gpt"), ("face_match_threshold", 5), ("api_key", "x")]:
        try:
            rt.s.set_value(key, bad)
            raise AssertionError("should have been rejected")
        except ValueError:
            pass
