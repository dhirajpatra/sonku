import time

import numpy as np

from app.vision.faces import FaceEngine
from app.vision.presence import PresenceWatcher
from tests.test_social import GridFaces, row, vectors

FRAME = np.zeros((480, 640, 3), np.uint8)  # real size: the 'too small to judge' limit depends on it


def watcher(rt, people=(1, 2)):
    vec = vectors()
    ids = {}
    for i in people:
        ids[i] = rt.store.add_person({1: "Asha", 2: "Ravi", 3: "Meera"}[i])
        rt.store.add_face(ids[i], vec[i])
    f = GridFaces(rt.s)
    f.embs = dict(vec)  # a copy: tests change what the 'camera' sees without touching the stored templates
    return PresenceWatcher(rt.s, None, f, rt.store), f, vec, ids


def run(w, t0, seconds, step=0.7):
    t = t0
    while t < t0 + seconds:
        w._step(FRAME, t)
        t += step
    return t


def test_one_strong_match_is_enough_for_everyone_in_the_frame_at_once(rt):
    w, f, vec, ids = watcher(rt)
    f.rows = [row(1, 120), row(2, 90)]
    w._step(FRAME, 1000.0)  # a single frame
    p = w.snapshot()
    assert [x["name"] for x in p.people] == ["Asha", "Ravi"] and p.seq == 2


def test_a_weaker_match_needs_two_agreeing_samples(rt):
    w, f, vec, ids = watcher(rt, people=(1,))
    orth = vec[3] - np.dot(vec[3], vec[1]) * vec[1]
    orth /= np.linalg.norm(orth)
    f.embs[1] = (0.45 * vec[1] + np.sqrt(1 - 0.45 ** 2) * orth).astype(np.float32)  # cosine 0.45: real but not strong
    f.rows = [row(1, 100)]
    w._step(FRAME, 1000.0)
    assert w.snapshot().people == []  # one sample is not enough
    w._step(FRAME, 1000.7)
    assert [x["name"] for x in w.snapshot().people] == ["Asha"]


def test_new_person_is_recognised_quickly_when_the_old_one_leaves(rt):
    w, f, vec, ids = watcher(rt)
    f.rows = [row(1, 100)]
    t = run(w, 1000.0, 2.0)
    assert [x["name"] for x in w.snapshot().people] == ["Asha"]
    f.rows = []
    t = run(w, t, 2.0)  # Asha walks out; the track is dropped
    assert w.snapshot().people == []
    f.rows = [row(2, 100)]  # Ravi sits in the same spot
    w._step(FRAME, t + 0.1)
    assert [x["name"] for x in w.snapshot().people] == ["Ravi"]  # one frame, no 8-second wait


def test_someone_swapping_into_the_same_spot_without_a_gap_is_noticed(rt):
    w, f, vec, ids = watcher(rt)
    f.rows = [row(1, 100)]
    t = run(w, 1000.0, 2.0)
    f.rows = [row(2, 100)]  # same position, same size, no empty frame between
    t = run(w, t, 5.0)
    assert [x["name"] for x in w.snapshot().people] == ["Ravi"]


def test_far_face_is_counted_but_not_called_a_stranger(rt):
    w, f, vec, ids = watcher(rt, people=(1,))
    f.rows = [row(1, 100), row(3, 55)]  # Asha close, a stranger too small to judge
    run(w, 1000.0, 4.0)
    p = w.snapshot()
    assert p.faces == 2 and [x["name"] for x in p.people] == ["Asha"] and p.unknown == 0
    f.rows = [row(1, 100), row(3, 80)]  # the stranger comes closer
    f.embs[3] = vec[3]
    run(w, 1004.0, 4.0)
    assert w.snapshot().unknown == 1


def test_a_stranger_who_is_really_known_gets_recognised_on_recheck(rt):
    w, f, vec, ids = watcher(rt, people=(1,))
    wrong = vec[3]
    f.embs[1] = wrong  # a bad first impression: the face looks like nobody
    f.rows = [row(1, 100)]
    t = run(w, 1000.0, 3.0)
    assert w.snapshot().unknown == 1
    f.embs[1] = vec[1]  # better view
    run(w, t, 4.0)
    p = w.snapshot()
    assert p.unknown == 0 and [x["name"] for x in p.people] == ["Asha"]


def test_detection_works_on_big_frames_and_scales_the_size_limit(rt):
    class FakeDet:
        def setInputSize(self, size):
            self.size = size

        def detect(self, img):
            self.seen = img.shape[:2]
            return 1, np.array([[100, 50, 60, 60] + [0] * 10 + [0.9], [10, 10, 40, 40] + [0] * 10 + [0.9]], dtype=np.float32)

    eng = FaceEngine(rt.s)
    eng.det = FakeDet()
    rt.s.face_detect_max_side, rt.s.face_min_width_px = 640, 48
    rows = eng.detect(np.zeros((720, 1280, 3), np.uint8))
    assert eng.det.seen == (360, 640)  # searched on a smaller copy
    assert len(rows) == 1 and rows[0][0] == 200 and rows[0][2] == 120  # full-size coordinates; the 80px face is below 48*2
