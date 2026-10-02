import numpy as np

from autocut import model as M
from autocut.features import build, targets


def _fake_signals(duration, rng, play):
    t = np.arange(0, duration, 0.1)
    pt = np.arange(0, duration, 0.5)
    active = np.array([any(s <= x <= e for s, e in play) for x in pt])
    counts = np.full(len(pt), 2)
    xy = []
    for a in active:
        depth = 12.0 if a else 8.0
        xy += [(rng.normal(0, 1), depth + rng.normal(0, .3)), (rng.normal(0, 1), -depth + rng.normal(0, .3))]
    audio = rng.normal(0, 0.01, int(duration * 16000)).astype(np.float32)
    return {"t": t, "near": rng.random(len(t)) * 0.01, "far": rng.random(len(t)) * 0.01, "pt": pt,
            "pxy": np.float32(xy), "pcount": counts.astype(np.int32), "audio": audio}


def test_feature_names_stable_and_shapes():
    rng = np.random.default_rng(0)
    g1, X1, n1 = build(_fake_signals(60, rng, [(10, 20)]), 60.0)
    g2, X2, n2 = build(_fake_signals(90, rng, []), 90.0)
    assert n1 == n2 and X1.shape == (len(g1), len(n1)) and X2.shape[1] == len(n1)
    assert targets(g1, [(10, 20)]).sum() == 21


def test_train_learns_a_separable_signal():
    rng = np.random.default_rng(1)
    examples = []
    for play in ([(10, 25), (40, 55), (70, 85)], [(5, 20), (35, 50), (65, 80)]):
        g, X, names = build(_fake_signals(100, rng, play), 100.0)
        examples.append({"video": "v.mp4", "duration": 100.0, "grid": g, "X": X, "y": targets(g, play),
                         "names": names, "rallies": play, "preserve": []})
    bundle = M.train(examples, log=lambda *a: None)
    assert min(bundle["held_out"]["auc"]) > 0.9
    assert all(r["keep_time_kept"] >= 0.95 for r in bundle["held_out"]["reports"])
