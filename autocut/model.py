"""Play/dead classifier: training (with honest held-out tuning) and prediction."""
import itertools
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone

import joblib
import numpy as np
from scipy.ndimage import median_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import calibrate as cal
from .config import Config
from .evaluate import evaluate
from .features import build, targets
from .label import load_labels
from .segment import merge, segments
from .signals import extract
from .video import probe

DEFAULT_MODEL = os.path.join(cal.ROOT, "models", "autocut_model.joblib")
SEG_GRID = {"seg_on": (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5), "seg_off": (0.05, 0.1, 0.15, 0.25, 0.35),
            "seg_min_s": (0.5, 1.5, 3.0), "smooth": (1, 5, 9)}
MIN_KEEP = 0.95


def classifier():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, l2_regularization=1.0,
                                          min_samples_leaf=40, random_state=0)


def smooth(p, k):
    return median_filter(p, size=k) if k > 1 else p


def load_example(video, court_id, labels_csv, cfg, log=print):
    info = probe(video)
    sig = extract(video, cal.load(court_id), cfg, log)
    grid, X, names = build(sig, info.duration)
    rallies = load_labels(labels_csv, "rally")
    preserve = load_labels(labels_csv, "warmup")  # W marks = pre-serve routine, which is kept
    y = targets(grid, merge(rallies + preserve))
    return {"video": video, "duration": info.duration, "grid": grid, "X": X, "y": y, "names": names,
            "rallies": rallies, "preserve": preserve}


def _held_out_probs(examples):
    """Leave-one-video-out probabilities; with a single video, 5 contiguous time blocks instead."""
    if len(examples) >= 2:
        probs = []
        for i, ex in enumerate(examples):
            rest = [e for j, e in enumerate(examples) if j != i]
            m = classifier().fit(np.vstack([e["X"] for e in rest]), np.concatenate([e["y"] for e in rest]))
            probs.append(m.predict_proba(ex["X"])[:, 1])
        return probs
    ex = examples[0]
    oof = np.zeros(len(ex["y"]))
    idx = np.arange(len(ex["y"]))
    for f in np.array_split(idx, 5):
        tr = np.setdiff1d(idx, f)
        oof[f] = classifier().fit(ex["X"][tr], ex["y"][tr]).predict_proba(ex["X"][f])[:, 1]
    return [oof]


def _score(ex, prob, cfg, k):
    segs = segments(ex["grid"], smooth(prob, k), ex["duration"], cfg)
    return evaluate(segs, ex["rallies"], ex["preserve"], ex["duration"], cfg.pad_pre_s, cfg.pad_post_s,
                    cfg.merge_gap_s)


def train(examples, cfg=None, log=print, min_keep=MIN_KEEP):
    """Pick the segment settings that keep >= min_keep of play on every held-out video and cut the most dead
    time; if none reaches min_keep, the settings that keep the most."""
    cfg = cfg or Config()
    probs = _held_out_probs(examples)
    aucs = [round(float(roc_auc_score(ex["y"], p)), 3) for ex, p in zip(examples, probs)]
    log(f"held-out AUC per video: {aucs}")
    best = None
    for on, off, mn, k in itertools.product(*SEG_GRID.values()):
        if off > on:
            continue
        c = replace(cfg, seg_on=on, seg_off=off, seg_min_s=mn)
        reports = [_score(ex, p, c, k) for ex, p in zip(examples, probs)]
        worst = min(r["keep_time_kept"] for r in reports)
        mean_cut = float(np.mean([r["dead_time_cut"] for r in reports]))
        key = (worst >= min_keep, mean_cut if worst >= min_keep else worst)
        if best is None or key > best[0]:
            best = (key, c, k, reports)
    (met, _), seg_cfg, k, reports = best
    if not met:
        log(f"warning: no setting keeps {min_keep:.0%} of play on every video; using the safest one")
    for ex, r in zip(examples, reports):
        log(f"held-out {os.path.basename(ex['video'])}: rally kept {r['rally_time_kept']:.1%}, "
            f"rally+pre-serve kept {r['keep_time_kept']:.1%}, dead cut {r['dead_time_cut']:.1%} "
            f"(perfect {r['dead_time_cut_perfect']:.1%}), output {r['output_share']:.1%}")
    final = classifier().fit(np.vstack([e["X"] for e in examples]), np.concatenate([e["y"] for e in examples]))
    return {
        "classifier": final,
        "feature_names": examples[0]["names"],
        "seg": {k2: getattr(seg_cfg, k2) for k2 in ("seg_on", "seg_off", "seg_min_s", "pad_pre_s", "pad_post_s",
                                                   "merge_gap_s")},
        "smooth": k,
        "trained_on": [os.path.basename(e["video"]) for e in examples],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "held_out": {"auc": aucs, "reports": reports, "min_keep": min_keep},
        "config": asdict(cfg),
    }


def save(bundle, path=DEFAULT_MODEL):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)


def load(path=DEFAULT_MODEL):
    return joblib.load(path)


def predict(bundle, video, calib, log=print):
    """Returns (segments, info dict)."""
    cfg = Config(**bundle["config"])
    cfg = replace(cfg, **bundle["seg"])
    info = probe(video)
    if not info.has_audio:
        log("warning: video has no audio; the model was trained with audio, expect weaker results")
    sig = extract(video, calib, cfg, log)
    grid, X, names = build(sig, info.duration)
    if names != bundle["feature_names"]:
        raise ValueError("feature set differs from the trained model; retrain with `autocut train`")
    prob = smooth(bundle["classifier"].predict_proba(X)[:, 1], bundle["smooth"])
    return segments(grid, prob, info.duration, cfg), {"duration": info.duration, "has_audio": info.has_audio}
