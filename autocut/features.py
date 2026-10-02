"""Window features on a fixed time grid, from the raw signals. Same code for training and running."""
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, find_peaks, sosfilt

from .signals import AUDIO_SR, people_lists

STEP = 0.5  # seconds between feature rows


def _win_stats(ts, x, grid, w, kind):
    """mean/std/max of x over a w-second window: centred (c), past (p) or future (f)."""
    lo = {"c": -w / 2, "p": -w, "f": 0}[kind]
    out = np.full((len(grid), 3), np.nan)
    if len(x) == 0:
        return out
    i0 = np.searchsorted(ts, grid + lo)
    i1 = np.searchsorted(ts, grid + lo + w)
    for k, (a, b) in enumerate(zip(i0, i1)):
        seg = x[a:max(b, a + 1)] if a < len(x) else x[-1:]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[k] = (seg.mean(), seg.std(), seg.max())
    return out


def _pick(ps, nearside):
    """Deepest person on one half, within the court width: (depth from net m, x m, count)."""
    c = [(x, y) for x, y in ps if (y > 0) == nearside and abs(x) <= 6.5 and abs(y) > 2]
    if not c:
        return np.nan, np.nan, 0
    x, y = max(c, key=lambda p: abs(p[1]))
    return abs(y), x, len(c)


def _audio_onsets(audio):
    """Impulsiveness of 1-6 kHz sound (ball hits) at 100 Hz: (times, onset ratio, peak indicator)."""
    if audio is None or len(audio) < AUDIO_SR:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    band = sosfilt(butter(4, [1000, 6000], btype="band", fs=AUDIO_SR, output="sos"), audio)
    hop = AUDIO_SR // 100
    env = np.sqrt(uniform_filter1d(band ** 2, hop)[::hop])
    at = np.arange(len(env)) * hop / AUDIO_SR
    on = env / (median_filter(env, size=101) + 1e-6)
    peaks, _ = find_peaks(on, height=4, distance=15)
    imp = np.zeros_like(on)
    imp[peaks] = 1
    return at, on, imp


def build(sig, duration):
    """Returns (grid, X, names)."""
    grid = np.arange(0, duration, STEP)
    cols, names = [], []

    def add(block, prefix, stats=("mean", "std", "max")):
        for j, s in enumerate(("mean", "std", "max")):
            if s in stats:
                cols.append(block[:, j])
                names.append(f"{prefix}_{s}")

    for nm in ("near", "far"):
        for w, kd in ((1, "c"), (3, "c"), (8, "c"), (4, "p"), (4, "f")):
            add(_win_stats(sig["t"], sig[nm], grid, w, kd), f"mot_{nm}_{kd}{w}")

    pt, people = sig["pt"], people_lists(sig)
    per = {}
    for side, ns in (("n", True), ("f", False)):
        arr = np.array([_pick(ps, ns) for ps in people], float).reshape(-1, 3)
        per[f"{side}_depth"], per[f"{side}_x"], per[f"{side}_cnt"] = arr[:, 0], arr[:, 1], arr[:, 2]
        xy = np.stack([arr[:, 1], arr[:, 0]], 1)
        speed = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(pt)] if len(pt) > 1 else np.full(len(pt), np.nan)
        per[f"{side}_speed"] = np.clip(speed, 0, 8)
    per["all_cnt"] = sig["pcount"].astype(float)
    for nm, x in per.items():
        for w, kd in ((2, "c"), (6, "c"), (4, "p"), (4, "f")):
            add(_win_stats(pt, x, grid, w, kd), f"{nm}_{kd}{w}")

    at, on, imp = _audio_onsets(sig.get("audio"))
    for w, kd in ((1, "c"), (3, "c"), (8, "c"), (4, "p"), (4, "f")):
        cols.append(_win_stats(at, imp, grid, w, kd)[:, 0] * w)
        names.append(f"aud_hits_{kd}{w}")
        add(_win_stats(at, on, grid, w, kd), f"aud_on_{kd}{w}")
    return grid, np.stack(cols, 1), names


def targets(grid, keep_spans):
    y = np.zeros(len(grid), int)
    for s, e in keep_spans:
        y[(grid >= s) & (grid <= e)] = 1
    return y
