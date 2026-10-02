"""Per-sample play probability -> padded, merged keep segments."""
import numpy as np


def merge(spans, gap=0.0):
    out = []
    for s, e in sorted(spans):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def runs(t, mask):
    """Contiguous True runs of mask as (start_t, end_t); each sample covers half a step either side."""
    if len(t) == 0:
        return []
    step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    spans, start = [], None
    for ti, m in zip(t, mask):
        if m and start is None:
            start = ti
        elif not m and start is not None:
            spans.append((start - step / 2, prev + step / 2))
            start = None
        prev = ti
    if start is not None:
        spans.append((start - step / 2, prev + step / 2))
    return spans


def segments(t, prob, duration, cfg):
    """Hysteresis threshold -> drop short -> pad -> merge, clamped to [0, duration]."""
    on = np.zeros(len(prob), bool)
    active = False
    for i, p in enumerate(prob):
        active = p >= cfg.seg_on if not active else p >= cfg.seg_off
        on[i] = active
    spans = [(s, e) for s, e in runs(t, on) if e - s >= cfg.seg_min_s]
    spans = [(max(0.0, s - cfg.pad_pre_s), min(duration, e + cfg.pad_post_s)) for s, e in spans]
    return merge(spans, cfg.merge_gap_s)
