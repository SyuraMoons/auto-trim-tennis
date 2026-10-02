"""Score predicted keep segments against labels.

Kept = rally + pre-serve routine (labels kind 'rally' and 'warmup'; the user's W marks are the pre-serve routine).
"""
from .segment import merge


def total(spans):
    return sum(e - s for s, e in spans)


def overlap(a, b):
    return sum(max(0.0, min(e, e2) - max(s, s2)) for s, e in a for s2, e2 in b)


def evaluate(pred, rallies, preserve, duration, pad_pre, pad_post, gap):
    keep = merge(list(rallies) + list(preserve))
    dead = duration - total(keep)
    perfect = merge([(max(0, s - pad_pre), min(duration, e + pad_post)) for s, e in keep], gap)
    dead_cut = lambda p: (dead - (total(p) - overlap(p, keep))) / dead
    starts_clipped = []
    for s, e in rallies:
        cover = [p for p in pred if p[0] <= s + 0.01 and p[1] > s]
        if not cover:
            first = min([p[0] for p in pred if p[1] > s and p[0] < e] or [e])
            starts_clipped.append((round(s, 2), round(first - s, 2)))
    return {
        "duration_s": round(duration, 1),
        "output_s": round(total(pred), 1),
        "output_share": round(total(pred) / duration, 3),
        "rally_time_kept": round(overlap(pred, rallies) / total(rallies), 3),
        "keep_time_kept": round(overlap(pred, keep) / total(keep), 3),
        "dead_time_cut": round(dead_cut(pred), 3),
        "dead_time_cut_perfect": round(dead_cut(perfect), 3),
        "rallies": len(rallies),
        "rallies_fully_missed": sum(overlap([r], pred) == 0 for r in rallies),
        "serve_starts_clipped_over_0.5s": [c for c in starts_clipped if c[1] > 0.5],
    }
