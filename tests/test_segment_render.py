import subprocess

import numpy as np

from autocut.config import Config
from autocut.evaluate import evaluate
from autocut.render import render
from autocut.segment import merge, segments
from autocut.video import probe


def test_merge_joins_close_spans():
    assert merge([(10, 12), (0, 2), (2.5, 4)], gap=1) == [(0, 4), (10, 12)]


def test_segments_pad_merge_clamp_and_drop_short():
    cfg = Config(seg_on=0.5, seg_off=0.5, seg_min_s=1.0, pad_pre_s=1.5, pad_post_s=1.5, merge_gap_s=2.0)
    t = np.arange(0, 30, 0.5)
    p = np.zeros_like(t)
    p[(t >= 0) & (t <= 3)] = 1      # at the very start -> clamped to 0
    p[(t >= 6) & (t <= 8)] = 1      # overlaps the first after padding -> merged
    p[(t >= 11) & (t <= 13)] = 1
    p[t == 20] = 1                  # 0.5 s blip -> dropped
    segs = segments(t, p, 30.0, cfg)
    assert segs == [(0.0, 14.75)]  # first two overlap after padding, third is within the merge gap


def test_evaluate_perfect_prediction():
    r = evaluate([(8.5, 21.5)], [(10, 20)], [], 100.0, 1.5, 1.5, 2.0)
    assert r["rally_time_kept"] == 1.0 and r["dead_time_cut"] == r["dead_time_cut_perfect"]


def test_render_keeps_audio_and_duration(tmp_path):
    src, dst = str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=20",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=20", "-shortest", "-c:v", "libx264",
                    "-c:a", "aac", src], check=True)
    render(src, [(1.0, 4.0), (10.0, 15.5)], dst)
    info = probe(dst)
    assert info.has_audio
    assert abs(info.duration - 8.5) < 0.15
