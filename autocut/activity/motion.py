"""Motion inside the play area: MOG2 background subtraction, foreground share per court half."""
import cv2
import numpy as np

from ..calibrate import regions, scaled_homography
from ..video import iter_frames


def half_masks(calib, cfg, w, h):
    reg = regions(scaled_homography(calib, w, h), cfg)
    masks = {}
    for name in ("near", "far"):
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(reg[name]).astype(np.int32)], 255)
        masks[name] = m
    return masks


def motion_signal(video, calib, cfg):
    """Returns dict of arrays sampled at cfg.analysis_fps: t, near, far (foreground pixel share of each half)."""
    sub = cv2.createBackgroundSubtractorMOG2(history=int(cfg.mog_history_s * cfg.analysis_fps),
                                             varThreshold=cfg.mog_var_threshold, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    masks, areas = None, None
    ts, near, far = [], [], []
    for t, frame in iter_frames(video, cfg.analysis_fps, cfg.analysis_width):
        if masks is None:
            h, w = frame.shape[:2]
            masks = half_masks(calib, cfg, w, h)
            areas = {k: max(1, int((m > 0).sum())) for k, m in masks.items()}
        fg = sub.apply(frame)
        fg = cv2.morphologyEx((fg == 255).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)  # 127 = shadow
        ts.append(t)
        near.append(cv2.countNonZero(cv2.bitwise_and(fg, masks["near"])) / areas["near"])
        far.append(cv2.countNonZero(cv2.bitwise_and(fg, masks["far"])) / areas["far"])
    return {"t": np.array(ts), "near": np.array(near), "far": np.array(far)}
