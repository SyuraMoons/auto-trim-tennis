"""Per-court calibration: detect once, save to courts/<court_id>.json, reuse for every video from that camera."""
import json
import os
from datetime import datetime, timezone

import cv2
import numpy as np

from .court.homography import project
from .court.reference import CourtReference

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COURTS_DIR = os.path.join(ROOT, "courts")
WEIGHTS = os.path.join(ROOT, "weights", "court.pt")
ref = CourtReference()


def court_path(court_id):
    return os.path.join(COURTS_DIR, f"{court_id}.json")


def save(court_id, data):
    os.makedirs(COURTS_DIR, exist_ok=True)
    data = {"court_id": court_id, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), **data}
    with open(court_path(court_id), "w") as f:
        json.dump(data, f, indent=2)
    return court_path(court_id)


def load(court_id):
    with open(court_path(court_id)) as f:
        return json.load(f)


def manual_homography(corners):
    """corners: 4 image points of the doubles court: far-left, far-right, near-right, near-left."""
    matrix, _ = cv2.findHomography(np.float32(ref.court_corners()), np.float32(corners))
    return matrix


def scaled_homography(calib, frame_w, frame_h):
    """Calibration homography rescaled to a frame of a different size (e.g. the 640 px analysis frames)."""
    w, h = calib["frame_size"]
    s = np.diag([frame_w / w, frame_h / h, 1.0])
    return s @ np.array(calib["homography"])


def regions(matrix, cfg):
    """Image-space polygons for the court, the play area and its near/far halves (split at the net)."""
    play = ref.play_area(cfg.play_behind_m, cfg.play_side_m)
    net_y = ref.net[0][1]
    far = [play[0], play[1], (play[1][0], net_y), (play[0][0], net_y)]
    near = [(play[3][0], net_y), (play[2][0], net_y), play[2], play[3]]
    return {name: project(matrix, pts) for name, pts in
            {"court": ref.court_corners(), "play": play, "far": far, "near": near}.items()}


def draw_overlay(frame, matrix, cfg, keypoints=None):
    img = frame.copy()
    reg = regions(matrix, cfg)
    for poly, color in ((reg["far"], (255, 160, 0)), (reg["near"], (0, 200, 255))):
        layer = img.copy()
        cv2.fillPoly(layer, [poly.astype(np.int32)], color)
        img = cv2.addWeighted(layer, 0.18, img, 0.82, 0)
    cv2.polylines(img, [reg["play"].astype(np.int32)], True, (0, 255, 255), 2)
    for line in ref.lines:
        p = project(matrix, line).astype(np.int32)
        cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 255, 0), 2)
    for i, kp in enumerate(keypoints or []):
        if kp is not None:
            cv2.circle(img, (int(kp[0]), int(kp[1])), 7, (0, 0, 255), -1)
            cv2.putText(img, str(i), (int(kp[0]) + 8, int(kp[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return img
