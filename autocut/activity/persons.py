"""People standing in the play area, using torchvision Faster R-CNN (BSD-3 code, COCO weights).

The detector sees a full-resolution crop of the play area so the far player (~50 px tall at 1080p) is not lost.
Each person is placed on the court by their feet (bottom-centre of the box) through the inverse homography.
"""
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights

from ..calibrate import regions
from ..court.detect import pick_device
from ..court.homography import project
from ..court.reference import CourtReference, UNITS_PER_M

PERSON = 1  # COCO label id
ref = CourtReference()


class PersonDetector:
    def __init__(self, calib, cfg, device=None):
        self.cfg = cfg
        self.device = device or pick_device()
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights, box_score_thresh=cfg.person_score).to(self.device).eval()
        self.h = np.array(calib["homography"])
        self.h_inv = np.linalg.inv(self.h)
        w, hgt = calib["frame_size"]
        play = regions(self.h, cfg)["play"]
        # crop: play area bounding box, plus headroom above the far edge (people stand above their feet)
        x0, y0 = np.clip(play.min(axis=0), 0, [w, hgt]).astype(int)
        x1, y1 = np.clip(play.max(axis=0), 0, [w, hgt]).astype(int)
        self.crop = (x0, max(0, y0 - cfg.person_headroom_px), x1, y1)
        self.play_ref = np.float32(ref.play_area(cfg.play_behind_m, cfg.play_side_m))

    @torch.no_grad()
    def detect(self, frame):
        """List of (x_m, y_m) court positions in metres (origin = court centre, +y towards the near baseline)."""
        x0, y0, x1, y1 = self.crop
        img = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        inp = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(self.device)
        out = self.model([inp])[0]
        keep = (out["labels"] == PERSON).cpu().numpy()
        boxes = out["boxes"].cpu().numpy()[keep]
        if not len(boxes):
            return []
        feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2 + x0, boxes[:, 3] + y0], axis=1)
        court = project(self.h_inv, feet)
        (px0, py0), (px1, py1) = self.play_ref[0], self.play_ref[2]
        inside = (court[:, 0] >= px0) & (court[:, 0] <= px1) & (court[:, 1] >= py0) & (court[:, 1] <= py1)
        cx, cy = (ref.baseline_top[0][0] + ref.baseline_top[1][0]) / 2, ref.net[0][1]
        return [(float(x - cx) / UNITS_PER_M, float(y - cy) / UNITS_PER_M) for x, y in court[inside]]
