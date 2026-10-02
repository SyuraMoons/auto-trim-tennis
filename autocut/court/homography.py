# Adapted from AymenMB/Tennis-Match-Analysis- (homography.py get_trans_matrix): same RANSAC search over
# the 12 four-point court configurations, scored by median reprojection error of the remaining points.
import cv2
import numpy as np

from .reference import CourtReference

court_ref = CourtReference()
refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))
court_conf_ind = {k: [court_ref.key_points.index(p) for p in conf] for k, conf in court_ref.court_conf.items()}


def get_trans_matrix(points):
    """Best reference->image homography from 14 keypoints (None for missing).

    Returns (matrix, median_error_px); matrix is None if no configuration has all 4 points.
    """
    matrix_trans, dist_best = None, np.inf
    for conf_ind, conf in court_ref.court_conf.items():
        inds = court_conf_ind[conf_ind]
        required = [points[i] for i in inds]
        if any(p is None for p in required):
            continue
        matrix, _ = cv2.findHomography(np.float32(conf), np.float32(required), cv2.RANSAC, 5.0)
        if matrix is None:
            continue
        trans_kps = cv2.perspectiveTransform(refer_kps, matrix).squeeze(1)
        dists = [np.linalg.norm(np.subtract(points[i], trans_kps[i]))
                 for i in range(len(points)) if points[i] is not None and i not in inds]
        dist = float(np.median(dists)) if dists else np.inf
        if matrix_trans is None or dist < dist_best:
            matrix_trans, dist_best = matrix, dist
    return matrix_trans, dist_best


def project(matrix, pts):
    """Apply a 3x3 homography to a list of (x, y)."""
    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, matrix).reshape(-1, 2)
