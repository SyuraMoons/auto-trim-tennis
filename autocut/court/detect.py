"""Court keypoints from a handful of frames, using the TennisCourtDetector network (via Tennis-Match-Analysis).

The model works at 640x360 and its heatmap peaks are scaled x2, so frames are first resized to 1280x720;
the resulting homography is then rescaled to the native frame size.
"""
import cv2
import numpy as np
import torch

from .homography import court_ref, get_trans_matrix, project
from .refine import refine_kp
from .track_net import Track

WORK_W, WORK_H = 1280, 720
NO_REFINE = {8, 9, 12}   # same exclusions as the original infer_model


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class CourtDetector:
    def __init__(self, weights, device=None):
        self.device = device or pick_device()
        self.model = Track(out_channels=15)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def keypoints(self, frame, conf=0.5):
        """14 keypoints (x, y) in 1280x720 coordinates, None where not found."""
        img = cv2.resize(frame, (WORK_W, WORK_H))
        inp = cv2.resize(img, (WORK_W // 2, WORK_H // 2)).astype(np.float32) / 255.0
        inp = torch.from_numpy(np.rollaxis(inp, 2, 0)).unsqueeze(0).to(self.device)
        pred = torch.sigmoid(self.model(inp)[0]).cpu().numpy()
        points = []
        for k in range(14):
            heat = pred[k]
            if heat.max() <= conf:
                points.append(None)
                continue
            _, thresh = cv2.threshold((heat * 255).astype(np.uint8), 170, 255, cv2.THRESH_BINARY)
            circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2,
                                       minRadius=10, maxRadius=25)
            if circles is None:
                points.append(None)
                continue
            x, y = circles[0][0][0] * 2, circles[0][0][1] * 2
            if k not in NO_REFINE:
                x, y = refine_kp(img, x, y)
            points.append((float(x), float(y)))
        return points


def _white_ridges(img):
    """Response image for thin bright lines on a darker court (works on both blue and grey surfaces)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))


def _ref_lines():
    """Court lines used for fitting (the net is skipped: its tape is above the ground plane)."""
    r = court_ref
    return [r.baseline_top, r.baseline_bottom, r.left_court_line, r.right_court_line,
            r.left_inner_line, r.right_inner_line, r.top_inner_line, r.bottom_inner_line, r.middle_line]


def line_hits(ridges, matrix, cfg):
    """For each reference line, walk along its projection and take the white-ridge peak along the normal.

    Returns a list of (ref_line, Nx2 image points)."""
    h, w = ridges.shape
    out = []
    radius = int(cfg.snap_max_px)
    offsets = np.arange(-radius, radius + 1)
    weight = np.exp(-(offsets ** 2) / (2 * (radius / 2) ** 2))
    for line in _ref_lines():
        (x0, y0), (x1, y1) = line
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / 20))
        ts = np.linspace(0.02, 0.98, n)
        ref_pts = np.stack([x0 + ts * (x1 - x0), y0 + ts * (y1 - y0)], axis=1)
        img_pts = project(matrix, ref_pts)
        d = np.gradient(img_pts, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        normals = np.stack([-d[:, 1], d[:, 0]], axis=1)
        hits = []
        for p, nv in zip(img_pts, normals):
            xs = np.round(p[0] + offsets * nv[0]).astype(int)
            ys = np.round(p[1] + offsets * nv[1]).astype(int)
            ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            if ok.sum() < len(offsets) // 2:
                continue
            prof = np.zeros(len(offsets))
            prof[ok] = ridges[ys[ok], xs[ok]]
            k = int(np.argmax(prof * weight))
            if prof[k] >= cfg.ridge_min:
                hits.append((xs[k], ys[k]))
        if len(hits) >= 6:
            out.append((line, np.float32(hits)))
    return out


def _fit_image_line(pts):
    """Robust 2D line fit; returns (point, direction) or None."""
    pts = np.float32(pts)
    for _ in range(2):
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        res = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        keep = res < max(2.0, 2.5 * np.median(res))
        if keep.sum() < 6:
            return None
        pts = pts[keep]
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect(l1, l2):
    (p, d), (q, e) = l1, l2
    a = np.array([d, -e]).T
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    t = np.linalg.solve(a, q - p)
    return p + t[0] * d


def snap_refit(background, matrix, cfg):
    """Fit every court line to the real white line, intersect the fitted lines at each court crossing, refit.

    Returns (matrix, median line residual px, n crossings) or None."""
    ridges = _white_ridges(background)
    fitted = {}
    for line, pts in line_hits(ridges, matrix, cfg):
        f = _fit_image_line(pts)
        if f is not None:
            fitted[line] = f
    src, dst = [], []
    lines = list(fitted)
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            c = _seg_crossing(a, b)
            if c is None:
                continue
            p = _intersect(fitted[a], fitted[b])
            if p is not None:
                src.append(c)
                dst.append(p)
    if len(src) < cfg.min_keypoints:
        return None
    new, _ = cv2.findHomography(np.float32(src), np.float32(dst), 0)
    if new is None:
        return None
    return new, line_residual(ridges, new, cfg), len(src)


def _seg_crossing(a, b):
    """Reference point where two axis-aligned court segments cross (including T and L junctions)."""
    (ax0, ay0), (ax1, ay1) = a
    (bx0, by0), (bx1, by1) = b
    if ay0 == ay1 and bx0 == bx1:
        x, y = bx0, ay0
    elif ax0 == ax1 and by0 == by1:
        x, y = ax0, by0
    else:
        return None
    within = lambda v, lo, hi: min(lo, hi) - 1 <= v <= max(lo, hi) + 1
    if within(x, ax0, ax1) and within(y, ay0, ay1) and within(x, bx0, bx1) and within(y, by0, by1):
        return (x, y)
    return None


def line_residual(ridges, matrix, cfg):
    """Median distance (px) between white-ridge peaks and the projected court lines: the calibration quality."""
    res = []
    for line, pts in line_hits(ridges, matrix, cfg):
        a, b = project(matrix, line)
        d = (b - a) / np.linalg.norm(b - a)
        res.extend(np.abs((pts[:, 0] - a[0]) * d[1] - (pts[:, 1] - a[1]) * d[0]).tolist())
    return float(np.median(res)) if res else float("inf")


def calibrate_from_frames(detector, frames, frame_size, cfg):
    """Median keypoints over frames -> homography. Returns a result dict; result['ok'] is the quality gate."""
    per_frame = [detector.keypoints(f, cfg.kp_conf) for f in frames]
    n = len(per_frame)
    median_pts, hits = [], []
    for k in range(14):
        found = [p[k] for p in per_frame if p[k] is not None]
        hits.append(len(found))
        if len(found) >= max(1, cfg.kp_min_hits * n):
            median_pts.append(tuple(np.median(np.array(found), axis=0).tolist()))
        else:
            median_pts.append(None)
    matrix, model_err = get_trans_matrix(median_pts)
    line_err_before = line_err = float("inf")
    crossings = 0
    if matrix is not None:
        background = np.median(np.stack([cv2.resize(f, (WORK_W, WORK_H)) for f in frames]), axis=0).astype(np.uint8)
        ridges = _white_ridges(background)
        line_err_before = line_err = line_residual(ridges, matrix, cfg)
        for _ in range(cfg.snap_iters):
            better = snap_refit(background, matrix, cfg)
            if better is None or better[1] >= line_err:
                break
            matrix, line_err, crossings = better
    n_kp = sum(p is not None for p in median_pts)
    w, h = frame_size
    scale = np.diag([w / WORK_W, h / WORK_H, 1.0])
    rnd = lambda v: None if not np.isfinite(v) else round(float(v), 2)
    result = {
        "frames_used": n,
        "keypoint_hits": hits,
        "n_keypoints": n_kp,
        "model_kp_err_px_1280": rnd(model_err),
        "line_err_px_1280_before_refit": rnd(line_err_before),
        "line_err_px_1280": rnd(line_err),
        "refit_crossings": crossings,
        "keypoints": [None if p is None else [p[0] * w / WORK_W, p[1] * h / WORK_H] for p in median_pts],
        "homography": None if matrix is None else (scale @ matrix).tolist(),
    }
    reasons = []
    if matrix is None:
        reasons.append("no 4-point court configuration found")
    if n_kp < cfg.min_keypoints:
        reasons.append(f"only {n_kp} keypoints (< {cfg.min_keypoints})")
    if not np.isfinite(line_err) or line_err > cfg.max_line_px:
        reasons.append(f"court lines off by {rnd(line_err)} px (> {cfg.max_line_px})")
    result["ok"] = not reasons
    result["problems"] = reasons
    return result
