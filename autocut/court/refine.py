# Adapted from AymenMB/Tennis-Match-Analysis- (post_process.py): keypoint refinement by intersecting the two
# court lines found near the predicted point. sympy line intersection replaced with numpy.
import cv2
import numpy as np


def _intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    a, b = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
    return (a * (x3 - x4) - (x1 - x2) * b) / den, (a * (y3 - y4) - (y1 - y2) * b) / den


def _detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=20)
    return [] if lines is None else lines.reshape(-1, 4).tolist()


def _merge_lines(lines):
    lines = sorted(lines, key=lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []
    for i, line in enumerate(lines):
        if not mask[i]:
            continue
        for j, s_line in enumerate(lines[i + 1:]):
            if mask[i + j + 1]:
                x1, y1, x2, y2 = line
                x3, y3, x4, y4 = s_line
                if np.hypot(x1 - x3, y1 - y3) < 20 and np.hypot(x2 - x4, y2 - y4) < 20:
                    line = [int((x1 + x3) / 2), int((y1 + y3) / 2), int((x2 + x4) / 2), int((y2 + y4) / 2)]
                    mask[i + j + 1] = False
        new_lines.append(line)
    return new_lines


def refine_kp(img, x, y, crop_size=40):
    """Snap (x, y) to the intersection of the two lines in a crop around it; unchanged if not found."""
    h, w = img.shape[:2]
    x0, x1 = max(int(x) - crop_size, 0), min(w, int(x) + crop_size)
    y0, y1 = max(int(y) - crop_size, 0), min(h, int(y) + crop_size)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return x, y
    lines = _detect_lines(crop)
    if len(lines) > 1:
        lines = _merge_lines(lines)
        if len(lines) == 2:
            inter = _intersection(lines[0], lines[1])
            if inter and 0 < inter[0] < crop.shape[1] and 0 < inter[1] < crop.shape[0]:
                return x0 + inter[0], y0 + inter[1]
    return x, y
