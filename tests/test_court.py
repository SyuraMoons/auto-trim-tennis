import numpy as np

from autocut.calibrate import manual_homography, regions, scaled_homography
from autocut.config import Config
from autocut.court.detect import _ref_lines, _seg_crossing
from autocut.court.homography import get_trans_matrix, project, refer_kps
from autocut.court.reference import CourtReference


def _fake_homography():
    # a plausible perspective: far end narrower and higher in the image
    src = np.float32(CourtReference().court_corners())
    dst = np.float32([(770, 300), (1165, 300), (1660, 832), (232, 832)])
    import cv2
    return cv2.getPerspectiveTransform(src, dst)


def test_trans_matrix_recovers_reference_points():
    h = _fake_homography()
    pts = [tuple(p) for p in project(h, refer_kps.reshape(-1, 2))]
    pts[0] = None  # a missing keypoint must not break it
    m, err = get_trans_matrix(pts)
    back = project(m, refer_kps.reshape(-1, 2))
    assert err < 1.0
    assert np.max(np.linalg.norm(back[1:] - np.float32(pts[1:]), axis=1)) < 1.0


def test_manual_corners_match():
    h = _fake_homography()
    corners = project(h, CourtReference().court_corners())
    m = manual_homography(corners)
    assert np.allclose(project(m, refer_kps.reshape(-1, 2)), project(h, refer_kps.reshape(-1, 2)), atol=0.5)


def test_every_line_crossing_is_found():
    lines = _ref_lines()
    found = {_seg_crossing(a, b) for i, a in enumerate(lines) for b in lines[i + 1:]} - {None}
    # 4 sidelines x 2 baselines, plus 2 singles sidelines and the centre line x 2 service lines
    expected = {(x, y) for x in (286, 423, 1242, 1379) for y in (561, 2935)}
    expected |= {(x, y) for x in (423, 832, 1242) for y in (1110, 2386)}
    assert found == expected


def test_regions_split_at_net_and_scale():
    h = _fake_homography()
    reg = regions(h, Config())
    assert reg["far"][:, 1].mean() < reg["near"][:, 1].mean()
    calib = {"frame_size": [1920, 1080], "homography": h.tolist()}
    s = scaled_homography(calib, 640, 360)
    assert np.allclose(project(s, [(286, 561)]), project(h, [(286, 561)]) / 3, atol=1e-3)
