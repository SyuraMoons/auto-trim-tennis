"""Every tunable threshold in one place."""
from dataclasses import dataclass


@dataclass
class Config:
    # court calibration
    calib_frames: int = 15            # frames sampled across the video for the court model
    kp_conf: float = 0.5              # min heatmap peak for a keypoint
    kp_min_hits: float = 0.34         # keypoint must be found in this share of sampled frames
    min_keypoints: int = 8            # quality gate: keypoints kept after the median
    max_line_px: float = 2.5          # quality gate: median distance of white lines to the fitted court (1280 px)
    snap_iters: int = 2               # snap projected line crossings to real lines, refit
    snap_max_px: float = 20.0         # search this far (1280 px) either side of each projected line
    ridge_min: int = 25               # min white-line (top-hat) response to accept a line hit
    play_behind_m: float = 4.0        # play area past each baseline
    play_side_m: float = 2.0          # play area past each sideline

    # analysis sampling
    analysis_fps: float = 10.0
    analysis_width: int = 640

    # motion (MOG2 background subtraction)
    mog_history_s: float = 20.0
    mog_var_threshold: float = 25.0

    # person detector
    person_fps: float = 2.0
    person_score: float = 0.5
    person_headroom_px: int = 120     # crop headroom above the far edge of the play area (native px)

    # segmenting
    seg_on: float = 0.5               # hysteresis: start a segment at this play probability
    seg_off: float = 0.35             # ... and end it below this one
    seg_min_s: float = 1.5            # drop segments shorter than this (before padding)
    pad_pre_s: float = 1.5
    pad_post_s: float = 1.5
    merge_gap_s: float = 2.0
