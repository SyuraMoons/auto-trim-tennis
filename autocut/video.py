"""Streaming video access: never holds more than one frame in memory."""
import json
import subprocess
from dataclasses import dataclass

import cv2


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    has_audio: bool


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate,nb_frames",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    num, den = v["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration = float(data["format"]["duration"])
    frames = int(v.get("nb_frames") or round(duration * fps))
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    return VideoInfo(path, int(v["width"]), int(v["height"]), fps, frames, duration, has_audio)


def read_at(path, times):
    """Yield (t, frame) for each requested time in seconds (seek per frame; for sparse sampling)."""
    cap = cv2.VideoCapture(path)
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if ok:
                yield t, frame
    finally:
        cap.release()


def iter_frames(path, sample_fps, width=None):
    """Yield (t, frame) at roughly sample_fps by decoding sequentially and skipping frames."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / sample_fps))
    idx = 0
    try:
        while True:
            if idx % step:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            if width and frame.shape[1] != width:
                h = round(frame.shape[0] * width / frame.shape[1])
                frame = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)
            yield idx / fps, frame
            idx += 1
    finally:
        cap.release()
