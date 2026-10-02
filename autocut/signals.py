"""Raw per-video signals (motion, people, audio), cached on disk so training/tuning never recomputes them."""
import hashlib
import os
import subprocess

import numpy as np

from .activity.motion import motion_signal
from .calibrate import ROOT
from .video import iter_frames

CACHE_DIR = os.path.join(ROOT, "cache")
AUDIO_SR = 16000


def _cache_path(video, calib):
    st = os.stat(video)
    key = f"{os.path.abspath(video)}|{st.st_size}|{st.st_mtime}|{calib['homography']}"
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(CACHE_DIR, f"{stem}_{hashlib.sha1(key.encode()).hexdigest()[:10]}.npz")


def read_audio(video):
    """Mono 16 kHz float32 audio, or None if the video has no audio stream."""
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
                           "-f", "s16le", "-"], capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768


def person_signal(video, calib, cfg, log=print):
    from .activity.persons import PersonDetector
    det = PersonDetector(calib, cfg)
    ts, xy, counts = [], [], []
    for t, frame in iter_frames(video, cfg.person_fps):
        people = det.detect(frame)
        ts.append(t)
        counts.append(len(people))
        xy.extend(people)
    return {"pt": np.array(ts), "pxy": np.array(xy, np.float32).reshape(-1, 2), "pcount": np.array(counts, np.int32)}


def extract(video, calib, cfg, log=print, use_cache=True):
    path = _cache_path(video, calib)
    if use_cache and os.path.exists(path):
        log(f"signals: cached ({os.path.basename(path)})")
        z = np.load(path)
        return {k: z[k] for k in z.files}
    log("signals: motion ...")
    sig = motion_signal(video, calib, cfg)
    log("signals: people ...")
    sig.update(person_signal(video, calib, cfg, log))
    log("signals: audio ...")
    audio = read_audio(video)
    sig["audio"] = audio if audio is not None else np.zeros(0, np.float32)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, **sig)
    return sig


def people_lists(sig):
    """Unpack the flat people array back into one list of (x_m, y_m) per sample."""
    out, i = [], 0
    for c in sig["pcount"]:
        out.append([tuple(p) for p in sig["pxy"][i:i + c]])
        i += c
    return out
