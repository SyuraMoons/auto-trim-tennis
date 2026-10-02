"""Labelling helper: watch a video and mark rally start/end times -> labels/<video>.csv (autosaved)."""
import csv
import os

import cv2

from .calibrate import ROOT

HELP = [
    "SPACE play/pause   S start rally   E end rally   W warm-up start/end   U undo last",
    "<- / -> 1 s (also A / D)   J / L 5 s   , / . one frame   [ / ] speed",
    "Q save + quit",
]
KEY_LEFT, KEY_RIGHT = {2424832, 63234, 65361, 81}, {2555904, 63235, 65363, 83}


def labels_path(video):
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(ROOT, "labels", f"{stem}.csv")


def load_labels(path, kind="rally"):
    """(start, end) spans of one kind ('rally' or 'warmup'); kind=None returns (start, end, kind)."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = [(float(r["start"]), float(r["end"]), r.get("kind") or "rally") for r in csv.DictReader(f)]
    return rows if kind is None else [(s, e) for s, e, k in rows if k == kind]


def save_labels(path, spans):
    """spans: (start, end) or (start, end, kind)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "kind"])
        for s, e, k in sorted(sp if len(sp) == 3 else (*sp, "rally") for sp in spans):
            w.writerow([f"{s:.2f}", f"{e:.2f}", k])


def _fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def run(video, out=None):
    path = out or labels_path(video)
    rallies = load_labels(path, kind=None)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    speeds = [0.5, 1.0, 2.0, 4.0]
    speed_i, playing, pending, warm = 1, False, None, None
    idx, frame = 0, None
    win = "autocut label"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 760)

    def seek(i):
        nonlocal idx, frame
        idx = max(0, min(n_frames - 1, i))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            frame = f

    seek(0)
    while True:
        t = idx / fps
        view = cv2.resize(frame, (1280, round(frame.shape[0] * 1280 / frame.shape[1])))
        inside = any(s <= t <= e for s, e, k in rallies if k == 'rally') or (pending is not None and t >= pending)
        bar = view.copy()
        cv2.rectangle(bar, (0, 0), (1280, 112), (0, 0, 0), -1)
        view = cv2.addWeighted(bar, 0.55, view, 0.45, 0)
        status = f"{_fmt(t)} / {_fmt(n_frames / fps)}   {'PLAY' if playing else 'PAUSE'} x{speeds[speed_i]}" \
                 f"   rallies: {sum(k == 'rally' for *_, k in rallies)}" \
                 + (f"   OPEN since {_fmt(pending)}" if pending is not None else "") \
                 + (f"   WARM-UP since {_fmt(warm)}" if warm is not None else "")
        cv2.putText(view, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if inside else (255, 255, 255), 2)
        for i, line in enumerate(HELP):
            cv2.putText(view, line, (12, 58 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        # timeline with labelled rallies
        h = view.shape[0]
        cv2.rectangle(view, (0, h - 14), (1280, h), (60, 60, 60), -1)
        total = n_frames / fps
        for s, e, k in rallies:
            color = (0, 200, 0) if k == "rally" else (0, 140, 255)
            cv2.rectangle(view, (int(s / total * 1280), h - 14), (int(e / total * 1280) + 1, h), color, -1)
        cv2.line(view, (int(t / total * 1280), h - 20), (int(t / total * 1280), h), (0, 0, 255), 2)
        cv2.imshow(win, view)

        key = cv2.waitKeyEx(max(1, int(1000 / fps / speeds[speed_i])) if playing else 0)
        ch = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 128 and key < 256 else ""
        if key == -1:
            if playing:
                step = max(1, round(speeds[speed_i])) if speeds[speed_i] >= 1 else 1
                for _ in range(step - 1):
                    cap.grab()
                ok, f = cap.read()
                if not ok:
                    playing = False
                else:
                    frame, idx = f, idx + step
            continue
        if ch == "q" or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif ch == " ":
            playing = not playing
        elif ch == "s":
            pending = t
        elif ch == "e" and pending is not None and t > pending:
            rallies.append((pending, t, "rally"))
            pending = None
            save_labels(path, rallies)
        elif ch == "w":
            if warm is None:
                warm = t
            elif t > warm:
                rallies.append((warm, t, "warmup"))
                warm = None
                save_labels(path, rallies)
        elif ch == "u":
            if pending is not None or warm is not None:
                pending = warm = None
            elif rallies:
                rallies.sort()
                rallies.pop()
                save_labels(path, rallies)
        elif ch == "[":
            speed_i = max(0, speed_i - 1)
        elif ch == "]":
            speed_i = min(len(speeds) - 1, speed_i + 1)
        elif key in KEY_LEFT or ch == "a":
            seek(idx - int(fps))
        elif key in KEY_RIGHT or ch == "d":
            seek(idx + int(fps))
        elif ch == "j":
            seek(idx - int(5 * fps))
        elif ch == "l":
            seek(idx + int(5 * fps))
        elif ch == ",":
            seek(idx - 1)
        elif ch == ".":
            seek(idx + 1)
    save_labels(path, rallies)
    cap.release()
    cv2.destroyAllWindows()
    print(f"{sum(k == 'rally' for *_, k in rallies)} rallies saved to {path}")
