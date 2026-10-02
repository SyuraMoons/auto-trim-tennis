import argparse
import json
import os
import sys

import cv2
import numpy as np

from . import calibrate as cal
from .config import Config
from .video import probe, read_at


def cmd_calibrate(args):
    cfg = Config()
    info = probe(args.video)
    t_pick = info.duration * 0.5
    if args.corners:
        corners = [tuple(map(float, p.split(","))) for p in args.corners.split(";")]
        if len(corners) != 4:
            sys.exit("--corners needs 4 points: far-left;far-right;near-right;near-left")
        matrix = cal.manual_homography(corners)
        result = {"ok": True, "problems": [], "keypoints": None, "homography": matrix.tolist()}
        source = "manual"
    else:
        from .court.detect import CourtDetector, calibrate_from_frames
        n = cfg.calib_frames
        times = [info.duration * (i + 0.5) / n for i in range(n)]
        frames = [f for _, f in read_at(args.video, times)]
        detector = CourtDetector(cal.WEIGHTS)
        result = calibrate_from_frames(detector, frames, (info.width, info.height), cfg)
        source = "auto"
    print(json.dumps({k: result[k] for k in result if k not in ("keypoints", "homography")}, indent=2))
    if result["homography"] is None:
        sys.exit("Calibration failed: no homography. Re-run with --corners.")

    _, frame = next(read_at(args.video, [t_pick]))
    overlay = cal.draw_overlay(frame, np.array(result["homography"]), cfg)
    overlay_path = args.overlay or os.path.join(cal.COURTS_DIR, f"{args.court_id}_overlay.jpg")
    os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
    cv2.imwrite(overlay_path, overlay)
    print(f"overlay: {overlay_path}")

    if not result["ok"] and not args.force:
        sys.exit("Quality gate failed (see problems). Check the overlay; use --force to save anyway "
                 "or --corners to set the court by hand.")
    path = cal.save(args.court_id, {
        "source": source, "video": os.path.abspath(args.video), "frame_size": [info.width, info.height],
        **{k: v for k, v in result.items() if k not in ("ok",)},
    })
    print(f"saved: {path}")


def _labels_for(video, given=None):
    from .label import labels_path
    path = given or labels_path(video)
    if not os.path.exists(path):
        sys.exit(f"no labels at {path}; run `autocut label {video}` first")
    return path


def cmd_train(args):
    from . import model
    cfg = Config()
    examples = []
    for ex in args.example:
        if len(ex) not in (2, 3):
            sys.exit("--example takes VIDEO COURT_ID [LABELS_CSV]")
        video, court_id = ex[0], ex[1]
        print(f"== {os.path.basename(video)}")
        examples.append(model.load_example(video, court_id, _labels_for(video, ex[2] if len(ex) == 3 else None), cfg))
    bundle = model.train(examples, cfg, min_keep=args.min_keep)
    model.save(bundle, args.model)
    print(f"model saved: {args.model}")


def cmd_run(args):
    import time
    from . import model
    from .render import render
    t0 = time.time()
    bundle = model.load(args.model)
    calib = cal.load(args.court_id)
    segs, info = model.predict(bundle, args.video, calib)
    out = args.output or os.path.splitext(args.video)[0] + "_trimmed.mp4"
    seg_path = os.path.splitext(out)[0] + ".segments.json"
    kept = sum(e - s for s, e in segs)
    with open(seg_path, "w") as f:
        json.dump({"video": os.path.abspath(args.video), "duration_s": round(info["duration"], 2),
                   "output_s": round(kept, 2), "model_trained_on": bundle["trained_on"],
                   "segments": [[round(s, 2), round(e, 2)] for s, e in segs]}, f, indent=2)
    print(f"{len(segs)} segments, keeping {kept:.0f}s of {info['duration']:.0f}s ({kept / info['duration']:.0%})")
    print(f"segments: {seg_path}")
    if args.segments_only:
        return
    if not segs:
        sys.exit("no gameplay found; nothing to render")
    print("rendering ...")
    render(args.video, segs, out, has_audio=info["has_audio"])
    print(f"video: {out}  ({time.time() - t0:.0f}s total)")


def cmd_eval(args):
    from .evaluate import evaluate
    from .label import load_labels
    from .video import probe
    with open(args.segments) as f:
        data = json.load(f)
    labels = _labels_for(data["video"], args.labels)
    cfg = Config()
    r = evaluate([tuple(s) for s in data["segments"]], load_labels(labels, "rally"), load_labels(labels, "warmup"),
                 data["duration_s"], cfg.pad_pre_s, cfg.pad_post_s, cfg.merge_gap_s)
    print(json.dumps(r, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="autocut", description="Trim tennis recordings to gameplay only.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="detect the court once and save it for this camera")
    c.add_argument("video")
    c.add_argument("--court-id", required=True)
    c.add_argument("--corners", help='manual override: "x,y;x,y;x,y;x,y" far-left;far-right;near-right;near-left')
    c.add_argument("--overlay", help="where to write the check image (default courts/<id>_overlay.jpg)")
    c.add_argument("--force", action="store_true", help="save even if the quality gate fails")
    c.set_defaults(func=cmd_calibrate)

    lb = sub.add_parser("label", help="watch a video and mark rally start/end (saved to labels/<video>.csv)")
    lb.add_argument("video")
    lb.add_argument("-o", "--out", help="labels CSV path (default labels/<video>.csv)")
    lb.set_defaults(func=lambda a: __import__("autocut.label", fromlist=["run"]).run(a.video, a.out))

    tr = sub.add_parser("train", help="train the play/dead model from labelled videos")
    tr.add_argument("--example", nargs="+", action="append", required=True, metavar="ARG",
                    help="VIDEO COURT_ID [LABELS_CSV]; repeat for each labelled video")
    tr.add_argument("--model", default=os.path.join(cal.ROOT, "models", "autocut_model.joblib"))
    tr.add_argument("--min-keep", type=float, default=0.95,
                    help="share of play (rally + pre-serve) that must survive on every held-out video (default 0.95)")
    tr.set_defaults(func=cmd_train)

    r = sub.add_parser("run", help="trim a video to gameplay only")
    r.add_argument("video")
    r.add_argument("--court-id", required=True)
    r.add_argument("-o", "--output", help="trimmed video path (default <video>_trimmed.mp4)")
    r.add_argument("--model", default=os.path.join(cal.ROOT, "models", "autocut_model.joblib"))
    r.add_argument("--segments-only", action="store_true", help="write the segments JSON, skip rendering")
    r.set_defaults(func=cmd_run)

    ev = sub.add_parser("eval", help="score a run's segments against labels")
    ev.add_argument("segments", help="the .segments.json written by `run`")
    ev.add_argument("--labels", help="labels CSV (default labels/<video>.csv)")
    ev.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
