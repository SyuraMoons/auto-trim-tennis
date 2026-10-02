# auto-trim-tennis (`autocut`)

Give it a tennis recording from a fixed camera, and it gives back a shorter video with only the gameplay:
- Dead time (walking, picking up balls, changeovers) is cut out.
- The **pre-serve routine** (bouncing the ball before serving) is kept.
- Each rally keeps 1.5 s of padding before and after.
- Sound is kept.

It also writes a small JSON file listing which parts of the original video were kept.

> **Status: prototype.** It has been tested on two 7-minute videos from **one court**. See [Current quality](#current-quality) before showing results to anyone.

---

## 1. Install (once)

You need:
- **Python 3.11 or newer**
- **ffmpeg** (to read and write video)
- **git**
- About 2 GB of free disk space (most of it is PyTorch), and internet for the first run

macOS (with [Homebrew](https://brew.sh)):
```bash
brew install python@3.11 ffmpeg git
```
Ubuntu/Debian:
```bash
sudo apt install python3.11 python3.11-venv ffmpeg git
```
Windows: install Python 3.11 from python.org and ffmpeg from https://www.gyan.dev/ffmpeg/builds/, and make sure `ffmpeg` works in a new terminal.

Then:
```bash
git clone https://github.com/SyuraMoons/auto-trim-tennis.git
cd auto-trim-tennis
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest                             # optional check: should say "11 passed"
```

**Every time you open a new terminal**, go into the folder and activate the environment again:
```bash
cd auto-trim-tennis
source .venv/bin/activate
```
After that, the `autocut` command works.

> The first time you trim a video, it downloads a person-detection model (about 75 MB) from PyTorch. This only happens once.

---

## 2. Trim a video (the everyday step)

Each camera position needs a **court ID**. The court ID tells the tool where the court lines are in the picture.

### A. Your video is from the same camera as the test videos
The repo already includes two court IDs: `test_221740` and `test_222509`. They are the same camera.
```bash
autocut run path/to/match.mp4 --court-id test_221740
```

### B. Your video is from a new camera or court
First tell the tool where the court is. You only do this **once per camera position**:
```bash
autocut calibrate path/to/match.mp4 --court-id my_court
```
Then open `courts/my_court_overlay.jpg` and check that the drawn lines sit on the real court lines.
- **Lines look right:** you're done. Use `--court-id my_court` from now on.
- **It fails, or the lines are wrong:** give the 4 court corners by hand, in pixels of the original video. The order is far-left, far-right, near-right, near-left:
  ```bash
  autocut calibrate path/to/match.mp4 --court-id my_court --corners "610,330;1310,330;1700,1000;220,1000"
  ```
  You can read the pixel positions from any image viewer that shows the cursor position.

Then trim:
```bash
autocut run path/to/match.mp4 --court-id my_court
```

### What you get
| File | What it is |
|---|---|
| `match_trimmed.mp4` | The trimmed video, next to the original. Pick another name with `-o out.mp4` |
| `match_trimmed.segments.json` | The kept time ranges, in seconds, from the original video |

To get only the JSON without rendering a video, add `--segments-only`.

**How long it takes:** about 2.7 minutes for a 7-minute 1080p video on an M-series Mac, using about 0.7 GB of RAM. Without an Apple GPU or NVIDIA GPU it runs on the CPU and is slower.

---

## 3. Make it better for a new court (optional)

The model learned from one court. On a new venue it will probably keep too much, or now and then cut a bit of play. To fix this, label 1–2 videos from that venue and retrain.

**Step 1: label a video.** A window opens and plays the video:
```bash
autocut label path/to/new_court_video.mp4
```
Labels save automatically to `labels/new_court_video.csv`. You can quit and continue later.

| Key | Action |
|---|---|
| `SPACE` | play / pause |
| `S` | rally **start**: when the server tosses the ball |
| `E` | rally **end**: when the ball is dead (out, net, missed, or a second bounce) |
| `W` | start / end of the **pre-serve routine** (bouncing the ball before serving) |
| `U` | undo the last mark |
| `←` `→` or `A` `D` | jump back / forward 1 s |
| `J` `L` | jump back / forward 5 s |
| `,` `.` | step one frame |
| `[` `]` | slower / faster playback |
| `Q` | save and quit |

Labelling rules, so everyone labels the same way:
- A rally **starts at the server's ball toss** and **ends when the ball is dead**.
- Warm-up hitting before the match is **not** a rally. Don't mark it.
- Use `W` around the pre-serve routine. It is kept in the output.

**Step 2: retrain, using every labelled video** (old ones included). Each `--example` is `VIDEO COURT_ID`:
```bash
autocut train \
  --example videos/VID_20260826_221740.mp4 test_221740 \
  --example videos/VID_20260826_222509.mp4 test_222509 \
  --example path/to/new_court_video.mp4 my_court
```
This overwrites `models/autocut_model.joblib`.
- `train` prints a score for each video, measured by a model that **did not** see that video. Those are the honest numbers.
- `--min-keep 0.95` (the default) means at least 95% of the play must survive on every video. A lower value, such as `0.93`, cuts more dead time but risks trimming a little play.
- To retrain on the original two videos, you need the two test videos. They are not in this repo, so ask Harfi for them.

**Step 3 (optional): score a trim** against labels:
```bash
autocut eval path/to/match_trimmed.segments.json
```
This uses `labels/<video>.csv`. It is only meaningful for a video the model was **not** trained on. On a training video it looks better than reality.

---

## Current quality

We have two labelled videos from the same court. Each one was scored with a model trained only on the *other* video:

| Video | Rallies lost | Rally time kept | Rally + pre-serve kept | Dead time cut | Best possible cut* | Output length |
|---|---|---|---|---|---|---|
| VID_20260826_221740 | 0 of 29 | 97.4% | 95.1% | 49% | 60% | 73% of original |
| VID_20260826_222509 | 0 of 35 | 98.6% | 97.1% | 19% | 64% | 88% of original |

\*Best possible cut means what a perfect detector would cut with the same 1.5 s padding.

What this means:
- **It almost never cuts play**, but it often keeps dead time it could have cut. It is tuned to err on the safe side.
- **It has only seen one court.** Results on other venues are unknown until a video from there is labelled.
- **Videos without sound work, but worse.** Ball-hit sounds are one of its three signals.

---

## How it works (technical)

1. **Court** (`autocut calibrate`):
   - A court-keypoint network runs on 15 frames spread across the video.
   - A line refit then snaps every court line onto the white lines in the image. This fixes far-court points the network misplaces.
   - The result is a homography from image to court metres, saved in `courts/<id>.json`.
   - The quality gate is a median line error of at most 2.5 px at 1280 px width.
2. **Signals:** these are cached in `cache/`, so retraining is fast.
   - Motion (OpenCV MOG2) in the near and far halves of the play area, at 10 fps. The play area is the court plus 4 m behind each baseline and 2 m at each side.
   - People: torchvision Faster R-CNN (MobileNetV3) at 2 fps. Each person is placed on the court by their feet, and people outside the play area are ignored.
   - Sound: impulsive 1–6 kHz onsets, which sound like ball hits.
3. **Model:**
   - Gradient boosting (scikit-learn) on mean/std/max of the signals over windows before, around and after each moment.
   - It outputs a play probability every 0.5 s. Each signal alone is weak (AUC 0.5–0.7); combined, they reach about AUC 0.76 on held-out video.
4. **Segments:** hysteresis threshold, then drop short segments, then pad 1.5 s on each side, then merge gaps under 2 s. The thresholds are chosen by `train`.
5. **Render:** one ffmpeg pass (trim/concat, H.264 CRF 20 + AAC 128k), with frame-accurate cuts.

All settings are in `autocut/config.py`.

### Repo layout
```
autocut/            the code (cli.py is the entry point)
  court/            court detection + homography
  activity/         motion and person signals
weights/court.pt    court keypoint network weights
models/             the trained play/dead model
courts/             saved court calibrations (+ overlay check images)
labels/             rally labels (CSV: start,end,kind)
tests/              pytest suite
```

## Where the code comes from
- The court network (`autocut/court/track_net.py`), its weights (`weights/court.pt`), the homography search and the court reference come from [AymenMB/Tennis-Match-Analysis-](https://github.com/AymenMB/Tennis-Match-Analysis-), which built on [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector). **The weights' licence has not been confirmed. Check it before any commercial use.**
- The feet-in-court player filter is based on [Chaganti-Reddy/Tennis-Analysis-YOLOV8](https://github.com/Chaganti-Reddy/Tennis-Analysis-YOLOV8). It is reimplemented without Ultralytics.
- The detect → join → drop-short segment idea comes from [lwc-alex/tennis-rally-detector](https://github.com/lwc-alex/tennis-rally-detector).
