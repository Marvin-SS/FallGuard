# FallGuard — Intelligent Camera-Based Fall Detection

Week 1 scaffold: pose estimation → feature engineering → rule-based fall
classifier → CSV log + annotated video. Runs on any video file today;
point it at a webcam feed once you have one.

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Get test data

You need a few labeled fall videos to validate the pipeline. Good options:

- **UR Fall Detection Dataset** — RGB + depth, labeled falls vs. ADLs
  (activities of daily living like sitting, walking, bending).
  http://fenix.ur.edu.pl/~mkepski/ds/uf.html
- **Le2i Fall Detection Dataset** — multiple room settings (home, coffee
  room, office), good for testing lighting/occlusion robustness.
- **Multiple Cameras Fall Dataset** (Université de Montréal) — same fall
  from multiple camera angles, useful later for angle-robustness testing.

Drop a handful of clips into `data/sample_videos/`. Start with ~10 fall
clips and ~10 non-fall clips (sitting, bending, lying down on purpose) —
that mix is what actually tells you if your thresholds work.

## Run it

```bash
python -m src.run_pipeline \
    --video data/sample_videos/fall_01.mp4 \
    --output data/results/fall_01_log.csv \
    --annotate data/results/fall_01_annotated.mp4
```

This prints an `[ALERT]` line the moment a fall is confirmed, writes a
per-frame CSV (torso angle, aspect ratio, hip velocity, state,
confidence), and optionally writes an annotated video with keypoints
and live state overlay — useful for a demo GIF later.

## How detection works (v1, rule-based)

See `src/fall_detector.py` for the full logic and comments. Short version:
a fall is treated as a two-phase event — a downward velocity **impact**,
followed by the torso angle **settling** near-horizontal for a sustained
number of frames. Requiring both phases is what filters out fast sitting
(impact-like velocity, but the torso recovers to a seated angle) and
bending or lying down on purpose (high torso angle, but no impact spike).

## Roadmap

- [x] Pose estimation wrapper (MediaPipe)
- [x] Feature extraction (torso angle, aspect ratio, hip velocity)
- [x] Rule-based fall classifier with confidence scoring
- [x] CLI pipeline + CSV logging + annotated video output
- [ ] Threshold tuning against labeled dataset — measure precision/recall
- [ ] Swap `frame_skip` tuning + resolution downscaling for real-time perf
- [ ] Webcam live-capture mode (swap file path for `cv2.VideoCapture(0)`)
- [ ] ML classifier (LSTM or random forest on the feature sequence) as v2,
      benchmarked against the rule-based baseline
- [ ] Backend API (FastAPI) to receive alerts + store incidents
- [ ] Frontend dashboard (React) — live status, incident history, confidence

## Project structure

```
fallguard/
├── requirements.txt
├── README.md
├── src/
│   ├── pose_estimation.py   # MediaPipe wrapper — swap CV backend here only
│   ├── features.py          # raw landmarks -> fall-relevant signals
│   ├── fall_detector.py     # rule-based state machine + confidence
│   └── run_pipeline.py      # CLI entry point, ties it all together
├── data/
│   ├── sample_videos/       # put dataset clips here
│   └── results/             # CSV logs + annotated videos land here
└── notebooks/                # for threshold tuning / EDA once you have data
```
