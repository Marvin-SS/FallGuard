# FallGuard

A real-time, camera-based fall detection system built on pose estimation and a rule-based classifier, designed and validated with a focus on the hardest part of this problem: **camera geometry**, not just detection thresholds.

> **Status:** Active development. Core detection pipeline is built and validated against the UR Fall Detection Dataset. FastAPI backend + React dashboard are planned next phases. See [Roadmap](#roadmap) below.

---

## What it does

FallGuard watches a video feed, tracks a person's body pose frame by frame, and flags when a fall has occurred, distinguishing a real fall from sitting down quickly, bending over, or other everyday movement.

The pipeline runs in three stages:

1. **Pose estimation** (MediaPipe): extracts body landmarks (shoulders, hips, etc.) from each frame.
2. **Feature engineering**: converts raw landmarks into three signals:
   - `torso_angle`: angle of the shoulder-hip line from vertical
   - `aspect_ratio`: width vs. height of the body's bounding box
   - `hip_velocity`: vertical speed of the hips between frames
3. **Fall classification**: a two-phase rule-based state machine (`normal` → `impact_detected` → `fall_confirmed`) that combines those signals with a confidence score, rather than firing on a single frame.

## Why this project is more than a MediaPipe wrapper

It would be easy to plug in a pose model, threshold two numbers, and call it done. Building and validating this against real dual-camera fall footage surfaced problems that don't show up until you test against real, varied data, and they turned out to be the most interesting part of the project. Two findings stand out:

**1. `torso_angle` is a hard gate, and every camera has a blind spot.**
The detector can only leave the `normal` state once `torso_angle` crosses threshold. `hip_velocity` alone was never enough to trigger detection under the original design. Testing against the same fall filmed from two synchronized camera angles proved this is a real reliability ceiling, not an edge case: the exact same fall, with an unambiguous velocity spike on both cameras, was caught on one and completely missed on the other, purely because of the fall's direction relative to that camera's viewing axis. Further testing showed this wasn't about "camera 0 vs. camera 1" at all: a camera that reliably catches most falls can still miss one that happens to travel toward/away from the lens rather than across it (foreshortening), and an overhead-mounted camera breaks the feature basis entirely, since a 2D shoulder-hip angle and vertical pixel velocity don't mean the same thing from directly above. `hip_velocity`, by contrast, held up correctly across every single one of these failures. That's strong, quantified evidence for the highest-value next architecture change: a velocity-only fallback trigger path (see [Roadmap](#roadmap)).

**2. A "camera health check" idea got proposed, then corrected by more data, before it was ever built.**
The natural fix for the camera-reliability problem seemed to be: calibrate each camera once at rest, and flag ones that look unreliable from the start (elevated baseline angle, noisy/high-dropout tracking). That design was deliberately logged and *not* built immediately, pending more test clips. The next clip disproved it: a camera with a completely normal, healthy resting baseline still missed a real fall, because the failure was trajectory-dependent, not a property of the camera itself, invisible until a fall actually happened along that camera's bad axis. The design was rescoped accordingly rather than shipped as a general solution it wasn't. The full reasoning is in `findings.txt` (Entries 10-11): it's the clearest example in this project of catching a design assumption before it became a false sense of coverage.

## Experiment: rule-based vs. learned classifier

To test whether the rule-based system's blind spots were fundamental or just a modeling choice, a second detector was built: a Random Forest trained on whole-clip summary features (max angle, max velocity, resting baseline angle, frame-to-frame volatility, dropout rate, and related stats extracted from the per-frame logs), evaluated with leave-one-out cross-validation given the small dataset size (~15-17 clips).

**Result: 73.3% LOOCV accuracy (11/15 correct), a genuinely mixed outcome, not a clean win.**

What it fixed: 3 of the rule-based system's documented misses, all cases where frame-by-frame timing broke down but the overall shape of the clip still carried the fall signature (fall-01-cam1's foreshortening, fall-02-cam1's overhead camera, fall-03-cam1's noisy/high-dropout tracking).

What it broke: 2 new false positives on ADL clips the rule-based system never flagged, and 1 regression, a previously-solved fall (fall-03-cam0) that the rule-based system's velocity-window fix correctly caught, now missed. Neither root cause is understood yet.

Working theory: whole-clip summary statistics are less sensitive to exact-frame timing, which is what let the model recover the timing-dependent misses, but that same insensitivity may be what's costing it precision elsewhere. Feature importances put `max_hip_velocity` well ahead of everything else, followed by `dropout_rate`, though `dropout_rate` ranking that high is suspicious: only 2 clips in the whole dataset were ever documented as having real dropout issues, and it may be a spurious signal contributing to the new false positives rather than a generalizable one.

The point of documenting this rather than glossing over it: a learned model isn't strictly better than hand-tuned rules, it trades one set of failure modes for a different one. This is still an open investigation, not a completed feature, and it's presented as one honestly. See `findings.txt`, Entry 14.

---

## Project structure

```
FallGuard/
├── data/
│   ├── sample_videos/       # UR Fall Dataset clips (fall-01..05, adl-01..02, cam0/cam1)
│   ├── results/              # Per-clip CSV logs, annotated videos, summary.csv
│   └── labels.csv            # clip name -> expected fall / no_fall
├── notebooks/                 # Exploratory analysis
├── src/
│   ├── pose_estimation.py    # MediaPipe wrapper
│   ├── features.py           # torso_angle, aspect_ratio, hip_velocity
│   ├── fall_detector.py      # Two-phase state machine + confidence scoring
│   ├── run_pipeline.py       # Single-clip CLI runner
│   ├── batch_runner.py       # Multi-clip batch runner + scorecard
│   └── sync_labels.py        # Keeps labels.csv in sync with sample_videos/
├── findings.txt              # Running engineering log (bugs, root causes, fixes)
├── .gitignore
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd fallguard
python -m venv venv
venv\Scripts\activate        # Windows (PowerShell)
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

> **Note:** `mediapipe` is pinned to `0.10.21` in `requirements.txt`. Version `1.0.0` replaced the `mp.solutions` API with a new Tasks API and will break this pipeline outright. This is intentional, not an oversight; see `findings.txt`, Entry 2.

## Usage

Run the pipeline on a single clip:

```bash
python -u -m src.run_pipeline --video data/sample_videos/fall-01-cam0.mp4 --output data/results/fall-01-cam0_log.csv
```

Add `--annotate` to also render an annotated output video (skeleton overlay + state label). This is a demo-only artifact and is **not** part of validation; it's excluded from batch runs by default:

```bash
python -u -m src.run_pipeline --video data/sample_videos/fall-01-cam0.mp4 --output data/results/fall-01-cam0_log.csv --annotate data/results/fall-01-cam0_annotated.mp4
```

Run every labeled clip at once and get a scored summary:

```bash
python -u -m src.batch_runner
```

This writes a full per-clip CSV log for every clip (same as the single-clip runner) plus one `summary.csv` scoring detected vs. expected outcome across the whole dataset. It's the primary tool used to validate detector changes against every previously-solved case at once, to catch regressions.

Sync `labels.csv` after adding new clips to `data/sample_videos/`:

```bash
python -u -m src.sync_labels
```

## Validation

Tested against the [UR Fall Detection Dataset](http://fenix.ur.edu.pl/~mkepski/ds/uf.html), a dual-camera synchronized dataset of real falls and activities of daily living (ADL). Validation specifically targeted dual-camera pairs of the *same* physical fall event, to isolate camera-geometry effects from detection-logic effects. Most fall-detection demos only test against a single friendly camera angle, which hides exactly the failure modes this project was built to surface.

## Known limitations

- **Trajectory-dependent blind spot:** falls that travel toward/away from the camera (rather than across it) can foreshorten the torso-angle signal below detection threshold, even when velocity is unambiguous.
- **Overhead/steep-angle mounts:** the current feature set assumes a roughly side-on camera. Directly overhead, `torso_angle` and `hip_velocity` no longer describe the same physical motion and produce unreliable readings: sometimes smoothly biased, sometimes noisy with heavy landmark dropout.
- **Single fixed camera only:** no multi-camera fusion yet; each feed is scored independently.

These aren't bugs so much as first-order constraints of single-camera 2D pose-based detection; see `findings.txt` for the full investigation behind each one.

## Roadmap

- [ ] **Diagnose Random Forest false positives/regression**: root-cause the 2 new ADL false positives and the fall-03-cam0 regression from the learned-classifier experiment before considering it a candidate to replace or complement the rule-based system (see `findings.txt`, Entry 14).
- [ ] **Velocity-only fallback trigger**: allow a strong, sustained velocity spike to trigger `impact_detected` without requiring `torso_angle` to cross threshold: needs false-positive testing against fast-but-normal motion (sitting quickly, bending).
- [ ] **Camera viewpoint health check**: flag cameras with abnormal resting baseline, high landmark dropout, or excessive frame-to-frame volatility. Explicitly scoped to catch *broken-camera* failures only, not trajectory-dependent ones (see Entry 11).
- [ ] **Overhead-specific feature set**: bounding-box area and motion-blob dispersal for ceiling-mounted cameras, common in real senior-care deployments.
- [ ] **Multi-camera fusion**: combine confidence scores across synchronized camera angles rather than scoring each independently.
- [ ] **FastAPI backend**: serve real-time detection results over an API.
- [ ] **React dashboard**: live monitoring UI on top of the FastAPI backend.

## Tech stack

Python · MediaPipe `0.10.21` · OpenCV · NumPy/pandas for CSV logging and scoring · scikit-learn (Random Forest, experimental) · (planned) FastAPI · (planned) React

## Engineering log

`findings.txt` is the running, unedited record this README was written from: every bug, root cause, fix, and design decision, in the order it actually happened. Worth a read if you want the full investigative detail behind the summaries above.