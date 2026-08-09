"""
batch_runner.py

Runs the full pipeline against every clip listed in data/labels.csv,
writes a per-clip CSV log (same as run_pipeline.py always did), and
produces ONE summary.csv scoring every clip against its expected label.

Annotated videos are OFF by default here — they're for demos, not bulk
testing. Use --annotate-all if you specifically want them for every
clip (slower, larger files), or run run_pipeline.py directly for a
single clip you want to showcase.

Usage:
    python -m src.batch_runner
    python -m src.batch_runner --labels data/labels.csv --videos-dir data/sample_videos
    python -m src.batch_runner --annotate-all
"""

import argparse
import csv
import os
import sys

from src.run_pipeline import run


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")


def find_video_file(videos_dir, clip_name):
    """Match a label's clip name (no extension) to an actual file on disk."""
    for ext in VIDEO_EXTENSIONS:
        candidate = os.path.join(videos_dir, clip_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def run_batch(labels_path, videos_dir, results_dir, annotate_all=False):
    with open(labels_path, newline="") as f:
        labels = list(csv.DictReader(f))

    os.makedirs(results_dir, exist_ok=True)

    summary_rows = []

    for i, label_row in enumerate(labels, 1):
        clip_name = label_row["clip"].strip()
        expected = label_row["expected"].strip()

        video_path = find_video_file(videos_dir, clip_name)
        print(f"[{i}/{len(labels)}] {clip_name}...", end=" ", flush=True)

        if video_path is None:
            print("SKIPPED (video file not found)")
            summary_rows.append({
                "clip": clip_name, "expected": expected, "detected": "N/A",
                "correct": "N/A", "confidence": "", "detection_frame": "",
                "detection_timestamp": "",
            })
            continue

        output_csv = os.path.join(results_dir, f"{clip_name}_log.csv")
        annotate_path = os.path.join(results_dir, f"{clip_name}_annotated.mp4") if annotate_all else None

        result = run(video_path, output_csv, annotate_path=annotate_path, quiet=True)

        detected = "fall" if result["fall_detected"] else "no_fall"
        correct = (detected == expected)

        print(f"expected={expected} detected={detected} {'✓' if correct else '✗ MISS'}")

        summary_rows.append({
            "clip": clip_name,
            "expected": expected,
            "detected": detected,
            "correct": correct,
            "confidence": result["max_confidence"],
            "detection_frame": result["detection_frame"] if result["detection_frame"] is not None else "",
            "detection_timestamp": result["detection_timestamp"] if result["detection_timestamp"] is not None else "",
        })

    summary_path = os.path.join(results_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["clip", "expected", "detected", "correct", "confidence",
                      "detection_frame", "detection_timestamp"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    # Quick scorecard printed to console
    scored = [r for r in summary_rows if r["correct"] != "N/A"]
    correct_count = sum(1 for r in scored if r["correct"])
    total = len(scored)
    skipped = len(summary_rows) - total

    print(f"\n{'='*50}")
    print(f"RESULTS: {correct_count}/{total} correct" + (f" ({skipped} skipped — file not found)" if skipped else ""))
    print(f"Summary written to {summary_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FallGuard against every labeled clip")
    parser.add_argument("--labels", default="data/labels.csv", help="Path to labels CSV (clip,expected)")
    parser.add_argument("--videos-dir", default="data/sample_videos", help="Directory containing video files")
    parser.add_argument("--results-dir", default="data/results", help="Directory to write per-clip logs + summary.csv")
    parser.add_argument("--annotate-all", action="store_true", help="Also generate annotated video for every clip (slow)")
    args = parser.parse_args()

    try:
        run_batch(args.labels, args.videos_dir, args.results_dir, annotate_all=args.annotate_all)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)