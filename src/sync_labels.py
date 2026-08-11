"""
sync_labels.py

Scans data/sample_videos/ for video files and adds any that are
missing from data/labels.csv, guessing the expected label from the
filename prefix:
    fall-*  -> fall
    adl-*   -> no_fall
    anything else -> left blank, flagged for you to fill in by hand

This does NOT remove or change any existing rows — it only appends
new ones for videos it hasn't seen before. Existing rows (including
any labels you've hand-corrected) are left exactly as they are.

Usage:
    python -m src.sync_labels
    python -m src.sync_labels --labels data/labels.csv --videos-dir data/sample_videos
"""

import argparse
import csv
import os

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")


def guess_label(clip_name):
    if clip_name.startswith("fall-"):
        return "fall"
    elif clip_name.startswith("adl-"):
        return "no_fall"
    else:
        return ""  # unknown prefix — leave blank rather than guess wrong


def sync_labels(labels_path, videos_dir):
    # Load existing rows (preserve exactly as-is, including any manual edits)
    existing_clips = set()
    rows = []
    if os.path.exists(labels_path):
        with open(labels_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                existing_clips.add(row["clip"].strip())

    # Find video files not yet represented in labels.csv
    added = []
    if os.path.isdir(videos_dir):
        for filename in sorted(os.listdir(videos_dir)):
            name, ext = os.path.splitext(filename)
            if ext.lower() not in VIDEO_EXTENSIONS:
                continue
            if name in existing_clips:
                continue
            label = guess_label(name)
            rows.append({"clip": name, "expected": label})
            added.append((name, label))

    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "expected"])
        writer.writeheader()
        writer.writerows(rows)

    if not added:
        print("No new video files found — labels.csv already up to date.")
    else:
        print(f"Added {len(added)} new row(s) to {labels_path}:")
        for name, label in added:
            flag = "  <-- please fill in manually" if not label else ""
            print(f"  {name},{label}{flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add missing video files to labels.csv")
    parser.add_argument("--labels", default="data/labels.csv", help="Path to labels CSV")
    parser.add_argument("--videos-dir", default="data/sample_videos", help="Directory containing video files")
    args = parser.parse_args()

    sync_labels(args.labels, args.videos_dir)