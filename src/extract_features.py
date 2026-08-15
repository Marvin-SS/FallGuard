"""
extract_features.py

Collapses each clip's per-frame log (data/results/<clip>_log.csv) into
ONE summary row of features, suitable for training a classifier. This
is the "turn a whole clip into ~8 numbers" step.

Features chosen based on what findings.txt identified as meaningful
across Entries 3-13:
    max_torso_angle        - highest angle reached (fall signature)
    max_hip_velocity       - highest downward velocity (fall signature)
    resting_torso_angle    - mean angle over the first few valid frames
                              (baseline — should be low for a healthy
                              side-view camera; Entry 7/10 showed this
                              reads elevated on bad cameras)
    torso_angle_std        - frame-to-frame volatility of angle (Entry 9
                              showed erratic swings on noisy cameras)
    dropout_rate           - fraction of frames with no landmarks at all
                              (Entry 9's high, scattered dropout signal)
    velocity_std           - frame-to-frame volatility of velocity
    angle_range             - max minus min angle (how much movement
                              happened at all)
    frames_above_velocity_threshold - count of frames where velocity
                              cleared 0.8, as a proxy for "how many
                              separate strong movements happened"

Usage:
    python -m src.extract_features
    python -m src.extract_features --results-dir data/results --output data/clip_features.csv
"""

import argparse
import csv
import glob
import os

import numpy as np
import pandas as pd


def extract_clip_features(log_path, n_baseline_frames=10, velocity_threshold=0.8):
    df = pd.read_csv(log_path)

    angle = pd.to_numeric(df["torso_angle"], errors="coerce")
    velocity = pd.to_numeric(df["hip_velocity"], errors="coerce")

    total_frames = len(df)
    valid_angle = angle.dropna()
    valid_velocity = velocity.dropna()

    if len(valid_angle) == 0:
        # Degenerate clip — no landmarks detected at all. Return NaNs;
        # caller can decide whether to drop or impute.
        return None

    resting_angle = valid_angle.iloc[:n_baseline_frames].mean()
    dropout_rate = 1.0 - (len(valid_angle) / total_frames)

    features = {
        "max_torso_angle": valid_angle.max(),
        "max_hip_velocity": valid_velocity.max() if len(valid_velocity) else 0.0,
        "resting_torso_angle": resting_angle,
        "torso_angle_std": valid_angle.std() if len(valid_angle) > 1 else 0.0,
        "velocity_std": valid_velocity.std() if len(valid_velocity) > 1 else 0.0,
        "dropout_rate": dropout_rate,
        "angle_range": valid_angle.max() - valid_angle.min(),
        "frames_above_velocity_threshold": int((valid_velocity >= velocity_threshold).sum()),
    }
    return features


def build_feature_table(results_dir, output_path):
    log_files = sorted(glob.glob(os.path.join(results_dir, "*_log.csv")))
    # Exclude summary.csv itself if it happens to match the glob pattern
    log_files = [f for f in log_files if not os.path.basename(f).startswith("summary")]

    rows = []
    skipped = []

    for log_path in log_files:
        clip_name = os.path.basename(log_path).replace("_log.csv", "")
        features = extract_clip_features(log_path)
        if features is None:
            skipped.append(clip_name)
            continue
        features["clip"] = clip_name
        rows.append(features)

    if not rows:
        print("No usable clip logs found — nothing written.")
        return

    out_df = pd.DataFrame(rows)
    # Put clip name first for readability
    cols = ["clip"] + [c for c in out_df.columns if c != "clip"]
    out_df = out_df[cols]
    out_df.to_csv(output_path, index=False)

    print(f"Extracted features for {len(rows)} clip(s) -> {output_path}")
    if skipped:
        print(f"Skipped {len(skipped)} clip(s) with no usable landmark data: {', '.join(skipped)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract per-clip summary features from pose logs")
    parser.add_argument("--results-dir", default="data/results", help="Directory containing *_log.csv files")
    parser.add_argument("--output", default="data/clip_features.csv", help="Output feature table path")
    args = parser.parse_args()

    build_feature_table(args.results_dir, args.output)