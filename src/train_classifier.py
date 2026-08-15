"""
train_classifier.py

Trains a small classifier (Random Forest) on the per-clip summary
features from extract_features.py, using LEAVE-ONE-OUT cross-
validation — chosen specifically because the dataset is small
(~20 clips). A normal train/test split would leave too few clips in
the test set to mean anything; LOOCV trains on every-clip-but-one and
predicts that one, repeated for every clip, so every single clip gets
used as a genuine held-out test case at some point without wasting
any data.

This is deliberately transparent, not just an accuracy number: it
prints a per-clip prediction table (same spirit as batch_runner.py's
scorecard) so you can see exactly which clips the model got right or
wrong, plus feature importances so the result is explainable.

Usage:
    python -m src.train_classifier
    python -m src.train_classifier --features data/clip_features.csv --labels data/labels.csv
"""

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut


FEATURE_COLUMNS = [
    "max_torso_angle",
    "max_hip_velocity",
    "resting_torso_angle",
    "torso_angle_std",
    "velocity_std",
    "dropout_rate",
    "angle_range",
    "frames_above_velocity_threshold",
]


def load_dataset(features_path, labels_path):
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    merged = features_df.merge(labels_df, on="clip", how="inner")

    missing_labels = set(features_df["clip"]) - set(labels_df["clip"])
    if missing_labels:
        print(f"Warning: {len(missing_labels)} clip(s) have features but no label, excluded: {', '.join(sorted(missing_labels))}")

    missing_features = set(labels_df["clip"]) - set(features_df["clip"])
    if missing_features:
        print(f"Warning: {len(missing_features)} labeled clip(s) have no extracted features (run extract_features.py first?), excluded: {', '.join(sorted(missing_features))}")

    unlabeled_blank = merged[merged["expected"].isna() | (merged["expected"] == "")]
    if len(unlabeled_blank):
        print(f"Warning: {len(unlabeled_blank)} clip(s) have a blank label, excluded: {', '.join(unlabeled_blank['clip'])}")
        merged = merged.dropna(subset=["expected"])
        merged = merged[merged["expected"] != ""]

    return merged


def run_loocv(dataset):
    X = dataset[FEATURE_COLUMNS].values
    y = (dataset["expected"] == "fall").astype(int).values
    clips = dataset["clip"].values

    loo = LeaveOneOut()
    predictions = []
    correct_count = 0

    for train_idx, test_idx in loo.split(X):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])[0]

        clip_name = clips[test_idx][0]
        actual = y[test_idx][0]
        correct = (pred == actual)
        correct_count += int(correct)

        predictions.append({
            "clip": clip_name,
            "expected": "fall" if actual == 1 else "no_fall",
            "predicted": "fall" if pred == 1 else "no_fall",
            "correct": correct,
        })

    accuracy = correct_count / len(clips)
    return predictions, accuracy


def print_feature_importances(dataset):
    X = dataset[FEATURE_COLUMNS].values
    y = (dataset["expected"] == "fall").astype(int).values

    # Fit one final model on ALL data just to inspect feature importances
    # (not used for prediction — LOOCV above is the actual evaluation)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    importances = sorted(zip(FEATURE_COLUMNS, clf.feature_importances_), key=lambda x: -x[1])
    print("\nFeature importances (fit on full dataset, for interpretability only):")
    for name, importance in importances:
        bar = "#" * int(importance * 50)
        print(f"  {name:<35} {importance:.3f}  {bar}")


def main(features_path, labels_path):
    dataset = load_dataset(features_path, labels_path)

    if len(dataset) < 3:
        print(f"Only {len(dataset)} labeled clip(s) with features available — need at least a few to train anything meaningful. "
              f"Run extract_features.py and check labels.csv.")
        return

    print(f"Training on {len(dataset)} labeled clips using leave-one-out cross-validation...\n")

    predictions, accuracy = run_loocv(dataset)

    print(f"{'clip':<25} {'expected':<10} {'predicted':<10} {'correct'}")
    print("-" * 60)
    for p in predictions:
        mark = "OK" if p["correct"] else "WRONG"
        print(f"{p['clip']:<25} {p['expected']:<10} {p['predicted']:<10} {mark}")

    print(f"\n{'='*50}")
    print(f"LOOCV ACCURACY: {accuracy:.1%} ({sum(p['correct'] for p in predictions)}/{len(predictions)} correct)")
    print(f"{'='*50}")

    print_feature_importances(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier on clip features")
    parser.add_argument("--features", default="data/clip_features.csv", help="Path to extracted features CSV")
    parser.add_argument("--labels", default="data/labels.csv", help="Path to labels CSV")
    args = parser.parse_args()

    main(args.features, args.labels)