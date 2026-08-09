"""
run_pipeline.py

End-to-end runner: video file -> pose estimation -> features -> fall
detection -> CSV log + optional annotated video output.

Usage:
    python -m src.run_pipeline --video data/sample_videos/fall_01.mp4 \
        --output data/results/fall_01_log.csv \
        --annotate data/results/fall_01_annotated.mp4

This is your integration test: if this runs end to end on one clip,
your whole week-1 pipeline is proven out.
"""

import argparse
import csv
import sys

import cv2

from src.pose_estimation import PoseEstimator, iter_video_poses, KEY_LANDMARKS
from src.features import extract_feature_row
from src.fall_detector import FallDetector


def draw_annotations(frame, pose_frame, detector_result):
    """Draw keypoints + current fall state on a frame (for a visual demo)."""
    h, w = frame.shape[:2]

    if pose_frame["landmarks"]:
        for name, (x, y, z, vis) in pose_frame["landmarks"].items():
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    state = detector_result["state"]
    confidence = detector_result["confidence"]
    color = (0, 0, 255) if state == "fall_confirmed" else (0, 255, 255) if state == "impact_detected" else (0, 255, 0)
    label = f"{state.upper()} ({confidence:.2f})"
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame


def run(video_path, output_csv, annotate_path=None, quiet=False):
    """
    Runs the full pipeline on one video.
    Returns a summary dict:
        {
            "frames_processed": int,
            "fall_detected": bool,
            "detection_frame": int or None,
            "detection_timestamp": float or None,
            "max_confidence": float,
        }
    """
    estimator = PoseEstimator()
    detector = FallDetector()

    writer = None
    if annotate_path:
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(annotate_path, fourcc, fps, (width, height))
        cap.release()  # we'll re-read frames below via a dedicated capture, in lockstep

    prev_pose_frame = None
    rows = []

    annotate_cap = cv2.VideoCapture(video_path) if annotate_path else None

    fall_detected = False
    detection_frame = None
    detection_timestamp = None
    max_confidence = 0.0

    for pose_frame in iter_video_poses(video_path, estimator=estimator):
        feature_row = extract_feature_row(prev_pose_frame, pose_frame)
        result = detector.update(feature_row)

        row = {**feature_row, "state": result["state"], "confidence": result["confidence"]}
        rows.append(row)

        max_confidence = max(max_confidence, result["confidence"])

        if result["state"] == "fall_confirmed" and not fall_detected:
            fall_detected = True
            detection_frame = row["frame_idx"]
            detection_timestamp = row["timestamp"]
            if not quiet:
                print(f"[ALERT] Fall confirmed at frame {row['frame_idx']} "
                      f"(t={row['timestamp']:.2f}s, confidence={result['confidence']})")

        if annotate_path and annotate_cap is not None:
            ret, frame = annotate_cap.read()
            if ret:
                annotated = draw_annotations(frame, pose_frame, result)
                writer.write(annotated)

        prev_pose_frame = pose_frame

    estimator.close()
    if writer is not None:
        writer.release()
    if annotate_cap is not None:
        annotate_cap.release()

    with open(output_csv, "w", newline="") as f:
        fieldnames = ["frame_idx", "timestamp", "torso_angle", "aspect_ratio",
                      "hip_velocity", "state", "confidence"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if not quiet:
        print(f"\nProcessed {len(rows)} frames. Log written to {output_csv}")
        if annotate_path:
            print(f"Annotated video written to {annotate_path}")

    return {
        "frames_processed": len(rows),
        "fall_detected": fall_detected,
        "detection_frame": detection_frame,
        "detection_timestamp": detection_timestamp,
        "max_confidence": round(max_confidence, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FallGuard on a video file")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output CSV log")
    parser.add_argument("--annotate", default=None, help="Optional path for annotated output video")
    args = parser.parse_args()

    try:
        run(args.video, args.output, args.annotate)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)