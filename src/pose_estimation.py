"""
pose_estimation.py

Wraps MediaPipe Pose so the rest of the pipeline never has to think
about the underlying CV library. If you swap MediaPipe for YOLO-pose
or OpenPose later, this is the ONLY file that should need to change.

Output contract for the rest of the pipeline:
    A "PoseFrame" is a dict:
        {
            "frame_idx": int,
            "timestamp": float,         # seconds
            "landmarks": dict[str, (x, y, z, visibility)] or None,
            "bbox": (x_min, y_min, x_max, y_max) or None
        }
    Coordinates are normalized [0, 1] relative to frame width/height,
    matching MediaPipe's native output.
"""

import os
# Suppress noisy TensorFlow Lite / absl startup logs (delegate creation,
# signature warnings, etc.) — these are harmless but drown out our own
# [ALERT] / progress output. Must be set BEFORE mediapipe is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# The subset of MediaPipe's 33 landmarks we actually care about for
# fall detection. Keeping this explicit (instead of using all 33)
# makes the feature engineering step easier to reason about.
KEY_LANDMARKS = {
    "nose": mp_pose.PoseLandmark.NOSE,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
}


class PoseEstimator:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr, frame_idx, timestamp):
        """
        Run pose estimation on a single BGR frame (as returned by cv2.VideoCapture).
        Returns a PoseFrame dict (see module docstring).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "landmarks": None,
                "bbox": None,
            }

        lm = results.pose_landmarks.landmark
        landmarks = {}
        xs, ys = [], []
        for name, idx in KEY_LANDMARKS.items():
            point = lm[idx.value]
            landmarks[name] = (point.x, point.y, point.z, point.visibility)
            xs.append(point.x)
            ys.append(point.y)

        bbox = (min(xs), min(ys), max(xs), max(ys))

        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "landmarks": landmarks,
            "bbox": bbox,
        }

    def close(self):
        self.pose.close()


def iter_video_poses(video_path, estimator=None, frame_skip=0):
    """
    Generator that yields a PoseFrame for every processed frame of a video file.

    frame_skip: process every (frame_skip + 1)th frame. Use this to trade
    temporal resolution for speed once you're tuning on long clips.
    """
    owns_estimator = estimator is None
    estimator = estimator or PoseEstimator()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_skip == 0 or frame_idx % (frame_skip + 1) == 0:
                timestamp = frame_idx / fps
                yield estimator.process_frame(frame, frame_idx, timestamp)

            frame_idx += 1
    finally:
        cap.release()
        if owns_estimator:
            estimator.close()