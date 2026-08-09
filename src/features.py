"""
features.py

Turns raw pose landmarks into the handful of signals that actually
distinguish a fall from sitting/bending/lying down on purpose:

    1. torso_angle       - angle of the shoulder-hip line vs vertical.
                            Upright standing ~0-20 deg. A fall usually
                            ends with the torso near-horizontal (~70-90 deg).
    2. bbox_aspect_ratio  - width/height of the pose bounding box.
                            Standing: tall & narrow (<1). Fallen: wide & short (>1).
    3. hip_vertical_velocity - how fast the hip midpoint is dropping,
                            normalized by frame height per second.
                            Falls have a sharp downward spike; sitting
                            down is much slower.
    4. torso_drop_ratio  - how much the hip has descended relative to
                            shoulder height, used to catch the "collapse"
                            shape distinct from a controlled sit.

Each function takes a PoseFrame (see pose_estimation.py) or a short
history of them, and returns a float or None if landmarks are missing.
"""

import math


def _midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def torso_angle_degrees(pose_frame):
    """
    Angle between the shoulder-hip line and the vertical axis.
    0 deg = perfectly upright. 90 deg = perfectly horizontal (lying down).
    """
    lm = pose_frame["landmarks"]
    if lm is None:
        return None

    shoulder_mid = _midpoint(lm["left_shoulder"], lm["right_shoulder"])
    hip_mid = _midpoint(lm["left_hip"], lm["right_hip"])

    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]

    # atan2(dx, dy): dy is the "vertical" component in image coords
    # (y grows downward), so this gives 0 when the torso line is vertical.
    angle_rad = math.atan2(abs(dx), abs(dy) + 1e-6)
    return math.degrees(angle_rad)


def bbox_aspect_ratio(pose_frame):
    bbox = pose_frame["bbox"]
    if bbox is None:
        return None
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    if height <= 1e-6:
        return None
    return width / height


def hip_midpoint_y(pose_frame):
    lm = pose_frame["landmarks"]
    if lm is None:
        return None
    return _midpoint(lm["left_hip"], lm["right_hip"])[1]


def hip_vertical_velocity(pose_frame_prev, pose_frame_curr):
    """
    Rate of change of hip height (normalized units per second).
    Positive = moving downward (toward the floor) in image coordinates.
    """
    y_prev = hip_midpoint_y(pose_frame_prev)
    y_curr = hip_midpoint_y(pose_frame_curr)
    if y_prev is None or y_curr is None:
        return None

    dt = pose_frame_curr["timestamp"] - pose_frame_prev["timestamp"]
    if dt <= 1e-6:
        return None

    return (y_curr - y_prev) / dt


def extract_feature_row(pose_frame_prev, pose_frame_curr):
    """
    Combine everything into one row for a given frame, using the
    previous frame for velocity. Returns a dict suitable for building
    a pandas DataFrame across a whole clip.
    """
    return {
        "frame_idx": pose_frame_curr["frame_idx"],
        "timestamp": pose_frame_curr["timestamp"],
        "torso_angle": torso_angle_degrees(pose_frame_curr),
        "aspect_ratio": bbox_aspect_ratio(pose_frame_curr),
        "hip_velocity": hip_vertical_velocity(pose_frame_prev, pose_frame_curr)
        if pose_frame_prev is not None
        else None,
    }
