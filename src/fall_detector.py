"""
fall_detector.py

Rule-based fall classifier (v1 of the project — see README for the
plan to swap this for a trained model later).

Why rule-based first:
    - Zero training data required to get a working demo
    - Every threshold is explainable ("we flag a fall when torso angle
      exceeds X and hip velocity exceeds Y") which is exactly what a
      recruiter/interviewer will ask about
    - Gives you a baseline to measure a future ML model against

Design: a fall isn't a single frame, it's a SEQUENCE:
    1. IMPACT phase   - sudden spike in downward hip velocity +
                         rapidly increasing torso angle
    2. SETTLED phase  - torso angle stays high (near-horizontal) and
                         doesn't recover for N consecutive frames

Requiring both phases is what separates a real fall from:
    - Sitting down quickly (impact-like velocity, but torso angle
      recovers to a seated ~20-40 deg, not fully horizontal)
    - Bending to pick something up (torso angle rises, but no
      downward velocity spike, and it recovers within a second)
    - Lying down on purpose (torso angle goes high, but WITHOUT the
      preceding velocity spike — no impact phase)
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum


class FallState(Enum):
    NORMAL = "normal"
    IMPACT_DETECTED = "impact_detected"
    FALL_CONFIRMED = "fall_confirmed"


@dataclass
class FallDetectorConfig:
    # Tune these against your dataset — see notebooks/threshold_tuning.ipynb
    velocity_spike_threshold: float = 0.8    # normalized hip-height units/sec
    torso_angle_fall_threshold: float = 60.0  # degrees from vertical
    torso_angle_recovery_threshold: float = 40.0  # below this = "recovered"
    settled_frames_required: int = 10         # ~0.3-0.5s at 20-30fps
    impact_window_frames: int = 15            # how long after impact to confirm
    recovery_frames_required: int = 5         # consecutive low-angle frames needed
                                               # before we believe a "recovery" —
                                               # prevents one noisy/occluded frame
                                               # (e.g. foreshortening when prone)
                                               # from cancelling a real fall
    stillness_velocity_threshold: float = 0.2  # |hip_velocity| below this = "not moving"
                                               # (0.15 was too tight — real post-impact
                                               # settling motion briefly exceeds it;
                                               # see notebooks/threshold_tuning.ipynb)
    stillness_frames_required: int = 4        # consecutive still frames post-impact
                                               # that confirm a fall WITHOUT relying
                                               # on torso angle at all — a person who
                                               # actually fell typically stops moving;
                                               # this is the fallback for when angle
                                               # reads unreliably (see foreshortening
                                               # note above)


class FallDetector:
    def __init__(self, config: FallDetectorConfig = None):
        self.config = config or FallDetectorConfig()
        self.state = FallState.NORMAL
        self.frames_since_impact = 0
        self.settled_frame_count = 0
        self.recovery_frame_count = 0
        self.still_frame_count = 0
        self.history = deque(maxlen=30)  # rolling buffer for debugging/logging

    def update(self, feature_row: dict) -> dict:
        """
        Feed one feature row (from features.extract_feature_row) at a time.
        Returns a dict describing current state + confidence, e.g.:
            {"state": "fall_confirmed", "confidence": 0.87, "frame_idx": 142}
        """
        self.history.append(feature_row)
        cfg = self.config

        angle = feature_row.get("torso_angle")
        velocity = feature_row.get("hip_velocity")

        confidence = 0.0

        if angle is None or velocity is None:
            # Missing pose data (occlusion, person left frame, person now
            # prone and hard to track, etc.) — this is EXPECTED right after
            # a real fall, since hitting the ground is exactly when MediaPipe
            # is most likely to lose clean landmarks. Hold current state and
            # confidence rather than resetting or advancing counters.
            if self.state == FallState.IMPACT_DETECTED:
                confidence = min(0.4 + 0.5 * (self.settled_frame_count / cfg.settled_frames_required), 0.9)
            elif self.state == FallState.FALL_CONFIRMED:
                confidence = 0.95
            return self._result(confidence)

        if self.state == FallState.NORMAL:
            if velocity >= cfg.velocity_spike_threshold and angle >= cfg.torso_angle_fall_threshold:
                self.state = FallState.IMPACT_DETECTED
                self.frames_since_impact = 0
                self.settled_frame_count = 0
                self.recovery_frame_count = 0
                confidence = 0.4  # impact alone is a weak signal

        elif self.state == FallState.IMPACT_DETECTED:
            self.frames_since_impact += 1

            # --- Signal 1: angle staying near-horizontal ---
            if angle >= cfg.torso_angle_fall_threshold:
                self.settled_frame_count += 1
                self.recovery_frame_count = 0
            elif angle < cfg.torso_angle_recovery_threshold:
                self.recovery_frame_count += 1
            else:
                self.recovery_frame_count = 0  # ambiguous middle ground

            # --- Signal 2: lack of movement, independent of angle ---
            # This is what actually catches the case where a fallen person
            # reads a low torso angle due to camera-angle foreshortening —
            # they're still not moving, which angle alone can't tell us.
            if abs(velocity) < cfg.stillness_velocity_threshold:
                self.still_frame_count += 1
            else:
                self.still_frame_count = 0

            # Confirmation check FIRST — either signal reaching its threshold
            # confirms the fall and takes priority over a recovery reset.
            if (self.settled_frame_count >= cfg.settled_frames_required or
                    self.still_frame_count >= cfg.stillness_frames_required):
                self.state = FallState.FALL_CONFIRMED
                self.recovery_frame_count = 0
                confidence = 0.95
                return self._result(confidence)

            # Only now check recovery — and only trust it after several
            # consecutive low-angle frames, since a single frame can be
            # noise from foreshortening or a tracking glitch right after
            # occlusion, not a real recovery.
            if self.recovery_frame_count >= cfg.recovery_frames_required:
                self.state = FallState.NORMAL
                self.settled_frame_count = 0
                self.recovery_frame_count = 0
                self.still_frame_count = 0
                confidence = 0.0
                return self._result(confidence)

            confidence = min(0.4 + 0.5 * (self.settled_frame_count / cfg.settled_frames_required), 0.9)

            if self.frames_since_impact > cfg.impact_window_frames and self.state != FallState.FALL_CONFIRMED:
                # Impact happened but never settled into a fall shape — reset
                self.state = FallState.NORMAL
                confidence = 0.0

        elif self.state == FallState.FALL_CONFIRMED:
            confidence = 0.95
            if angle < cfg.torso_angle_recovery_threshold:
                self.recovery_frame_count += 1
                if self.recovery_frame_count >= cfg.recovery_frames_required:
                    # Sustained low angle — person actually got back up
                    self.state = FallState.NORMAL
                    self.settled_frame_count = 0
                    self.recovery_frame_count = 0
                    confidence = 0.0
            else:
                self.recovery_frame_count = 0

        return self._result(confidence)

    def _result(self, confidence):
        return {
            "frame_idx": self.history[-1]["frame_idx"] if self.history else None,
            "state": self.state.value,
            "confidence": round(confidence, 2),
        }

    def reset(self):
        self.state = FallState.NORMAL
        self.frames_since_impact = 0
        self.settled_frame_count = 0
        self.recovery_frame_count = 0
        self.history.clear()