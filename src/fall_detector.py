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
                                               # (used for the PRIMARY/angle-gated
                                               # entry path — that path's falls
                                               # have settled quickly in every
                                               # case seen so far, ~10-13 frames)
    impact_window_frames_velocity_only: int = 35  # separate, longer window used
                                               # when entry was via the velocity-
                                               # only fallback. Empirically needed:
                                               # fall-05-cam1 (findings.txt Entry 11)
                                               # only reaches stillness confirmation
                                               # at frame ~126 after impact at frame
                                               # ~100 (26 frames) — this camera's
                                               # post-impact velocity decays much
                                               # more slowly than side-view falls,
                                               # so the fallback path (already
                                               # operating with less corroboration)
                                               # needs more patience before timing
                                               # out, not less.
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
    velocity_spike_window_frames: int = 20    # how many frames a velocity spike
                                               # "counts" for, even after velocity
                                               # itself has dropped back down — peak
                                               # downward velocity and peak torso
                                               # angle often don't land on the same
                                               # frame (person speeds up while still
                                               # rotating, then decelerates on impact
                                               # as rotation finishes), so requiring
                                               # both simultaneously misses real falls
    velocity_only_threshold: float = 1.0      # if set, a velocity spike at or above
                                               # THIS value (stricter than
                                               # velocity_spike_threshold) can trigger
                                               # impact on its own, with no angle
                                               # confirmation needed. This exists
                                               # because angle has repeatedly failed
                                               # to open the gate for real falls
                                               # (foreshortened trajectories, bad
                                               # camera geometry — see findings.txt
                                               # Entries 4, 8, 11) while velocity kept
                                               # reading correctly every single time.
                                               # Set to None to disable this path
                                               # entirely (falls back to the original
                                               # angle-gated-only behavior).
    velocity_only_sustained_frames: int = 2   # velocity must clear
                                               # velocity_only_threshold for this many
                                               # CONSECUTIVE frames before the fallback
                                               # fires. Added after a real false
                                               # positive: fall-01-cam1 has known noisy
                                               # tracking (Entry 4), and a single wild
                                               # frame (velocity=1.07 immediately
                                               # followed by -1.68 — clearly a tracking
                                               # glitch, not real motion) falsely
                                               # triggered the fallback and then
                                               # blocked the clip's ACTUAL fall later
                                               # in the same episode via the extended
                                               # timeout window. Every other trigger
                                               # path in this detector requires
                                               # sustained evidence; this one needs it
                                               # too, given this exact camera's known
                                               # noise profile.


class FallDetector:
    def __init__(self, config: FallDetectorConfig = None):
        self.config = config or FallDetectorConfig()
        self.state = FallState.NORMAL
        self.frames_since_impact = 0
        self.settled_frame_count = 0
        self.recovery_frame_count = 0
        self.still_frame_count = 0
        self.velocity_spike_countdown = 0
        self.velocity_only_sustained_count = 0
        self.entered_via_velocity_only = False  # tracks which path triggered
                                                  # IMPACT_DETECTED — controls
                                                  # whether angle-based recovery
                                                  # is trusted (see update())
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

        # Track a recent velocity spike in a sliding window, independent of
        # whether angle data is available this frame. This runs BEFORE the
        # missing-data check below on purpose — velocity can spike on a
        # frame where landmarks are otherwise fine, and we don't want a
        # later occlusion gap to erase that memory.
        if velocity is not None and velocity >= cfg.velocity_spike_threshold:
            self.velocity_spike_countdown = cfg.velocity_spike_window_frames
        elif self.velocity_spike_countdown > 0:
            self.velocity_spike_countdown -= 1

        # Track consecutive frames clearing the stricter fallback threshold.
        # Also tracked before the missing-data check so a single glitchy
        # frame surrounded by gaps doesn't accidentally look "sustained".
        if (cfg.velocity_only_threshold is not None and velocity is not None
                and velocity >= cfg.velocity_only_threshold):
            self.velocity_only_sustained_count += 1
        else:
            self.velocity_only_sustained_count = 0

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
            # PRIMARY PATH: a velocity spike within the last
            # `velocity_spike_window_frames` PLUS the angle threshold being
            # crossed now (even if that happens on a later frame than the
            # spike itself) counts as an impact. This is what catches falls
            # where the person is still rotating downward when velocity
            # peaks, and only finishes tipping past the angle threshold a
            # beat later.
            if angle >= cfg.torso_angle_fall_threshold and self.velocity_spike_countdown > 0:
                self.state = FallState.IMPACT_DETECTED
                self.frames_since_impact = 0
                self.settled_frame_count = 0
                self.recovery_frame_count = 0
                self.entered_via_velocity_only = False
                confidence = 0.4  # impact alone is a weak signal

            # FALLBACK PATH: velocity alone, no angle confirmation needed.
            # Only used if the primary path didn't already trigger this
            # frame. Starts at a LOWER confidence than the primary path
            # (0.3 vs 0.4) since it's a weaker basis on its own. Marks
            # entered_via_velocity_only=True so the recovery check below
            # knows NOT to trust angle for this episode — we already know
            # angle is unreliable on whatever camera produced this trigger
            # (that's the whole reason this path exists), so using angle to
            # CANCEL the fall would undo the exact problem we just fixed.
            # Confirmation still requires the stillness signal or settling
            # via angle IF angle happens to become reliable later — this
            # only disables angle-based CANCELLATION, not confirmation.
            elif (cfg.velocity_only_threshold is not None
                  and self.velocity_only_sustained_count >= cfg.velocity_only_sustained_frames):
                self.state = FallState.IMPACT_DETECTED
                self.frames_since_impact = 0
                self.settled_frame_count = 0
                self.recovery_frame_count = 0
                self.entered_via_velocity_only = True
                confidence = 0.3

        elif self.state == FallState.IMPACT_DETECTED:
            self.frames_since_impact += 1

            # --- Signal 1: angle staying near-horizontal ---
            # Only trust angle for settling/recovery if we got here via the
            # PRIMARY path (angle already proved reliable enough to trigger
            # in the first place). If we entered via the velocity-only
            # fallback, angle is known-unreliable for this camera — don't
            # let it cancel a fall it was never trusted to detect.
            if not self.entered_via_velocity_only:
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

            active_window = (cfg.impact_window_frames_velocity_only
                              if self.entered_via_velocity_only
                              else cfg.impact_window_frames)
            if self.frames_since_impact > active_window and self.state != FallState.FALL_CONFIRMED:
                # Impact happened but never settled into a fall shape — reset
                self.state = FallState.NORMAL
                self.entered_via_velocity_only = False
                confidence = 0.0

        elif self.state == FallState.FALL_CONFIRMED:
            confidence = 0.95
            # Same guard as above — don't use angle to decide "they got back
            # up" if angle was never trusted to detect this fall in the
            # first place.
            if not self.entered_via_velocity_only:
                if angle < cfg.torso_angle_recovery_threshold:
                    self.recovery_frame_count += 1
                    if self.recovery_frame_count >= cfg.recovery_frames_required:
                        # Sustained low angle — person actually got back up
                        self.state = FallState.NORMAL
                        self.settled_frame_count = 0
                        self.recovery_frame_count = 0
                        self.entered_via_velocity_only = False
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
        self.velocity_spike_countdown = 0
        self.velocity_only_sustained_count = 0
        self.entered_via_velocity_only = False
        self.history.clear()

    # NOTE / known limitation: once FALL_CONFIRMED is reached via the
    # velocity-only fallback path, this detector currently has no way to
    # un-confirm if the person genuinely gets back up (since we deliberately
    # don't trust angle for recovery on that path, and there's no
    # velocity-based "they're moving normally again" check yet). In
    # practice this matches how a real alert system would likely work
    # anyway — a confirmed fall alert would need a human to acknowledge/
    # clear it, not silently auto-cancel. Flagged here in case that
    # assumption needs revisiting later.