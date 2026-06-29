"""
Classical keyframe-sequencing actor for Franka Panda pick-and-place.
Ctrl vectors are hand-tuned via scripts/tune_poses.py (mjviser web viewer).

Two retry paths:
  retry(params)                  — Gemma-direction path (j1/depth offset from parse_correction)
  retry_to_position(above, low)  — IK path (exact joint targets from simulator/ik.py)
"""
import numpy as np

# Hand-tuned ctrl vectors [joint1..joint7, gripper] — verified in viser.
ABOVE_CUBE = np.array([ 0.000,  0.626,  0.000, -1.547,  0.000,  2.225, -0.7143, 0.04])
LOWER      = np.array([ 0.000,  0.800,  0.000, -1.517,  0.000,  2.380, -0.7783, 0.04])
GRASP      = np.array([ 0.000,  0.800,  0.000, -1.517,  0.000,  2.380, -0.7783, 0.00])
LIFT       = np.array([ 0.000,  0.393,  0.000, -2.034,  0.000,  2.380, -0.7783, 0.00])
SWING      = np.array([ 0.792,  0.646,  0.000, -1.768,  0.000,  2.450, -0.7738, 0.00])
RELEASE    = np.array([ 0.792,  0.646,  0.000, -1.768,  0.000,  2.450, -0.7738, 0.04])
HOME       = np.array([ 0.000,  0.300,  0.000, -1.571,  0.000,  2.000, -0.785,  0.04])

# (ctrl, steps_to_lerp, phase_label)
SEQUENCE = [
    (ABOVE_CUBE, 200, "approach"),
    (LOWER,      200, "lower"),
    (GRASP,      120, "grasp"),
    (LIFT,       250, "lift"),
    (SWING,      300, "swing"),
    (RELEASE,     80, "release"),
    (HOME,       200, "return"),
]

# Snapshot of the original approach/lower entries so restore_sequence() can reset them
# after retry_to_position() patches SEQUENCE[0] and SEQUENCE[1] during an episode.
_BASE_SEQUENCE = SEQUENCE[:]

# phases that aim at the cube — direction-offset path applies here
_GRASP_PHASES = {"approach", "lower", "grasp"}


def _lerp(a, b, t):
    return a + (b - a) * np.clip(t, 0.0, 1.0)


class ClassicalActor:
    def reset(self):
        self.restore_sequence()   # undo any SEQUENCE patches from a previous episode
        self._phase = 0
        self._phase_step = 0
        self._prev = HOME.copy()
        self._j1_offset = 0.0     # direction path: from Gemma via parse_correction
        self._depth_offset = 0.0  # direction path: from Gemma via parse_correction
        self._ik_above = None     # IK path: exact joint targets from simulator/ik.py
        self._ik_lower = None     # IK path: exact joint targets from simulator/ik.py

    def restore_sequence(self) -> None:
        """Restore SEQUENCE[0] and SEQUENCE[1] to the original hand-tuned vectors."""
        SEQUENCE[0] = _BASE_SEQUENCE[0]
        SEQUENCE[1] = _BASE_SEQUENCE[1]

    def act(self, obs=None) -> np.ndarray:
        if self._phase >= len(SEQUENCE):
            return HOME.copy()

        target_raw, duration, name = SEQUENCE[self._phase]

        # IK path: use solved joint targets for approach and lower phases
        if self._ik_above is not None and name == "approach":
            target = np.concatenate([self._ik_above, [0.04]])  # gripper open
        elif self._ik_lower is not None and name in ("lower", "grasp"):
            gripper = 0.04 if name == "lower" else 0.00
            target = np.concatenate([self._ik_lower, [gripper]])
        # Direction-offset path
        elif name in _GRASP_PHASES and (self._j1_offset != 0.0 or self._depth_offset != 0.0):
            target = target_raw.copy()
            target[0] += self._j1_offset
            if name == "lower":
                # positive depth_offset → increase joint2 → arm descends further
                target[1] = np.clip(target[1] + self._depth_offset, 0.6, 1.1)
        else:
            target = target_raw

        t = self._phase_step / max(duration, 1)
        ctrl = _lerp(self._prev, target, t)

        self._phase_step += 1
        if self._phase_step >= duration:
            self._prev = target.copy()
            self._phase += 1
            self._phase_step = 0

        return ctrl

    def done(self) -> bool:
        return self._phase >= len(SEQUENCE)

    def retry(self, params=None) -> None:
        """Restart from approach using RetryParams derived from Gemma's correction.
        Causal chain: Gemma text → parse_correction() → RetryParams → here.
        No sim state is read."""
        if params is not None:
            self._j1_offset = float(params.j1_offset)
            self._depth_offset = float(params.depth_offset)
        self._ik_above = None
        self._ik_lower = None
        self._phase = 0
        self._phase_step = 0

    def retry_to_position(
        self,
        above_joints: np.ndarray,
        lower_joints: np.ndarray,
    ) -> None:
        """IK path: restart with exact joint targets computed by simulator/ik.py.
        Gemma triggers this (causally necessary); sim state provides WHERE via IK.
        above_joints and lower_joints replace ABOVE_CUBE and LOWER in the sequence."""
        self._ik_above = above_joints.copy()
        self._ik_lower = lower_joints.copy()
        self._j1_offset = 0.0
        self._depth_offset = 0.0
        self._phase = 0
        self._phase_step = 0

    @property
    def phase_name(self) -> str:
        if self._phase >= len(SEQUENCE):
            return "done"
        return SEQUENCE[self._phase][2]
