"""
Classical keyframe-sequencing actor for Franka Panda pick-and-place.
Ctrl vectors are hand-tuned via scripts/tune_poses.py (mjviser web viewer).
"""
import numpy as np

# Hand-tuned ctrl vectors [joint1..joint7, gripper] — verified in viser.
ABOVE_CUBE = np.array([ 0.000,  0.626,  0.000, -1.547,  0.000,  2.225, -0.7143, 0.04])
LOWER      = np.array([ 0.000,  0.800,  0.000, -1.517,  0.000,  2.380, -0.7783, 0.04])
GRASP      = np.array([ 0.000,  0.800,  0.000, -1.517,  0.000,  2.380, -0.7783, 0.00])  # LOWER + close
LIFT       = np.array([ 0.000,  0.393,  0.000, -2.034,  0.000,  2.380, -0.7783, 0.00])
SWING      = np.array([ 0.792,  0.646,  0.000, -1.768,  0.000,  2.450, -0.7738, 0.00])  # RELEASE + closed
RELEASE    = np.array([ 0.792,  0.646,  0.000, -1.768,  0.000,  2.450, -0.7738, 0.04])
HOME       = np.array([ 0.000,  0.300,  0.000, -1.571,  0.000,  2.000, -0.785,  0.04])

# (ctrl, steps_to_lerp, phase_label)
SEQUENCE = [
    (ABOVE_CUBE, 200, "approach"),
    (LOWER,      200, "lower"),    # descend around cube, gripper open
    (GRASP,      120, "grasp"),    # close fingers on cube
    (LIFT,       250, "lift"),     # raise cube clear
    (SWING,      300, "swing"),    # carry to place zone, gripper closed
    (RELEASE,     80, "release"),  # open gripper, cube drops
    (HOME,       200, "return"),
]


def _lerp(a, b, t):
    return a + (b - a) * np.clip(t, 0.0, 1.0)


class ClassicalActor:
    def reset(self):
        self._step = 0
        self._phase = 0
        self._phase_step = 0
        self._prev = HOME.copy()

    def act(self, obs=None) -> np.ndarray:
        if self._phase >= len(SEQUENCE):
            return HOME.copy()

        target, duration, _ = SEQUENCE[self._phase]
        t = self._phase_step / max(duration, 1)
        ctrl = _lerp(self._prev, target, t)

        self._step += 1
        self._phase_step += 1
        if self._phase_step >= duration:
            self._prev = target.copy()
            self._phase += 1
            self._phase_step = 0

        return ctrl

    def done(self) -> bool:
        return self._phase >= len(SEQUENCE)

    def retry(self) -> None:
        """Restart from approach, lerping smoothly from current arm position."""
        self._phase = 0
        self._phase_step = 0
        # _prev stays as current ctrl so lerp is continuous

    @property
    def phase_name(self) -> str:
        if self._phase >= len(SEQUENCE):
            return "done"
        return SEQUENCE[self._phase][2]
