"""
Pick-and-place task definition.
Sets up scene (cube start, place target), defines success predicate
and cheap failure signals for the supervisor.
"""
import mujoco
import numpy as np

# cube starts at (0.7, 0, 0.03) per scene XML default
CUBE_START = np.array([0.7, 0.0, 0.03])

# place target — where we set the mocap_target marker
PLACE_POS = np.array([0.45, 0.45, 0.03])

SUCCESS_XY_RADIUS = 0.08   # cube must be within this XY distance of PLACE_POS
SUCCESS_MIN_Z     = 0.01   # cube must have settled (not floating)
NO_PROGRESS_STEPS = 300    # steps with no cube XY movement → failure


def init_task(model, data) -> None:
    """Call once per episode after reset_to_keyframe."""
    data.mocap_pos[0] = PLACE_POS.copy()
    mujoco.mj_forward(model, data)


def cube_pos(model, data) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    return data.xpos[bid].copy()


def is_success(model, data) -> bool:
    pos = cube_pos(model, data)
    xy_dist = np.linalg.norm(pos[:2] - PLACE_POS[:2])
    return xy_dist < SUCCESS_XY_RADIUS and pos[2] < SUCCESS_MIN_Z + 0.04


def gripper_closed_empty(model, data) -> bool:
    """Failure signal: gripper is closed but cube is still near the floor."""
    gripper_open = data.ctrl[7] > 0.02
    if gripper_open:
        return False
    pos = cube_pos(model, data)
    return pos[2] < 0.05  # cube on ground while gripper is closed = missed


class ProgressTracker:
    """Detects no-progress timeout for the supervisor."""
    def __init__(self):
        self._last_xy = None
        self._stale_count = 0

    def update(self, model, data) -> bool:
        """Returns True if no cube XY movement for NO_PROGRESS_STEPS steps."""
        pos = cube_pos(model, data)
        xy = pos[:2].copy()
        if self._last_xy is not None:
            moved = np.linalg.norm(xy - self._last_xy)
            if moved < 0.001:
                self._stale_count += 1
            else:
                self._stale_count = 0
        self._last_xy = xy
        return self._stale_count >= NO_PROGRESS_STEPS
