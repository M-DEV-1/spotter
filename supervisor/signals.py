"""
Cheap ground-truth failure signals from MuJoCo sim state.
These are fed to the Cerebras supervisor for semantic interpretation.
"""
import mujoco
import numpy as np

# workspace bounding box — cube outside this is "out of region"
WORKSPACE_XY_MIN = np.array([-0.1, -0.8])
WORKSPACE_XY_MAX = np.array([ 1.2,  0.8])
FLOOR_Z = 0.01   # below this = on the floor


def cube_pos(model, data) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    return data.xpos[bid].copy()


def gripper_closed_empty(model, data) -> bool:
    """Gripper is commanded closed but cube is still on the floor."""
    gripper_closed = data.ctrl[7] < 0.01
    if not gripper_closed:
        return False
    return cube_pos(model, data)[2] < FLOOR_Z + 0.04


def cube_out_of_region(model, data) -> bool:
    """Cube has left the reachable XY workspace or fallen below the floor."""
    pos = cube_pos(model, data)
    in_xy = np.all(pos[:2] >= WORKSPACE_XY_MIN) and np.all(pos[:2] <= WORKSPACE_XY_MAX)
    return not in_xy or pos[2] < -0.05


class NoProgressTracker:
    """Detects when the cube XY hasn't moved for `patience` steps."""
    def __init__(self, patience: int = 300):
        self._patience = patience
        self._last_xy: np.ndarray | None = None
        self._stale = 0

    def reset(self):
        self._last_xy = None
        self._stale = 0

    def update(self, model, data) -> bool:
        xy = cube_pos(model, data)[:2].copy()
        if self._last_xy is not None:
            if np.linalg.norm(xy - self._last_xy) < 0.001:
                self._stale += 1
            else:
                self._stale = 0
        self._last_xy = xy
        return self._stale >= self._patience


def compute_signals(model, data, no_progress: bool = False) -> dict:
    """Return all failure signals as a plain dict (JSON-serialisable)."""
    return {
        "gripper_closed_empty": gripper_closed_empty(model, data),
        "cube_out_of_region": cube_out_of_region(model, data),
        "no_progress": no_progress,
    }
