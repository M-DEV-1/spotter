"""
Jacobian pseudoinverse IK solver for Franka Panda in MuJoCo.

Finds joint angles that bring a named site to a target 3D position.
Operates on a COPY of data — never disturbs the running simulation.

Site name: "panda/hand" (find with: python3 -c "import mujoco; from simulator.scene import
load_model, make_data; m=load_model(); [print(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i))
for i in range(m.nsite)]")
"""
import copy
import numpy as np
import mujoco

# Damping term for numerical stability (damped least-squares)
_LAMBDA_SQ = 1e-4
# Workspace safety bounds for cube recovery
_CUBE_BOUNDS = {
    "x": (0.25, 1.05),
    "y": (-0.55, 0.55),
    "z": (-0.05, 0.20),
}
_JOINT_INDICES = slice(0, 7)   # first 7 joints = Panda arm (not gripper)


def _site_id(model: mujoco.MjModel, site_name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(
            f"Site '{site_name}' not found. Available sites: "
            + ", ".join(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
                for i in range(model.nsite)
            )
        )
    return sid


def within_workspace(pos: np.ndarray) -> bool:
    """Return True if pos (xyz) is within the reachable cube workspace."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    return (
        _CUBE_BOUNDS["x"][0] <= x <= _CUBE_BOUNDS["x"][1]
        and _CUBE_BOUNDS["y"][0] <= y <= _CUBE_BOUNDS["y"][1]
        and _CUBE_BOUNDS["z"][0] <= z <= _CUBE_BOUNDS["z"][1]
    )


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    site_name: str = "gripper",
    max_iters: int = 150,
    tol: float = 5e-3,
    step_size: float = 0.4,
) -> np.ndarray | None:
    """
    Iterative Jacobian IK. Returns 7-element joint position array or None if failed.
    Does NOT modify the live data — works on a deepcopy.

    Args:
        target_pos: desired xyz position for the site
        site_name:  MuJoCo site to drive to target
        max_iters:  iteration cap
        tol:        convergence threshold (metres)
        step_size:  Jacobian step scale (smaller = more stable, slower)
    """
    if not within_workspace(target_pos):
        return None

    d = copy.copy(data)
    sid = _site_id(model, site_name)
    nv = model.nv

    for _ in range(max_iters):
        mujoco.mj_forward(model, d)
        current = d.site_xpos[sid].copy()
        error = target_pos - current

        if np.linalg.norm(error) < tol:
            return d.qpos[_JOINT_INDICES].copy()

        jacp = np.zeros((3, nv))
        mujoco.mj_jacSite(model, d, jacp, None, sid)
        J = jacp[:, _JOINT_INDICES]  # (3, 7)

        # damped least-squares pseudoinverse
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + _LAMBDA_SQ * np.eye(3))  # (7, 3)

        dq = step_size * (J_pinv @ error)
        d.qpos[_JOINT_INDICES] += dq

        # clip to joint limits
        lo = model.jnt_range[_JOINT_INDICES, 0]
        hi = model.jnt_range[_JOINT_INDICES, 1]
        d.qpos[_JOINT_INDICES] = np.clip(d.qpos[_JOINT_INDICES], lo, hi)

    # check final error
    mujoco.mj_forward(model, d)
    final_error = np.linalg.norm(target_pos - d.site_xpos[sid])
    if final_error < tol * 5:   # lenient convergence check
        return d.qpos[_JOINT_INDICES].copy()
    return None


def solve_approach_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_pos: np.ndarray,
    site_name: str = "gripper",
    above_offset: float = 0.14,
    grasp_offset: float = 0.0,  # site AT cube center → fingers fully surround cube
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Solve IK for two poses: above the cube and at grasp height.
    Returns (above_joints, grasp_joints) — either may be None if IK fails.
    """
    above_target = cube_pos + np.array([0.0, 0.0, above_offset])
    # grasp_offset is relative to cube CENTER (qpos z = center, not bottom)
    # target 0.0 = fingers at cube center height → full grip around cube body
    grasp_target = cube_pos + np.array([0.0, 0.0, grasp_offset])

    above_joints = solve_ik(model, data, above_target, site_name)
    grasp_joints  = solve_ik(model, data, grasp_target, site_name)

    return above_joints, grasp_joints
