"""
Pi0 VLA actor wrapper for Spotter.

Drop-in replacement for ClassicalActor — same interface:
  reset(), act(obs), done(), retry(params), set_instruction(text), phase_name

Model:  lerobot/pi0_libero_finetuned_v044
Download (run on spark before first use):
  huggingface-cli download lerobot/pi0_libero_finetuned_v044

Action space (LIBERO variant): 7 joint positions + 1 gripper = 8D.
If the deployed model outputs 6D Cartesian EE deltas instead of joint positions,
set USE_CARTESIAN_ACTIONS = True and the Jacobian path will be used automatically.

Verify on spark with:
  python -c "
  from lerobot.policies.pi0.modeling_pi0 import Pi0Policy
  p = Pi0Policy.from_pretrained('lerobot/pi0_libero_finetuned_v044')
  print('action_dim:', p.config.output_shapes)
  "

# API confirmed from lerobot source — verify on spark if import fails
"""
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import copy
import numpy as np

import torch
import torch._dynamo
torch._dynamo.config.disable = True

DEFAULT_MODEL_ID = "lerobot/pi0_libero_finetuned_v044"
DEFAULT_INSTRUCTION = "pick up the green cube and place it on the red target"

# Set True if the model outputs 6D Cartesian EE deltas + 1D gripper.
# Set False (default) if the model outputs 7 joint positions + 1 gripper.
# Verify with print(policy.config.output_shapes) on spark.
USE_CARTESIAN_ACTIONS = False

# Step size for Jacobian-based Cartesian-to-joint conversion.
# Only used when USE_CARTESIAN_ACTIONS = True.
_ALPHA = 0.1

# Damped least-squares regularisation — matches simulator/ik.py
_LAMBDA_SQ = 1e-4


class Pi0Actor:
    """
    Frozen pi0 VLA actor.  Only the language instruction changes between episodes.
    When Gemma fires a correction, retry() calls set_instruction() and the policy
    re-executes from step 0 with the new conditioning text.
    This is the core Spotter claim: Gemma's words change what the VLA does.
    """

    def __init__(
        self,
        model,                           # mujoco.MjModel — for Jacobian (Cartesian path only)
        data,                            # mujoco.MjData  — for Jacobian (Cartesian path only)
        model_id: str = DEFAULT_MODEL_ID,
    ):
        self._mj_model = model
        self._mj_data = data
        self._policy = self._load_policy(model_id)
        self._instruction = DEFAULT_INSTRUCTION
        self._step = 0


    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset policy state and restore default instruction."""
        self._instruction = DEFAULT_INSTRUCTION
        self._step = 0
        # Pi0 is diffusion-based; each select_action call is nominally
        # stateless, but reset() clears any internal step counters.
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def set_instruction(self, text: str) -> None:
        """Update the language conditioning instruction."""
        self._instruction = text

    def retry(self, params=None) -> None:
        """Called by supervise.py when Gemma fires a correction.
        Updates the instruction; resets the step counter so the policy
        re-samples from t=0 with the new text.
        """
        if params is not None and hasattr(params, "corrected_instruction"):
            self.set_instruction(params.corrected_instruction)
        self._step = 0
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    # ------------------------------------------------------------------
    # Actor interface
    # ------------------------------------------------------------------

    def act(self, obs: dict | None = None) -> np.ndarray:
        """
        Build observation dict, call policy, return 8-element ctrl array.

        obs must contain:
          "state"  — (8,) float32  [joint1..joint7, gripper current positions]
          "image"  — (H, W, 3) uint8  RGB camera frame (top-down view)

        Returns (8,) float64 ctrl vector [joint1..joint7, gripper_width].
        """
        import torch  # noqa: PLC0415

        if obs is None:
            obs = {}

        device = next(self._policy.parameters()).device

        state = np.asarray(obs.get("state", np.zeros(8, dtype=np.float32)), dtype=np.float32)
        image = np.asarray(obs.get("image", np.zeros((480, 640, 3), dtype=np.uint8)), dtype=np.uint8)

        state_t = torch.tensor(state).unsqueeze(0).to(device)          # (1, 8)

        # (H, W, 3) uint8 → (1, 3, H, W) float32 in [0, 1]
        img_t = (
            torch.tensor(image, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
            / 255.0
        )                                                               # (1, 3, H, W)

        # Observation dict key names from lerobot pi0_libero config.
        # Image key and state dim need verification on spark when pi0 is unblocked.
        obs_dict = {
            "observation.state": state_t,           # (1, 8) joint positions
            "observation.images.top": img_t,        # (1, 3, H, W) top-down camera
            "task": [self._instruction],             # list[str] — language conditioning
        }

        with torch.no_grad():
            # select_action(obs_dict) → (1, action_dim) tensor
            # API confirmed from lerobot source — verify on spark if this errors.
            action = self._policy.select_action(obs_dict)

        action_np = action.squeeze(0).cpu().numpy()   # (action_dim,)

        if USE_CARTESIAN_ACTIONS:
            # action_np[:6] = [Δx, Δy, Δz, ΔRx, ΔRy, ΔRz]
            # action_np[6]  = gripper delta/command
            ctrl = self._cartesian_delta_to_ctrl(action_np[:6], state)
            ctrl[7] = float(np.clip(action_np[6], 0.0, 0.04))
        else:
            # action_np[:7] = absolute joint positions
            # action_np[7]  = gripper width (0 = closed, 0.04 = open)
            ctrl = np.zeros(8, dtype=np.float64)
            ctrl[:7] = np.clip(action_np[:7].astype(np.float64), -3.15, 3.15)
            gripper_raw = action_np[7] if len(action_np) > 7 else 0.04
            ctrl[7] = float(np.clip(gripper_raw, 0.0, 0.04))

        self._step += 1
        return ctrl

    def done(self) -> bool:
        """VLA runs until externally stopped — always returns False."""
        return False

    @property
    def phase_name(self) -> str:
        return "vla"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_policy(self, model_id: str):
        """Load pi0 policy from lerobot hub.  Raises ImportError with a helpful
        message if lerobot is not installed."""
        try:
            # API confirmed from lerobot source — verify on spark if import fails
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "lerobot is not installed. To install:\n"
                "  pip install lerobot\n"
                "To download the model weights:\n"
                f"  huggingface-cli download {model_id}\n"
                "Or use Pi0StubActor (actor/pi0_stub.py) to test the pipeline "
                "without lerobot."
            ) from exc

        import torch  # noqa: PLC0415

        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = PI0Policy.from_pretrained(model_id)
        policy.eval()
        policy = policy.to(device)
        return policy

    def _cartesian_delta_to_ctrl(
        self,
        dx6: np.ndarray,
        current_state: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a 6D Cartesian EE delta [Δx, Δy, Δz, ΔRx, ΔRy, ΔRz] to a
        joint-position ctrl vector using the full (position + rotation) Jacobian.

        Uses damped least-squares pseudoinverse — same formulation as simulator/ik.py.

            dq     = J_pinv @ dx6
            ctrl[:7] = current_qpos[:7] + alpha * dq

        Only called when USE_CARTESIAN_ACTIONS = True.
        """
        import mujoco  # noqa: PLC0415

        model = self._mj_model
        data = self._mj_data

        # Work on a copy — never disturb the live simulation.
        d = copy.copy(data)
        d.qpos = data.qpos.copy()
        d.qvel = data.qvel.copy()
        d.ctrl = data.ctrl.copy()
        mujoco.mj_forward(model, d)

        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        nv = model.nv

        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(model, d, jacp, jacr, sid)

        # Full 6×7 Jacobian (translational + rotational rows, arm joints only)
        J6 = np.vstack([jacp[:, :7], jacr[:, :7]])          # (6, 7)

        # Damped least-squares pseudoinverse: J^T (J J^T + λ²I)^{-1}
        J_pinv = J6.T @ np.linalg.inv(J6 @ J6.T + _LAMBDA_SQ * np.eye(6))  # (7, 6)

        dq = J_pinv @ dx6                                    # (7,)

        ctrl = np.zeros(8, dtype=np.float64)
        ctrl[:7] = current_state[:7] + _ALPHA * dq

        # Clip to joint limits
        lo = model.jnt_range[:7, 0]
        hi = model.jnt_range[:7, 1]
        ctrl[:7] = np.clip(ctrl[:7], lo, hi)
        return ctrl
