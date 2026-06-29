"""
SmolVLA actor wrapper for Spotter.

Drop-in replacement for ClassicalActor — same interface:
  reset(), act(obs), done(), retry(params), set_instruction(text), phase_name

Model:  lerobot/smolvla_libero
Input:  state (6,), images camera1/2/3 (3,256,256), language tokens (48,)
Output: action (7,) — 6 joint positions + 1 gripper (SO100 space)

Action mapping: SO100 joints → Franka via Jacobian pseudoinverse (approximate).
State mapping:  Franka 8D → first 6 joints only (truncation).
"""
import copy
import numpy as np

DEFAULT_MODEL_ID = "lerobot/smolvla_libero"
DEFAULT_INSTRUCTION = "pick up the green cube and place it on the red target"

_ALPHA = 0.02        # small step to prevent large joint jumps from SO100→Franka mismatch
_LAMBDA_SQ = 1e-4
_TOKEN_MAX_LEN = 48  # from TokenizerProcessorStep in make_smolvla_pre_post_processors


class SmolVLAActor:
    """
    Frozen SmolVLA actor. Only the language instruction changes between episodes.
    When Gemma fires a correction, retry() calls set_instruction() so the policy
    re-executes with the new language conditioning.
    This is the core Spotter claim: Gemma's words change what the VLA does.
    """

    def __init__(
        self,
        model,
        data,
        model_id: str = DEFAULT_MODEL_ID,
    ):
        self._mj_model = model
        self._mj_data = data
        self._policy = self._load_policy(model_id)
        self._instruction = DEFAULT_INSTRUCTION
        self._step = 0

        from transformers import AutoTokenizer
        # Tokenizer confirmed from SmolVLAConfig.vlm_model_name on spark
        self._tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )

    def reset(self) -> None:
        self._instruction = DEFAULT_INSTRUCTION
        self._step = 0
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def set_instruction(self, text: str) -> None:
        self._instruction = text

    def retry(self, params=None) -> None:
        if params is not None and hasattr(params, "corrected_instruction"):
            self.set_instruction(params.corrected_instruction)
        self._step = 0
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def act(self, obs: dict | None = None) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if obs is None:
            obs = {}

        device = next(self._policy.parameters()).device

        # State: smolvla_libero expects (6,) — truncate Franka 8D to first 6 joints
        franka_state = np.asarray(
            obs.get("state", np.zeros(8, dtype=np.float32)), dtype=np.float32
        )
        state6 = franka_state[:6]
        state_t = torch.tensor(state6).unsqueeze(0).to(device)  # (1, 6)

        # Image: resize to 256×256, provide as all three camera slots
        image = np.asarray(
            obs.get("image", np.zeros((480, 640, 3), dtype=np.uint8)), dtype=np.uint8
        )
        img_t = (
            torch.tensor(image, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )                                                          # (1, 3, H, W)
        img_t = F.interpolate(img_t, size=(256, 256), mode="bilinear", align_corners=False)
        img_t = img_t.to(device)

        # Language: NewLineTaskProcessorStep appends \n; max_length=48
        task_text = self._instruction + "\n"
        tokenized = self._tokenizer(
            [task_text],
            max_length=_TOKEN_MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        obs_dict = {
            "observation.state": state_t,
            "observation.images.camera1": img_t,
            "observation.images.camera2": img_t,   # duplicate — we have one camera
            "observation.images.camera3": img_t,
            "task": [self._instruction],
            "observation.language.tokens": tokenized["input_ids"].to(device),
            "observation.language.attention_mask": tokenized["attention_mask"].to(
                dtype=torch.bool, device=device
            ),
        }

        with torch.no_grad():
            action = self._policy.select_action(obs_dict)

        action_np = action.squeeze(0).cpu().numpy()  # (7,)

        # Map 6D SO100 joint actions → Franka 8D ctrl via Jacobian
        # action_np[:6] treated as Cartesian-ish delta; action_np[6] = gripper
        pose_delta = np.zeros(6, dtype=np.float64)
        pose_delta[:min(6, len(action_np))] = action_np[:6]

        ctrl = self._cartesian_delta_to_ctrl(pose_delta, franka_state)

        gripper = float(action_np[6]) if len(action_np) > 6 else franka_state[7]
        ctrl[7] = float(np.clip(gripper, 0.0, 0.04))

        self._step += 1
        return ctrl

    def done(self) -> bool:
        return False

    @property
    def phase_name(self) -> str:
        return "vla"

    def _cartesian_delta_to_ctrl(
        self,
        dx6: np.ndarray,
        current_state: np.ndarray,
    ) -> np.ndarray:
        import mujoco

        model = self._mj_model
        data = self._mj_data

        d = copy.copy(data)
        d.qpos = data.qpos.copy()
        d.qpos[:7] = current_state[:7]
        d.qvel = data.qvel.copy()
        d.ctrl = data.ctrl.copy()
        mujoco.mj_forward(model, d)

        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        nv = model.nv

        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(model, d, jacp, jacr, sid)

        J6 = np.vstack([jacp[:, :7], jacr[:, :7]])          # (6, 7)
        J_pinv = J6.T @ np.linalg.inv(J6 @ J6.T + _LAMBDA_SQ * np.eye(6))  # (7, 6)
        dq = J_pinv @ dx6                                    # (7,)

        ctrl = np.zeros(8, dtype=np.float64)
        ctrl[:7] = current_state[:7] + _ALPHA * dq

        lo = model.jnt_range[:7, 0]
        hi = model.jnt_range[:7, 1]
        ctrl[:7] = np.clip(ctrl[:7], lo, hi)
        return ctrl

    def _load_policy(self, model_id: str):
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as exc:
            raise ImportError("lerobot not installed") from exc

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = SmolVLAPolicy.from_pretrained(model_id)
        policy.eval()
        policy = policy.to(device)
        return policy
