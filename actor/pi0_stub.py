"""
Pi0 stub actor — same interface as Pi0Actor but requires no lerobot install.

Slowly interpolates from HOME toward a fixed approach target, ignoring the
language instruction.  Use this to validate supervise.py wiring (supervisor
calls, retry path, signal detection) on machines without the VLA downloaded.

Usage:
  from actor.pi0_stub import Pi0StubActor
  actor = Pi0StubActor()
  actor.reset()
  ctrl = actor.act(obs)   # same signature as Pi0Actor and ClassicalActor
"""
import numpy as np

DEFAULT_INSTRUCTION = "pick up the green cube and place it on the red target"

# Reuse the same hand-tuned approach pose as ClassicalActor — reasonable stub target.
_HOME        = np.array([ 0.000,  0.300,  0.000, -1.571,  0.000,  2.000, -0.785,  0.04])
_STUB_TARGET = np.array([ 0.000,  0.626,  0.000, -1.547,  0.000,  2.225, -0.7143, 0.04])

# Fraction of the interpolation covered per act() call.
# At 200Hz physics with 1:1 act, this reaches the target in ~50 steps.
_STEP_SIZE = 0.02


class Pi0StubActor:
    """
    Stub drop-in for Pi0Actor.

    - No lerobot or torch required.
    - Same interface: reset(), act(obs), done(), retry(params),
      set_instruction(text), phase_name.
    - retry() prints the new instruction so you can verify Gemma's text
      reaches the actor, even though the stub ignores it for motion.
    """

    def __init__(
        self,
        model=None,       # unused — accepted to match Pi0Actor signature
        data=None,        # unused — accepted to match Pi0Actor signature
        model_id: str = "stub",
    ):
        self._instruction = DEFAULT_INSTRUCTION
        self._step = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._instruction = DEFAULT_INSTRUCTION
        self._step = 0

    def set_instruction(self, text: str) -> None:
        self._instruction = text

    def retry(self, params=None) -> None:
        """Simulate instruction update — resets step counter and logs the new text."""
        if params is not None and hasattr(params, "corrected_instruction"):
            self.set_instruction(params.corrected_instruction)
        print(f"  [Pi0StubActor.retry] instruction: {self._instruction!r}")
        self._step = 0

    # ------------------------------------------------------------------
    # Actor interface
    # ------------------------------------------------------------------

    def act(self, obs: dict | None = None) -> np.ndarray:
        """
        Returns an 8-element ctrl vector that slowly moves toward _STUB_TARGET.
        obs is accepted but ignored.
        """
        t = min(self._step * _STEP_SIZE, 1.0)
        ctrl = _HOME + (_STUB_TARGET - _HOME) * t
        self._step += 1
        return ctrl.copy()

    def done(self) -> bool:
        """VLA runs until externally stopped — always returns False."""
        return False

    @property
    def phase_name(self) -> str:
        return "vla"
