"""
Maps a CorrectionInstruction (from Gemma) to RetryParams for the classical actor.

Causal chain: Gemma's cube_direction + approach_depth fields → RetryParams → actor.retry().
No sim state is read here. The parser is keyword-based and explicitly labeled as such.

Direction mapping calibrated to camera azimuth=145°, elevation=-22°, lookat=(0.5,0.1,0.15)
on spark-3100. From that NW viewpoint, world +Y is approximately camera-right, world +X is
approximately camera-forward-right. 'right' in the camera frame requires increasing joint1
(CCW rotation, toward +Y).
"""
from dataclasses import dataclass, field

from supervisor.cerebras_client import CorrectionInstruction


@dataclass
class RetryParams:
    j1_offset: float = 0.0    # radians added to joint1 in grasp phases
    depth_offset: float = 0.0  # radians added to joint2 in lower/grasp phases (positive = lower)


# Calibrated: camera-right (+Y) → increase j1; camera-left (-Y) → decrease j1.
# camera-forward (+X) → small j2 increase to extend reach.
_DIRECTION_J1 = {
    "right":   +0.06,
    "left":    -0.06,
    "forward": +0.00,
    "back":    +0.00,
    "center":  +0.00,
    "unknown": +0.00,
}
_DIRECTION_J2 = {
    "forward": +0.03,
    "back":    -0.03,
}


def parse_correction(correction: CorrectionInstruction) -> RetryParams:
    direction = correction.cube_direction or "unknown"
    j1 = _DIRECTION_J1.get(direction, 0.0)
    j2 = _DIRECTION_J2.get(direction, 0.0)

    # depth from structured field (primary) or instruction text (fallback)
    depth = 0.0
    if correction.approach_depth == "deeper":
        depth = 0.04
    else:
        text = (correction.corrected_instruction or "").lower()
        if any(w in text for w in ("descend", "lower", "deeper", "further down")):
            depth = 0.04

    return RetryParams(j1_offset=j1, depth_offset=j2 + depth)
