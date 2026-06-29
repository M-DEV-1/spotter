"""
Multi-agent supervisor: three Gemma 4 31B calls with genuinely different roles.

  Perception  → scene description from camera + signals (no goal knowledge)
  Planner     → rewrites the instruction from description + goal (no camera)
  Validator   → approves or rejects; bounces back to Planner ONCE if rejected

Each agent has different inputs. The reject-bounce makes it collaboration, not a pipeline.

Causal chain: Gemma-trio decides IF and HOW to correct → parse_correction → actor.retry()
Rate: 3 calls per cycle. At MULTI_AGENT_INTERVAL=1000 steps (2s/cycle) = 90 RPM.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from cerebras.cloud.sdk import Cerebras

from supervisor.cerebras_client import CorrectionInstruction, _encode_frame, MODEL

GOAL = "Pick up the green cube and place it on the red target marker."

_PERCEPTION_PROMPT = (Path(__file__).parent.parent / "prompts" / "perception.txt").read_text()
_PLANNER_PROMPT    = (Path(__file__).parent.parent / "prompts" / "planner.txt").read_text()
_VALIDATOR_PROMPT  = (Path(__file__).parent.parent / "prompts" / "validator.txt").read_text()

# ── tool schemas ────────────────────────────────────────────────────────────

_PERCEPTION_TOOL = {
    "type": "function",
    "function": {
        "name": "report_observation",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "scene_description": {
                    "type": "string",
                    "description": "Full description of what is visible in the frame.",
                },
                "cube_visible": {"type": "boolean"},
                "cube_relative_position": {
                    "type": "string",
                    "enum": ["left_of_gripper", "right_of_gripper", "forward_of_gripper",
                             "back_of_gripper", "aligned_with_gripper", "unknown"],
                },
                "gripper_state": {
                    "type": "string",
                    "enum": ["open", "closed", "partially_open", "unknown"],
                },
                "notable_change": {
                    "type": "string",
                    "description": "One sentence: what looks wrong or different from a normal scene.",
                },
            },
            "required": ["scene_description", "cube_visible", "cube_relative_position",
                         "gripper_state", "notable_change"],
            "additionalProperties": False,
        },
    },
}

_PLANNER_TOOL = {
    "type": "function",
    "function": {
        "name": "report_plan",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "rewritten_instruction": {"type": "string"},
                "cube_direction": {
                    "type": "string",
                    "enum": ["left", "right", "forward", "back", "center", "unknown"],
                },
                "approach_depth": {
                    "type": "string",
                    "enum": ["normal", "deeper"],
                },
                "reasoning": {
                    "type": "string",
                    "description": "One sentence: why this correction addresses the described failure.",
                },
            },
            "required": ["rewritten_instruction", "cube_direction", "approach_depth", "reasoning"],
            "additionalProperties": False,
        },
    },
}

_VALIDATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "report_validation",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "approval_status": {
                    "type": "string",
                    "enum": ["approved", "rejected"],
                },
                "rejection_reason": {
                    "type": "string",
                    "description": "Empty string if approved. Specific reason + suggestion if rejected.",
                },
                "validated_instruction": {
                    "type": "string",
                    "description": "The final instruction to send to the actor (same as planner's if approved).",
                },
                "cube_direction": {
                    "type": "string",
                    "enum": ["left", "right", "forward", "back", "center", "unknown"],
                },
                "approach_depth": {
                    "type": "string",
                    "enum": ["normal", "deeper"],
                },
            },
            "required": ["approval_status", "rejection_reason", "validated_instruction",
                         "cube_direction", "approach_depth"],
            "additionalProperties": False,
        },
    },
}


# ── dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class PerceptionResult:
    scene_description: str
    cube_visible: bool
    cube_relative_position: str
    gripper_state: str
    notable_change: str

@dataclass
class PlannerResult:
    rewritten_instruction: str
    cube_direction: str
    approach_depth: str
    reasoning: str

@dataclass
class ValidatorResult:
    approval_status: str
    rejection_reason: str
    validated_instruction: str
    cube_direction: str
    approach_depth: str


# ── individual agent calls ───────────────────────────────────────────────────

def _call_perception(frame: np.ndarray, signals: dict, client: Cerebras) -> PerceptionResult:
    b64 = _encode_frame(frame)
    signal_lines = "\n".join(f"- {k}: {v}" for k, v in signals.items())
    user_text = f"Physics signals:\n{signal_lines}\n\nDescribe what you see. Call report_observation."

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _PERCEPTION_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": user_text},
            ]},
        ],
        tools=[_PERCEPTION_TOOL],
        tool_choice={"type": "function", "function": {"name": "report_observation"}},
        max_tokens=300,
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        args = json.loads(msg.tool_calls[0].function.arguments)
        return PerceptionResult(**args)
    return PerceptionResult(
        scene_description="observation unavailable",
        cube_visible=False,
        cube_relative_position="unknown",
        gripper_state="unknown",
        notable_change="no tool call returned",
    )


def _call_planner(
    perception: PerceptionResult,
    client: Cerebras,
    rejection_feedback: Optional[str] = None,
) -> PlannerResult:
    context = (
        f"Scene description from observer:\n{perception.scene_description}\n\n"
        f"Notable change: {perception.notable_change}\n"
        f"Cube position: {perception.cube_relative_position}\n"
        f"Gripper state: {perception.gripper_state}\n\n"
        f"Goal: {GOAL}"
    )
    if rejection_feedback:
        context += f"\n\nPrevious plan was REJECTED by validator: {rejection_feedback}\nWrite a corrected plan."

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _PLANNER_PROMPT},
            {"role": "user", "content": context + "\n\nCall report_plan."},
        ],
        tools=[_PLANNER_TOOL],
        tool_choice={"type": "function", "function": {"name": "report_plan"}},
        max_tokens=256,
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        args = json.loads(msg.tool_calls[0].function.arguments)
        return PlannerResult(**args)
    return PlannerResult(
        rewritten_instruction="Reposition the gripper above the green cube and attempt to grasp again.",
        cube_direction="unknown",
        approach_depth="normal",
        reasoning="fallback plan",
    )


def _call_validator(plan: PlannerResult, client: Cerebras) -> ValidatorResult:
    context = (
        f"Proposed instruction: {plan.rewritten_instruction}\n"
        f"Cube direction estimate: {plan.cube_direction}\n"
        f"Approach depth: {plan.approach_depth}\n\n"
        f"Validate this instruction against workspace constraints. Call report_validation."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _VALIDATOR_PROMPT},
            {"role": "user", "content": context},
        ],
        tools=[_VALIDATOR_TOOL],
        tool_choice={"type": "function", "function": {"name": "report_validation"}},
        max_tokens=256,
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        args = json.loads(msg.tool_calls[0].function.arguments)
        return ValidatorResult(**args)
    return ValidatorResult(
        approval_status="approved",
        rejection_reason="",
        validated_instruction=plan.rewritten_instruction,
        cube_direction=plan.cube_direction,
        approach_depth=plan.approach_depth,
    )


# ── main pipeline ────────────────────────────────────────────────────────────

def _infer_failure_type(signals: dict) -> str:
    if signals.get("gripper_closed_empty"):
        return "gripper_missed"
    if signals.get("cube_out_of_region"):
        return "cube_displaced"
    if signals.get("no_progress"):
        return "no_progress"
    return "none"


def run_multi_agent_correction(
    frame: np.ndarray,
    signals: dict,
    step: int,
    client: Cerebras,
) -> tuple[CorrectionInstruction, list[dict]]:
    """
    Three-agent correction pipeline.
    Returns (CorrectionInstruction, agent_trace).
    agent_trace is a list of inter-agent messages for the dashboard.
    """
    trace = []
    t0 = time.time()

    # 1. Perception — sees the frame, no goal
    perception = _call_perception(frame, signals, client)
    trace.append({
        "from": "perception", "to": "planner", "step": step,
        "t": round(time.time() - t0, 2),
        "content": perception.scene_description,
        "cube_relative_position": perception.cube_relative_position,
        "notable_change": perception.notable_change,
    })

    # 2. Planner — text only, no camera
    plan = _call_planner(perception, client)
    trace.append({
        "from": "planner", "to": "validator", "step": step,
        "t": round(time.time() - t0, 2),
        "content": plan.rewritten_instruction,
        "cube_direction": plan.cube_direction,
        "reasoning": plan.reasoning,
    })

    # 3. Validator — may reject and bounce back to Planner once
    validator = _call_validator(plan, client)

    if validator.approval_status == "rejected":
        trace.append({
            "from": "validator", "to": "planner", "step": step,
            "t": round(time.time() - t0, 2),
            "content": f"REJECTED: {validator.rejection_reason}",
            "status": "rejected",
        })
        # bounce: planner gets one more attempt with rejection feedback
        plan = _call_planner(perception, client, rejection_feedback=validator.rejection_reason)
        trace.append({
            "from": "planner", "to": "validator", "step": step,
            "t": round(time.time() - t0, 2),
            "content": plan.rewritten_instruction,
            "cube_direction": plan.cube_direction,
            "reasoning": plan.reasoning + " [revised after rejection]",
        })
        validator = _call_validator(plan, client)

    trace.append({
        "from": "validator", "to": "actor", "step": step,
        "t": round(time.time() - t0, 2),
        "content": validator.validated_instruction,
        "status": validator.approval_status,
        "cube_direction": validator.cube_direction,
        "approach_depth": validator.approach_depth,
        "total_s": round(time.time() - t0, 2),
    })

    correction = CorrectionInstruction(
        failure_type=_infer_failure_type(signals),
        diagnosis=perception.notable_change,
        corrected_instruction=validator.validated_instruction,
        cube_direction=validator.cube_direction,
        approach_depth=validator.approach_depth,
    )

    return correction, trace
