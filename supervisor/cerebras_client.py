"""
Cerebras supervisor: sends a camera frame + failure signals to Gemma 4 31B,
returns a structured CorrectionInstruction via tool calling.
"""
import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemma-4-31b"
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "supervisor.txt"

_TOOL = {
    "type": "function",
    "function": {
        "name": "report_correction",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "failure_type": {
                    "type": "string",
                    "enum": ["gripper_missed", "cube_displaced", "no_progress", "none"],
                },
                "diagnosis": {"type": "string"},
                "corrected_instruction": {"type": "string"},
            },
            "required": ["failure_type", "diagnosis", "corrected_instruction"],
            "additionalProperties": False,
        },
    },
}

_TOOL_CHOICE = {"type": "function", "function": {"name": "report_correction"}}


@dataclass
class CorrectionInstruction:
    failure_type: str
    diagnosis: str
    corrected_instruction: str


def _encode_frame(frame: np.ndarray) -> str:
    buf = io.BytesIO()
    iio.imwrite(buf, frame, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def call_supervisor(
    frame: np.ndarray,
    signals: dict,
    client: Cerebras | None = None,
) -> CorrectionInstruction:
    if client is None:
        client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

    b64 = _encode_frame(frame)
    signal_lines = "\n".join(f"- {k}: {v}" for k, v in signals.items())
    user_text = f"Failure signals:\n{signal_lines}\n\nAnalyse the frame and signals, then call report_correction."

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _PROMPT_PATH.read_text()},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        tools=[_TOOL],
        tool_choice=_TOOL_CHOICE,
        max_tokens=256,
    )

    msg = response.choices[0].message
    if msg.tool_calls:
        args = json.loads(msg.tool_calls[0].function.arguments)
        return CorrectionInstruction(**args)

    # fallback — tool call not returned
    return CorrectionInstruction(
        failure_type="none",
        diagnosis="supervisor returned no tool call",
        corrected_instruction="pick up the green cube and place it on the red target",
    )
