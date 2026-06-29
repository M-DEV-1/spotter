"""
Rung 2 verification: one Cerebras call with a real sim frame + fake failure signal.
Usage: MUJOCO_GL=egl python scripts/test_supervisor.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
from simulator.scene import load_model, make_data, reset_to_keyframe
from simulator.record import make_renderer, render_frame
from supervisor.signals import compute_signals
from supervisor.cerebras_client import call_supervisor

model = load_model()
data = make_data(model)
reset_to_keyframe(model, data, "home")

# simulate a few steps so the arm is at home pose
for _ in range(100):
    mujoco.mj_step(model, data)

# render a frame
with make_renderer(model) as renderer:
    frame = render_frame(model, data, renderer)

# fake a gripper_closed_empty failure: close gripper ctrl but cube is on floor
data.ctrl[7] = 0.0
signals = compute_signals(model, data, no_progress=False)
print(f"signals: {signals}")

print("calling Cerebras supervisor...")
result = call_supervisor(frame, signals)

print(f"\nfailure_type:          {result.failure_type}")
print(f"diagnosis:             {result.diagnosis}")
print(f"corrected_instruction: {result.corrected_instruction}")
