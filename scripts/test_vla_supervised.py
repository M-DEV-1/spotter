"""
Single-episode Pi0 + Gemma supervised run — raw terminal output only, no mp4.

Usage on spark:
  cd ~/spotter
  MUJOCO_GL=egl CEREBRAS_API_KEY=... python scripts/test_vla_supervised.py
  MUJOCO_GL=egl ... python scripts/test_vla_supervised.py --no-supervisor   # baseline
  MUJOCO_GL=egl ... python scripts/test_vla_supervised.py --nudge 0.08      # nudge cube
"""
import sys, argparse, time
import numpy as np
import mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.scene import load_model, make_data, reset_to_keyframe
from simulator.record import make_renderer, render_frame
from supervisor.signals import compute_signals, NoProgressTracker, cube_pos as get_cube_pos
from supervisor.multi_agent import run_multi_agent_correction
from supervisor.parse_correction import parse_correction
from simulator.ik import solve_approach_grasp
from tasks.pick_place import init_task, is_success
from actor.pi0 import Pi0Actor

import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
load_dotenv()

PI0_MODEL_PATH = "/home/zugzwang/models/pi0_libero"
INTERVAL_CHECK = 500
MAX_RETRIES = 3

parser = argparse.ArgumentParser()
parser.add_argument("--no-supervisor", action="store_true")
parser.add_argument("--nudge", type=float, default=0.08, help="nudge cube Y at step 200")
parser.add_argument("--model-path", default=PI0_MODEL_PATH)
parser.add_argument("--max-steps", type=int, default=4000)
parser.add_argument("--print-ctrl-every", type=int, default=100, help="print ctrl vector N steps")
args = parser.parse_args()

print("Loading scene...")
model = load_model()
data = make_data(model)

print(f"Loading Pi0 from {args.model_path} ...")
actor = Pi0Actor(model, data, model_id=args.model_path)
print("Ready.\n")

reset_to_keyframe(model, data, "home")
init_task(model, data)
actor.reset()

client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"]) if not args.no_supervisor else None
progress = NoProgressTracker()
retries = 0
log = []
last_frame = None
t0 = time.time()

label = "SUPERVISED" if not args.no_supervisor else "UNSUPERVISED"
nudge_label = f" nudge={args.nudge}m" if args.nudge else ""
print(f"=== {label}{nudge_label} | max_steps={args.max_steps} | INTERVAL_CHECK={INTERVAL_CHECK} ===\n")

with make_renderer(model) as renderer:
    for step in range(args.max_steps):

        # nudge cube at step 200
        if args.nudge and step == 200:
            data.qpos[10] += args.nudge
            mujoco.mj_forward(model, data)
            print(f"  [step {step}] NUDGE cube +{args.nudge}m Y  cube_pos={get_cube_pos(model, data)}")

        # build obs
        if last_frame is not None:
            obs = {"state": data.qpos[:8].astype(np.float32), "image": last_frame}
        else:
            obs = None

        ctrl = actor.act(obs)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        # render every 4 steps (needed for VLA obs + signals)
        if step % 4 == 0:
            last_frame = render_frame(model, data, renderer)

        # periodic ctrl printout
        if step % args.print_ctrl_every == 0:
            cpos = get_cube_pos(model, data)
            ee_z = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")][2]
            signals = compute_signals(model, data)
            print(
                f"  step {step:4d} | ctrl={np.round(ctrl[:7],3).tolist()} g={ctrl[7]:.3f} "
                f"| cube=({cpos[0]:.3f},{cpos[1]:.3f},{cpos[2]:.3f}) ee_z={ee_z:.3f} "
                f"| closed_empty={int(signals['gripper_closed_empty'])} "
                f"out_region={int(signals['cube_out_of_region'])}"
            )

        # supervisor interval trigger
        if (client is not None
                and step > 0
                and step % INTERVAL_CHECK == 0
                and retries < MAX_RETRIES):
            no_prog = progress.update(model, data)
            signals = compute_signals(model, data, no_progress=no_prog)
            vla_missed = signals.get("gripper_closed_empty")
            if signals.get("cube_out_of_region") or signals.get("no_progress") or vla_missed:
                print(f"\n  >> SUPERVISOR FIRING @step={step}  signals={signals}")
                frame_now = render_frame(model, data, renderer)
                correction, agent_trace = run_multi_agent_correction(frame_now, signals, step, client)
                total_s = agent_trace[-1].get("total_s", "?")
                print(f"  << failure_type: {correction.failure_type}  ({total_s}s)")
                print(f"  << diagnosis:    {correction.diagnosis}")
                print(f"  << instruction:  {correction.corrected_instruction}")
                print(f"  << direction:    {correction.cube_direction}  depth: {correction.approach_depth}\n")
                log.append({"step": step, "correction": correction.corrected_instruction, "trace": agent_trace})
                if correction.failure_type != "none":
                    retries += 1
                    actor.set_instruction(correction.corrected_instruction)
                    actor.retry()

        if is_success(model, data):
            print(f"\n  *** SUCCESS at step {step} ***")
            break

elapsed = time.time() - t0
cpos = get_cube_pos(model, data)
print(f"\n=== DONE  steps={step+1}  retries={retries}  elapsed={elapsed:.1f}s ===")
print(f"    final cube pos: ({cpos[0]:.3f}, {cpos[1]:.3f}, {cpos[2]:.3f})")
print(f"    success: {is_success(model, data)}")
