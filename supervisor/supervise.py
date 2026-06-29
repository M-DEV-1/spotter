"""
Supervised episode runner.
Supervisor fires on phase transition grasp→lift (gripper just closed, arm still near cube).
This gives Gemma the clearest possible view of the spatial relationship.
Interval-based check still runs for cube_out_of_region and no_progress.
"""
import os
import mujoco
from pathlib import Path

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

from simulator.scene import load_model, make_data, reset_to_keyframe
from simulator.record import make_renderer, render_frame, save_mp4
from supervisor.signals import compute_signals, NoProgressTracker, cube_pos as get_cube_pos
from supervisor.multi_agent import run_multi_agent_correction
from supervisor.parse_correction import parse_correction
from simulator.ik import solve_approach_grasp, within_workspace
from tasks.pick_place import init_task

load_dotenv()

# Interval check for no_progress / cube_out_of_region (not for gripper_missed)
INTERVAL_CHECK = 2000
MAX_RETRIES = 3


def _fire_supervisor(frame, signals, step, client, log, retries, actor, model, data):
    """Call multi-agent pipeline, log the trace, issue retry. Returns new retries count."""
    correction, agent_trace = run_multi_agent_correction(frame, signals, step, client)

    entry = {
        "step": step,
        "signals": {k: bool(v) for k, v in signals.items()},
        "correction": {
            "failure_type": correction.failure_type,
            "diagnosis": correction.diagnosis,
            "corrected_instruction": correction.corrected_instruction,
            "cube_direction": correction.cube_direction,
            "approach_depth": correction.approach_depth,
        },
        "agent_trace": agent_trace,
    }
    log.append(entry)

    total_s = agent_trace[-1].get("total_s", "?")
    print(f"\n  [multi-agent @{step}] {correction.failure_type}  ({total_s}s)")
    print(f"  perception:  {correction.diagnosis}")
    print(f"  instruction: {correction.corrected_instruction}")
    print(f"  direction:   {correction.cube_direction}  depth: {correction.approach_depth}")

    if correction.failure_type != "none":
        retries += 1

        # VLA actors (Pi0Actor): update language conditioning from Gemma's output.
        # This is the core causal claim — Gemma's words change what the VLA does.
        if hasattr(actor, "set_instruction"):
            actor.set_instruction(correction.corrected_instruction)

        cpos = get_cube_pos(model, data)
        if hasattr(actor, "retry_to_position"):
            # Classical actor: IK path → exact joint targets
            above_j, lower_j = solve_approach_grasp(model, data, cpos)
            if above_j is not None and lower_j is not None:
                print(f"  [retry {retries}] IK path  cube=({cpos[0]:.3f},{cpos[1]:.3f})\n")
                actor.retry_to_position(above_j, lower_j)
                entry["recovery_path"] = "ik"
            else:
                # IK failed (out of workspace) — fall back to direction hint
                params = parse_correction(correction)
                print(f"  [retry {retries}] direction path  j1={params.j1_offset:+.3f}  depth={params.depth_offset:+.3f}\n")
                actor.retry(params=params)
                entry["recovery_path"] = "direction"
        else:
            # VLA actor: instruction already updated above; reset step counter
            print(f"  [retry {retries}] VLA path  cube=({cpos[0]:.3f},{cpos[1]:.3f})\n")
            actor.retry()
            entry["recovery_path"] = "vla"

    return retries


def run_supervised_episode(
    actor,
    perturb_fn=None,
    supervised: bool = True,
    max_steps: int = 6000,
    record: bool = True,
    out_path: str = "outputs/supervised.mp4",
    verbose: bool = True,
) -> dict:
    model = load_model()
    data = make_data(model)
    reset_to_keyframe(model, data, "home")
    init_task(model, data)
    actor.reset()

    client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"]) if supervised else None
    progress = NoProgressTracker()

    frames = []
    log = []
    retries = 0
    last_phase = None
    prev_phase = None
    post_grasp_armed = False   # True after entering "grasp", cleared after supervisor fires
    last_frame = None          # cached for VLA obs (rendered every 4 steps)
    is_vla = hasattr(actor, "set_instruction")

    with make_renderer(model) as renderer:
        for step in range(max_steps):
            if perturb_fn:
                perturb_fn(model, data, step)

            # build obs for VLA actors; classical actor ignores obs
            if is_vla and last_frame is not None:
                import numpy as np
                obs = {
                    "state": data.qpos[:8].astype(np.float32),
                    "image": last_frame,
                }
            else:
                obs = None

            ctrl = actor.act(obs)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

            if record and step % 4 == 0:
                last_frame = render_frame(model, data, renderer)
                frames.append(last_frame)

            # track phase transitions
            if actor.phase_name != last_phase:
                prev_phase = last_phase
                last_phase = actor.phase_name
                if verbose:
                    print(f"  step {step:4d}  phase: {last_phase}")
                # arm entering grasp — arm the post-grasp check
                if last_phase == "grasp":
                    post_grasp_armed = True

            # ── post-grasp trigger ─────────────────────────────────────────
            # fires the instant grasp→lift transition happens, while arm is
            # still at grasp height and cube is right next to the gripper
            if (supervised
                    and post_grasp_armed
                    and last_phase == "lift"
                    and prev_phase == "grasp"
                    and retries < MAX_RETRIES):
                post_grasp_armed = False
                signals = compute_signals(model, data, no_progress=False)
                if signals.get("gripper_closed_empty"):
                    frame_now = render_frame(model, data, renderer)
                    retries = _fire_supervisor(frame_now, signals, step, client, log, retries, actor, model, data)
                    prev_phase = None   # reset so next grasp→lift re-arms correctly

            # ── interval trigger: cube_out_of_region / no_progress only ───
            elif (supervised
                    and step > 0
                    and step % INTERVAL_CHECK == 0
                    and retries < MAX_RETRIES):
                no_prog = progress.update(model, data)
                signals = compute_signals(model, data, no_progress=no_prog)
                if signals.get("cube_out_of_region") or signals.get("no_progress"):
                    frame_now = render_frame(model, data, renderer)
                    retries = _fire_supervisor(frame_now, signals, step, client, log, retries, actor, model, data)

            if actor.done():
                break

        if record:
            last_frame = render_frame(model, data, renderer)
            frames.append(last_frame)

    if record and frames:
        save_mp4(frames, out_path)

    return {"steps": step + 1, "frames": len(frames), "log": log, "retries": retries}
