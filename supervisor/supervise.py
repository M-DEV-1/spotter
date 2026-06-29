"""
Supervised episode runner.
When a failure signal fires, the multi-agent supervisor (Perception→Planner→Validator)
diagnoses the scene and issues a corrective instruction; the actor retries.
"""
import os
import mujoco
from pathlib import Path

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

from simulator.scene import load_model, make_data, reset_to_keyframe
from simulator.record import make_renderer, render_frame, save_mp4
from supervisor.signals import compute_signals, NoProgressTracker
from supervisor.multi_agent import run_multi_agent_correction
from supervisor.parse_correction import parse_correction
from tasks.pick_place import init_task

load_dotenv()

# 2s cycle: 3 agent calls × 0.5Hz = 90 RPM — pushing limits without exceeding 100
SUPERVISOR_INTERVAL = 1000
MAX_RETRIES = 3


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
    last_phase = None
    retries = 0

    with make_renderer(model) as renderer:
        for step in range(max_steps):
            if perturb_fn:
                perturb_fn(model, data, step)

            ctrl = actor.act()
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

            if record and step % 4 == 0:
                frames.append(render_frame(model, data, renderer))

            if verbose and actor.phase_name != last_phase:
                last_phase = actor.phase_name
                print(f"  step {step:4d}  phase: {last_phase}")

            if supervised and step > 0 and step % SUPERVISOR_INTERVAL == 0 and retries < MAX_RETRIES:
                no_prog = progress.update(model, data)
                signals = compute_signals(model, data, no_progress=no_prog)

                if any(signals.values()):
                    frame_now = render_frame(model, data, renderer)
                    correction, agent_trace = run_multi_agent_correction(
                        frame_now, signals, step, client=client
                    )

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
                    print(f"\n  [multi-agent @{step}] {correction.failure_type}  ({total_s}s total)")
                    print(f"  perception:  {correction.diagnosis}")
                    print(f"  instruction: {correction.corrected_instruction}")
                    print(f"  direction:   {correction.cube_direction}  depth: {correction.approach_depth}\n")

                    if correction.failure_type != "none":
                        retries += 1
                        params = parse_correction(correction)
                        print(f"  [retry {retries}] j1={params.j1_offset:+.3f}  depth={params.depth_offset:+.3f}")
                        actor.retry(params=params)

            if actor.done():
                break

        if record:
            frames.append(render_frame(model, data, renderer))

    if record and frames:
        save_mp4(frames, out_path)

    return {"steps": step + 1, "frames": len(frames), "log": log, "retries": retries}
