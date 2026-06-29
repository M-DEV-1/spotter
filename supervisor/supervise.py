"""
Supervised episode runner.
Runs the classical actor with a ~1Hz Gemma supervisor call.
When a failure signal fires, supervisor diagnoses + actor retries.
"""
import os
import mujoco
from pathlib import Path

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

from simulator.scene import load_model, make_data, reset_to_keyframe
from simulator.record import make_renderer, render_frame, save_mp4
from supervisor.signals import compute_signals, NoProgressTracker
from supervisor.cerebras_client import call_supervisor
from tasks.pick_place import init_task

load_dotenv()

# ~1Hz supervision: MuJoCo default timestep is 0.002s → 500 steps = 1s
SUPERVISOR_INTERVAL = 500
MAX_RETRIES = 2


def run_supervised_episode(
    actor,
    perturb_fn=None,
    supervised: bool = True,
    max_steps: int = 4000,
    record: bool = True,
    out_path: str = "outputs/supervised.mp4",
    watch: bool = False,
    watch_port: int = 8080,
    verbose: bool = True,
) -> dict:
    model = load_model()
    data = make_data(model)
    reset_to_keyframe(model, data, "home")
    init_task(model, data)
    actor.reset()

    client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"]) if supervised else None
    progress = NoProgressTracker()

    viser_scene = None
    if watch:
        try:
            import time, viser, mjviser
            _srv = viser.ViserServer(port=watch_port)
            viser_scene = mjviser.ViserMujocoScene(_srv, model, num_envs=1)
            print(f"\n  open browser → http://spark-3100:{watch_port}")
            for i in range(8, 0, -1):
                print(f"  starting in {i}s ...  ", end="\r", flush=True)
                time.sleep(1)
            print("  starting now!          \n")
        except Exception as e:
            print(f"  watch unavailable: {e}")

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

            if viser_scene is not None and step % 4 == 0:
                viser_scene.update_from_mjdata(data)

            if verbose and actor.phase_name != last_phase:
                last_phase = actor.phase_name
                print(f"  step {step:4d}  phase: {last_phase}")

            # supervisor check
            if supervised and step > 0 and step % SUPERVISOR_INTERVAL == 0 and retries < MAX_RETRIES:
                no_prog = progress.update(model, data)
                signals = compute_signals(model, data, no_progress=no_prog)

                if any(signals.values()):
                    frame_now = render_frame(model, data, renderer)
                    correction = call_supervisor(frame_now, signals, client=client)
                    entry = {"step": step, "signals": {k: bool(v) for k, v in signals.items()}, "correction": vars(correction)}
                    log.append(entry)
                    print(f"\n  [supervisor @{step}] {correction.failure_type}")
                    print(f"  diagnosis:   {correction.diagnosis}")
                    print(f"  correction:  {correction.corrected_instruction}\n")

                    if correction.failure_type != "none":
                        retries += 1
                        actor.retry()

            if actor.done():
                break

        if record:
            frames.append(render_frame(model, data, renderer))

    if record and frames:
        save_mp4(frames, out_path)

    return {"steps": step + 1, "frames": len(frames), "log": log, "retries": retries}
