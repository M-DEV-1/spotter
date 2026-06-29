"""
Episode runner: drives actor through a task, optionally recording frames.
Pass watch=True to stream live to http://spark-3100:8080 via mjviser.
"""
import mujoco
from pathlib import Path

from simulator.scene import load_model, make_data, reset_to_keyframe
from simulator.record import make_renderer, render_frame, save_mp4


def run_episode(
    actor,
    task_init_fn=None,
    max_steps: int = 2000,
    record: bool = True,
    record_every: int = 4,
    out_path: str = "outputs/episode.mp4",
    fps: int = 30,
    verbose: bool = True,
    watch: bool = False,
    watch_port: int = 8080,
) -> dict:
    model = load_model()
    data = make_data(model)
    reset_to_keyframe(model, data, "home")

    if task_init_fn:
        task_init_fn(model, data)

    actor.reset()

    # optional live viewer
    viser_scene = None
    if watch:
        try:
            import time, viser, mjviser
            _server = viser.ViserServer(port=watch_port)
            viser_scene = mjviser.ViserMujocoScene(_server, model, num_envs=1)
            print(f"\n  open browser → http://spark-3100:{watch_port}")
            for i in range(8, 0, -1):
                print(f"  starting in {i}s ...  ", end="\r", flush=True)
                time.sleep(1)
            print("  starting now!          \n")
        except Exception as e:
            print(f"  watch mode unavailable: {e}")
            viser_scene = None

    frames = []
    last_phase = None

    with make_renderer(model) as renderer:
        for step in range(max_steps):
            ctrl = actor.act()
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

            if record and step % record_every == 0:
                frames.append(render_frame(model, data, renderer))

            if viser_scene is not None and step % 4 == 0:
                viser_scene.update_from_mjdata(data)

            if verbose and actor.phase_name != last_phase:
                last_phase = actor.phase_name
                print(f"  step {step:4d}  phase: {last_phase}")

            if actor.done():
                break

        if record:
            frames.append(render_frame(model, data, renderer))

    if record and frames:
        save_mp4(frames, out_path, fps=fps)

    return {
        "steps": step + 1,
        "frames": len(frames),
        "out_path": out_path if record else None,
    }
