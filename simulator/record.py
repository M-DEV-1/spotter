from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np


def make_renderer(model, height: int = 480, width: int = 640) -> mujoco.Renderer:
    return mujoco.Renderer(model, height, width)


def render_frame(model, data, renderer) -> np.ndarray:
    renderer.update_scene(data)
    return renderer.render()


def save_mp4(frames: list, path: str, fps: int = 30) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.mimsave(str(out), frames, fps=fps)
    print(f"wrote {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from simulator.scene import load_model, make_data, reset_to_keyframe

    model = load_model()
    data = make_data(model)
    reset_to_keyframe(model, data, "home")

    frames = []
    with make_renderer(model) as renderer:
        for _ in range(5):
            mujoco.mj_step(model, data)
            frames.append(render_frame(model, data, renderer))

    save_mp4(frames, "outputs/smoke/test.mp4")
