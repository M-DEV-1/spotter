import sys
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.scene import keyframe_ctrl, load_model, make_data, reset_to_keyframe
from simulator.record import make_renderer, render_frame

OUT = Path(__file__).parent.parent / "outputs/smoke"
OUT.mkdir(parents=True, exist_ok=True)

model = load_model()
data = make_data(model)
reset_to_keyframe(model, data, "home")

ctrl_home = keyframe_ctrl(model, "home")
ctrl_pickup = keyframe_ctrl(model, "pickup")

N = 300

with make_renderer(model) as renderer:
    for i in range(N):
        t = i / (N - 1)
        data.ctrl[:] = (1 - t) * ctrl_home + t * ctrl_pickup
        mujoco.mj_step(model, data)

        if i % 30 == 0:
            frame = render_frame(model, data, renderer)
            iio.imwrite(str(OUT / f"frame_{i:04d}.png"), frame)
            print(f"step {i:3d}  t={t:.2f}")

print("qpos[:7]:", data.qpos[:7])
print("smoke_move done — check outputs/smoke/")
