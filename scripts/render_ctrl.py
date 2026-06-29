"""
Render a single frame from a ctrl vector. Use this to visually check poses.
Usage: python scripts/render_ctrl.py 0 0.3 0 -1.57079 0 2.0 -0.7853 0.04
Output: outputs/pose.png  (pull with: make pull)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import numpy as np
import imageio.v2 as iio
from simulator.scene import load_model, make_data, reset_to_keyframe
from simulator.record import make_renderer, render_frame

ctrl = np.array([float(x) for x in sys.argv[1:]])
assert len(ctrl) == 8, f"need 8 ctrl values, got {len(ctrl)}"

model = load_model()
data = make_data(model)
reset_to_keyframe(model, data, "home")

# drive toward the target ctrl and let physics settle
for _ in range(500):
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)

out = Path("outputs/pose.png")
out.parent.mkdir(parents=True, exist_ok=True)
with make_renderer(model) as renderer:
    img = render_frame(model, data, renderer)

iio.imwrite(str(out), img)
print(f"ctrl: {ctrl}")
print(f"wrote {out} — run 'make pull' to view")
