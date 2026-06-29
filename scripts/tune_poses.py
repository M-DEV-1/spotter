"""
Interactive pose tuner via mjviser web viewer.
Run on spark: MUJOCO_GL=egl python scripts/tune_poses.py
Open browser at: http://spark-3100:8080
Adjust sliders to position the arm. Terminal prints ctrl values every 2s.

When you have a good pose, note the ctrl array and share it.
"""
import sys
import time
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import numpy as np
import viser
import mjviser
from simulator.scene import load_model, make_data, reset_to_keyframe
from tasks.pick_place import init_task

PORT = 8080

model = load_model()
data = make_data(model)
reset_to_keyframe(model, data, "home")
init_task(model, data)   # moves red block to place zone (0.45, 0.45, 0.03)

server = viser.ViserServer(port=PORT)

sliders = []
for i in range(model.nu):
    lo, hi = model.actuator_ctrlrange[i]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
    sl = server.gui.add_slider(
        label=name,
        min=float(lo),
        max=float(hi),
        step=0.001,
        initial_value=float(data.ctrl[i]),
    )
    sliders.append(sl)


def step_fn(m, d):
    for i, sl in enumerate(sliders):
        d.ctrl[i] = sl.value
    mujoco.mj_step(m, d)


def _print_loop():
    while True:
        vals = np.array([sl.value for sl in sliders])
        np.set_printoptions(precision=5, suppress=True)
        print(f"\rctrl: {vals}", end="", flush=True)
        time.sleep(2)


threading.Thread(target=_print_loop, daemon=True).start()

print(f"Open browser at:  http://spark-3100:{PORT}")
print("Adjust sliders to pose the arm.")
print("Copy ctrl values from terminal. Ctrl+C to stop.\n")

viewer = mjviser.Viewer(model, data, step_fn=step_fn, server=server)
viewer.run()
