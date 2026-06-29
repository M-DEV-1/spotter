import torch
import numpy as np
import time
import os
from importlib import reload

import mujoco

print("Loading mujoco scene...")
model = mujoco.MjModel.from_xml_path("/home/zugzwang/spotter/mujoco_menagerie/franka_emika_panda/mjx_single_cube.xml")
data = mujoco.MjData(model)

import actor.smolvla
print("Importing smolvla for the first time...")
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.disable = True

def run():
    print("Reloading actor.smolvla...")
    reload(actor.smolvla)
    
    actor_instance = actor.smolvla.SmolVLAActor(model, data, model_id="lerobot/smolvla_libero")
    
    obs = {
        "state": np.zeros(8, dtype=np.float32),
        "image": np.zeros((256, 256, 3), dtype=np.uint8)
    }
    
    print("Calling act()...")
    try:
        ctrl = actor_instance.act(obs)
        print("Success! Output:", ctrl)
        print("Shape:", ctrl.shape)
    except Exception as e:
        import traceback
        print("Error encountered.")
        traceback.print_exc()

while True:
    run()
    user_input = input("Press Enter to hot-reload and retry (or type 'q' to quit): ")
    if user_input.strip().lower() == 'q':
        break
