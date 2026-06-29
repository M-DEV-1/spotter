import sys
import importlib
import numpy as np
import mujoco
from simulator import scene

print("Loading mujoco scene...")
mj_model = scene.load_model()
mj_data = scene.make_data(mj_model)

print("Importing pi0 for the first time...")
import actor.pi0 as pi0
actor_instance = pi0.Pi0Actor(mj_model, mj_data)
policy = actor_instance._policy

obs = {"state": np.zeros(8, dtype=np.float32), "image": np.zeros((480, 640, 3), dtype=np.uint8)}

while True:
    try:
        print("Calling act()...")
        ctrl = actor_instance.act(obs)
        print("Success! Output:", ctrl)
        print("Shape:", ctrl.shape)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error encountered.")
    
    cmd = input("Press Enter to hot-reload and retry (or type 'q' to quit): ")
    if cmd.lower() == 'q':
        break
    
    print("Reloading actor.pi0...")
    importlib.reload(pi0)
    
    original_load = pi0.Pi0Actor._load_policy
    pi0.Pi0Actor._load_policy = lambda self, model_id: policy
    try:
        actor_instance = pi0.Pi0Actor(mj_model, mj_data)
    finally:
        pi0.Pi0Actor._load_policy = original_load
