from pathlib import Path

import mujoco
import numpy as np

SCENE_XML = Path(__file__).parent.parent / "mujoco_menagerie/franka_emika_panda/mjx_single_cube.xml"


def load_model(scene_xml=None) -> mujoco.MjModel:
    path = str(scene_xml or SCENE_XML)
    model = mujoco.MjModel.from_xml_path(path)
    print(f"loaded {path}")
    print(f"  nq={model.nq}  nu={model.nu}  nkey={model.nkey}")
    return model


def make_data(model) -> mujoco.MjData:
    return mujoco.MjData(model)


def reset_to_keyframe(model, data, name: str) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        raise ValueError(f"keyframe not found: {name!r}")
    mujoco.mj_resetDataKeyframe(model, data, key_id)


def keyframe_ctrl(model, name: str) -> np.ndarray:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        raise ValueError(f"keyframe not found: {name!r}")
    return model.key_ctrl[key_id].copy()


def body_id(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def site_id(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


if __name__ == "__main__":
    model = load_model()

    print("\nbody names:")
    for i in range(model.nbody):
        print(f"  [{i}] {model.body(i).name}")

    print("\nkeyframe ctrl vectors:")
    for i in range(model.nkey):
        name = model.key(i).name
        ctrl = model.key_ctrl[i]
        print(f"  {name}: {ctrl}")
