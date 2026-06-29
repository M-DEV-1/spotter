"""
Rung 3: side-by-side unsupervised vs supervised with a scripted cube nudge.

Scripted failure: at step 200 (arm mid-approach), cube nudged 4cm in Y.
- Unsupervised: arm misses cube, finishes empty → FAIL
- Supervised:   Gemma detects gripper_missed, actor retries → recovers

Usage: MUJOCO_GL=egl python scripts/run_rung3.py [--watch]
Output:
  outputs/episodes/NNN_unsupervised.mp4
  outputs/episodes/NNN_supervised.mp4
"""
import sys
import argparse
import mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from actor.classical import ClassicalActor
from supervisor.supervise import run_supervised_episode

NUDGE_STEP = 200   # step at which the cube is displaced
NUDGE_Y    = 0.04  # 4cm in Y — small enough retry may succeed


def nudge(model, data, step):
    if step == NUDGE_STEP:
        # cube freejoint Y is at qpos[10]
        data.qpos[10] += NUDGE_Y
        mujoco.mj_forward(model, data)
        print(f"  [perturbation] cube nudged +{NUDGE_Y}m in Y at step {step}")


def next_prefix(base="outputs/episodes"):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("???_*.mp4"))
    n = int(existing[-1].stem[:3]) + 1 if existing else 1
    # also check plain NNN.mp4
    plain = sorted(base.glob("???.mp4"))
    if plain:
        n = max(n, int(plain[-1].stem) + 1)
    return base / f"{n:03d}"


parser = argparse.ArgumentParser()
parser.add_argument("--watch", action="store_true")
args = parser.parse_args()

prefix = next_prefix()

print("=== RUN 1: unsupervised + perturbation ===")
r1 = run_supervised_episode(
    ClassicalActor(),
    perturb_fn=nudge,
    supervised=False,
    out_path=str(prefix) + "_unsupervised.mp4",
    watch=args.watch,
    verbose=True,
)
print(f"unsupervised done — {r1['steps']} steps\n")

print("=== RUN 2: supervised + perturbation ===")
r2 = run_supervised_episode(
    ClassicalActor(),
    perturb_fn=nudge,
    supervised=True,
    out_path=str(prefix) + "_supervised.mp4",
    watch=args.watch,
    verbose=True,
)
print(f"supervised done — {r2['steps']} steps, {r2['retries']} retries")
if r2["log"]:
    for entry in r2["log"]:
        print(f"  step {entry['step']}: {entry['correction']['failure_type']} — {entry['correction']['corrected_instruction']}")
