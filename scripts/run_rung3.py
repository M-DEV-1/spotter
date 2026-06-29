"""
Rung 3: unsupervised vs supervised with a scripted cube nudge.

Scripted failure: at step 200 (arm mid-approach), cube nudged 4cm in Y.
- Unsupervised: arm misses, finishes empty → FAIL
- Supervised:   Gemma detects gripper_missed, actor retries with corrected joint1 aim

Usage:
  MUJOCO_GL=egl python scripts/run_rung3.py           # record only
  MUJOCO_GL=egl python scripts/run_rung3.py --watch   # + live browser view
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

NUDGE_STEP = 200
NUDGE_Y    = 0.04  # 4cm


def nudge(model, data, step):
    if step == NUDGE_STEP:
        data.qpos[10] += NUDGE_Y
        mujoco.mj_forward(model, data)
        print(f"  [perturbation] cube nudged +{NUDGE_Y}m in Y at step {step}")


def next_prefix(base="outputs/episodes"):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    candidates = sorted([*base.glob("???_*.mp4"), *base.glob("???.mp4")])
    n = int(candidates[-1].stem[:3]) + 1 if candidates else 1
    return base / f"{n:03d}"


parser = argparse.ArgumentParser()
parser.add_argument("--watch", action="store_true", help="live browser view at http://spark-3100:8080")
args = parser.parse_args()

# start viser server once so browser loads before both runs
watch_server = None
if args.watch:
    import time, viser
    watch_server = viser.ViserServer(port=8080)
    print(f"\nopen browser → http://spark-3100:8080")
    print("(server stays live across both runs)\n")
    for i in range(12, 0, -1):
        print(f"  starting in {i}s ...  ", end="\r", flush=True)
        time.sleep(1)
    print("  starting now!          \n")

prefix = next_prefix()

print("=== RUN 1: unsupervised + perturbation ===")
r1 = run_supervised_episode(
    ClassicalActor(),
    perturb_fn=nudge,
    supervised=False,
    out_path=str(prefix) + "_unsupervised.mp4",
    watch=args.watch,
    viser_server=watch_server,
    verbose=True,
)
print(f"unsupervised done — {r1['steps']} steps\n")

if args.watch:
    import time
    print("--- switching to supervised run in 3s ---")
    time.sleep(3)

print("=== RUN 2: supervised + perturbation ===")
r2 = run_supervised_episode(
    ClassicalActor(),
    perturb_fn=nudge,
    supervised=True,
    out_path=str(prefix) + "_supervised.mp4",
    watch=args.watch,
    viser_server=watch_server,
    verbose=True,
)
print(f"supervised done — {r2['steps']} steps, {r2['retries']} retries")
for entry in r2["log"]:
    print(f"  step {entry['step']}: {entry['correction']['failure_type']} — {entry['correction']['corrected_instruction']}")

# save log for web demo export
import json
log_path = str(prefix) + "_log.json"
with open(log_path, "w") as f:
    json.dump({
        "unsupervised_mp4": str(prefix) + "_unsupervised.mp4",
        "supervised_mp4": str(prefix) + "_supervised.mp4",
        "unsupervised_steps": r1["steps"],
        "supervised_steps": r2["steps"],
        "retries": r2["retries"],
        "supervisor_log": r2["log"],
    }, f, indent=2)
print(f"log → {log_path}")
