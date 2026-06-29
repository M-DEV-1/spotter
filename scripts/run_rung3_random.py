"""
Build B test: random cube displacement, supervised recovery via IK.

For each of N trials:
  - Cube nudged to a random position within reachable radius (5–20cm)
  - Unsupervised: arm follows original trajectory → misses
  - Supervised: 3-agent pipeline fires after grasp, IK computes exact joint targets → recovers

This is the "throw it anywhere" demo. User can also manually move the cube
between runs by editing qpos in a script or using run_manual_place.py.

Usage:
  MUJOCO_GL=egl python3 scripts/run_rung3_random.py
  MUJOCO_GL=egl python3 scripts/run_rung3_random.py --trials 3 --min-radius 0.05 --max-radius 0.20
"""
import sys
import json
import random
import argparse
import numpy as np
import mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from actor.classical import ClassicalActor
from supervisor.supervise import run_supervised_episode

parser = argparse.ArgumentParser()
parser.add_argument("--trials",     type=int,   default=5)
parser.add_argument("--min-radius", type=float, default=0.05)
parser.add_argument("--max-radius", type=float, default=0.20)
parser.add_argument("--seed",       type=int,   default=None)
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

NUDGE_STEP = 200

def make_nudge(dx, dy):
    def nudge(model, data, step):
        if step == NUDGE_STEP:
            data.qpos[9]  += dx
            data.qpos[10] += dy
            mujoco.mj_forward(model, data)
            print(f"  [perturbation] cube → Δ({dx:+.3f}, {dy:+.3f})  "
                  f"r={np.hypot(dx,dy):.3f}m")
    return nudge


def next_prefix(base="outputs/random"):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    candidates = sorted(base.glob("???_*.mp4"))
    n = int(candidates[-1].stem[:3]) + 1 if candidates else 1
    return base / f"{n:03d}"


results = []
for trial in range(args.trials):
    angle  = random.uniform(0, 2 * np.pi)
    radius = random.uniform(args.min_radius, args.max_radius)
    dx = radius * np.cos(angle)
    dy = radius * np.sin(angle)
    nudge_fn = make_nudge(dx, dy)

    prefix = next_prefix()
    label = f"r={radius:.2f}_a={int(np.degrees(angle))}"
    print(f"\n{'='*60}")
    print(f"TRIAL {trial+1}/{args.trials}  {label}")
    print(f"{'='*60}")

    print("-- unsupervised --")
    r_unsup = run_supervised_episode(
        ClassicalActor(), perturb_fn=nudge_fn, supervised=False,
        out_path=str(prefix) + f"_{label}_unsupervised.mp4", verbose=True,
    )

    print("-- supervised (IK recovery) --")
    r_sup = run_supervised_episode(
        ClassicalActor(), perturb_fn=nudge_fn, supervised=True,
        out_path=str(prefix) + f"_{label}_supervised.mp4", verbose=True,
    )

    result = {
        "trial": trial + 1,
        "displacement": {"dx": dx, "dy": dy, "radius": radius, "angle_deg": float(np.degrees(angle))},
        "unsupervised_steps": r_unsup["steps"],
        "supervised_steps":   r_sup["steps"],
        "retries":            r_sup["retries"],
        "recovery_paths":     [e.get("recovery_path", "?") for e in r_sup["log"]],
    }
    results.append(result)
    print(f"\n  result: {r_sup['retries']} retries, "
          f"unsup={r_unsup['steps']}steps  sup={r_sup['steps']}steps")

# summary
print(f"\n{'='*60}")
print(f"SUMMARY  ({args.trials} trials)")
print(f"{'='*60}")
ik_recoveries = sum(1 for r in results if any(p == "ik" for p in r["recovery_paths"]))
print(f"  IK recovery fired:  {ik_recoveries}/{args.trials}")
for r in results:
    paths = ",".join(r["recovery_paths"]) or "none"
    print(f"  trial {r['trial']}  r={r['displacement']['radius']:.2f}m"
          f"  retries={r['retries']}  paths={paths}")

log_path = Path("outputs/random/summary.json")
with open(log_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nlog → {log_path}")
