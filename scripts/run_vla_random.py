"""
Random cube displacement test for SmolVLA, with supervised Multi-Agent LLM recovery.

For each of N trials:
  - Cube nudged to a random position within reachable radius (5–20cm)
  - Unsupervised: VLA follows original trajectory → misses
  - Supervised: 3-agent pipeline fires after grasp, rewrites language instruction → recovers

Usage:
  MUJOCO_GL=egl python3 scripts/run_vla_random.py
  MUJOCO_GL=egl python3 scripts/run_vla_random.py --trials 3
"""
import sys
import json
import random
import argparse
import numpy as np
import mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from actor.smolvla import SmolVLAActor
from supervisor.supervise import run_supervised_episode
from simulator.scene import load_model, make_data

parser = argparse.ArgumentParser()
parser.add_argument("--trials",     type=int,   default=2)
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

def next_prefix(base="outputs/episodes/smolVLA"):
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
    model_unsup = load_model()
    data_unsup = make_data(model_unsup)
    actor_unsup = SmolVLAActor(model_unsup, data_unsup)

    r_unsup = run_supervised_episode(
        model=model_unsup, data=data_unsup, actor=actor_unsup,
        perturb_fn=nudge_fn, supervised=False,
        out_path=str(prefix) + f"_{label}_unsupervised.mp4", verbose=True,
    )

    print("-- supervised (LLM instruction recovery) --")
    model_sup = load_model()
    data_sup = make_data(model_sup)
    actor_sup = SmolVLAActor(model_sup, data_sup)

    r_sup = run_supervised_episode(
        model=model_sup, data=data_sup, actor=actor_sup,
        perturb_fn=nudge_fn, supervised=True,
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
vla_recoveries = sum(1 for r in results if any(p == "vla" for p in r["recovery_paths"]))
print(f"  VLA LLM recovery fired:  {vla_recoveries}/{args.trials}")
for r in results:
    paths = ",".join(r["recovery_paths"]) or "none"
    print(f"  trial {r['trial']}  r={r['displacement']['radius']:.2f}m"
          f"  retries={r['retries']}  paths={paths}")

log_path = Path("outputs/episodes/smolVLA/summary.json")
with open(log_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nlog → {log_path}")
