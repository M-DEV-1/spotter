"""
Rung 3: unsupervised vs supervised with a scripted cube nudge.

Scripted failure: at step 200 (arm mid-approach), cube nudged 4cm in Y.
- Unsupervised: arm misses, finishes empty → FAIL
- Supervised:   3-agent pipeline (Perception→Planner→Validator) detects gripper_missed,
                issues corrected instruction, actor retries with corrected joint aim.

Usage:
  MUJOCO_GL=egl python scripts/run_rung3.py
Output:
  outputs/episodes/NNN_unsupervised.mp4
  outputs/episodes/NNN_supervised.mp4
  outputs/episodes/NNN_log.json
"""
import sys
import json
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


prefix = next_prefix()

print("=== RUN 1: unsupervised + perturbation ===")
r1 = run_supervised_episode(
    ClassicalActor(),
    perturb_fn=nudge,
    supervised=False,
    out_path=str(prefix) + "_unsupervised.mp4",
    verbose=True,
)
print(f"unsupervised done — {r1['steps']} steps\n")

print("=== RUN 2: supervised + perturbation (multi-agent) ===")
r2 = run_supervised_episode(
    ClassicalActor(),
    perturb_fn=nudge,
    supervised=True,
    out_path=str(prefix) + "_supervised.mp4",
    verbose=True,
)
print(f"supervised done — {r2['steps']} steps, {r2['retries']} retries")
for entry in r2["log"]:
    c = entry["correction"]
    trace_summary = [f"{m['from']}→{m['to']}" for m in entry.get("agent_trace", [])]
    print(f"  step {entry['step']}: {c['failure_type']}  trace: {' '.join(trace_summary)}")
    print(f"    {c['corrected_instruction']}")

log_path = str(prefix) + "_log.json"
with open(log_path, "w") as f:
    json.dump({
        "unsupervised_mp4": str(prefix) + "_unsupervised.mp4",
        "supervised_mp4":   str(prefix) + "_supervised.mp4",
        "unsupervised_steps": r1["steps"],
        "supervised_steps":   r2["steps"],
        "retries":            r2["retries"],
        "supervisor_log":     r2["log"],
    }, f, indent=2)
print(f"log → {log_path}")
