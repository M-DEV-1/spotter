"""
Run supervised Pi0 episodes in a loop — weights loaded once, reused across runs.

Usage:
  MUJOCO_GL=egl python scripts/run_supervised_pi0.py                  # supervised loop
  MUJOCO_GL=egl python scripts/run_supervised_pi0.py --no-supervisor  # unsupervised
  MUJOCO_GL=egl python scripts/run_supervised_pi0.py --nudge 0.08     # nudge cube Y
  MUJOCO_GL=egl python scripts/run_supervised_pi0.py --watch           # viser live view
"""
import sys
import argparse
import mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.scene import load_model, make_data
from actor.pi0 import Pi0Actor
from supervisor.supervise import run_supervised_episode

PI0_MODEL_PATH = "/home/zugzwang/models/pi0_libero"

parser = argparse.ArgumentParser()
parser.add_argument("--no-supervisor", action="store_true")
parser.add_argument("--nudge", type=float, default=0.0, help="nudge cube Y at step 200")
parser.add_argument("--watch", action="store_true")
parser.add_argument("--model-path", default=PI0_MODEL_PATH)
parser.add_argument("--max-steps", type=int, default=6000)
args = parser.parse_args()

def next_episode_path(base="outputs/episodes/pi0"):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("???.mp4"))
    n = int(existing[-1].stem) + 1 if existing else 1
    return str(base / f"{n:03d}.mp4")

def make_nudge(dy):
    def nudge(m, d, step):
        if step == 200:
            d.qpos[10] += dy
            mujoco.mj_forward(m, d)
            print(f"  [nudge] cube +{dy:.3f}m Y")
    return nudge

print("Loading scene...")
model = load_model()
data = make_data(model)

print(f"Loading Pi0 weights from {args.model_path} ...")
actor = Pi0Actor(model, data, model_id=args.model_path)
print("Ready.\n")

while True:
    out = next_episode_path()
    sup_label = "supervised" if not args.no_supervisor else "unsupervised"
    print(f"=== Pi0 {sup_label} → {out} ===")

    perturb = make_nudge(args.nudge) if args.nudge else None

    result = run_supervised_episode(
        actor=actor,
        model=model,
        data=data,
        perturb_fn=perturb,
        supervised=not args.no_supervisor,
        max_steps=args.max_steps,
        out_path=out,
        verbose=True,
        watch=args.watch,
    )
    print(f"done — {result['steps']} steps, {result['frames']} frames")
    print(f"supervisor fired {result['retries']} times\n")

    cmd = input("Enter to run again, 'q' to quit: ").strip().lower()
    if cmd == "q":
        break
