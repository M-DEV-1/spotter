"""
Run one classical pick-and-place episode and save a serially-numbered mp4.
Usage:
  MUJOCO_GL=egl python scripts/run_episode.py           # record only
  MUJOCO_GL=egl python scripts/run_episode.py --watch   # record + live browser view
Output: outputs/episodes/NNN.mp4  (pull with: make pull)
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from actor.classical import ClassicalActor
from tasks.pick_place import init_task
from simulator.control import run_episode


def next_episode_path(base="outputs/episodes"):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("???.mp4"))
    n = int(existing[-1].stem) + 1 if existing else 1
    return str(base / f"{n:03d}.mp4")


parser = argparse.ArgumentParser()
parser.add_argument("--watch", action="store_true", help="stream live to http://spark-3100:8080")
args = parser.parse_args()

out = next_episode_path()
print(f"=== classical pick-and-place → {out} ===")

result = run_episode(
    ClassicalActor(),
    task_init_fn=init_task,
    out_path=out,
    verbose=True,
    watch=args.watch,
)
print(f"done — {result['steps']} steps, {result['frames']} frames → {result['out_path']}")
