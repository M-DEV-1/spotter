"""
Run a supervised pick-and-place episode using SmolVLA and Cerebras LLM.
Usage:
  MUJOCO_GL=egl python scripts/run_supervised_vla.py           # record only
  MUJOCO_GL=egl python scripts/run_supervised_vla.py --watch   # record + live browser view
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.scene import load_model, make_data

from actor.smolvla import SmolVLAActor
from supervisor.supervise import run_supervised_episode

def next_episode_path(base="outputs/episodes"):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("???.mp4"))
    n = int(existing[-1].stem) + 1 if existing else 1
    return str(base / f"{n:03d}.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="stream live to http://spark-3100:8080")
    parser.add_argument("--no-supervisor", action="store_true", help="disable the multi-agent LLM supervisor")
    args = parser.parse_args()

    out = next_episode_path()
    print(f"=== SmolVLA Supervised Episode → {out} ===")

    model = load_model()
    data = make_data(model)

    actor = SmolVLAActor(model, data)

    # Create a small perturbation function to test the supervisor?
    # For now, we will just run the standard episode.
    def noop_perturb(m, d, step):
        pass

    result = run_supervised_episode(
        model=model,
        data=data,
        actor=actor,
        perturb_fn=noop_perturb,
        supervised=not args.no_supervisor,
        out_path=out,
        verbose=True,
        watch=args.watch,
    )
    
    print(f"done — {result['steps']} steps, {result['frames']} frames → {result['out_path']}")
    print(f"supervisor triggered {result['retries']} times")
