"""
Classical actor demo pair: unsupervised FAIL vs supervised RECOVER.

A cube nudge at step 200 makes the unsupervised run miss the grasp and finish
empty. The supervised run lets Gemma 4 31B (Cerebras) observe the miss, rewrite
the instruction, and the classical actor retries via IK recovery.

Usage:
  MUJOCO_GL=egl python scripts/run_classical_demo.py --nudge 0.08
Output:
  outputs/episodes/classical_unsupervised.mp4
  outputs/episodes/classical_supervised.mp4
  outputs/episodes/classical_log.json
"""
import sys, json, argparse
import mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.scene import load_model, make_data
from actor.classical import ClassicalActor
from supervisor.supervise import run_supervised_episode

parser = argparse.ArgumentParser()
parser.add_argument('--nudge', type=float, default=0.08)
parser.add_argument('--step', type=int, default=200)
args = parser.parse_args()

def make_nudge(dy, at):
    def nudge(m, d, step):
        if step == at:
            d.qpos[10] += dy
            mujoco.mj_forward(m, d)
            print(f'  [nudge] cube +{dy:.3f}m Y at step {step}')
    return nudge

base = Path('outputs/episodes'); base.mkdir(parents=True, exist_ok=True)
model = load_model()
data = make_data(model)
actor = ClassicalActor()

nudge = make_nudge(args.nudge, args.step)

print('=== RUN 1: unsupervised + nudge ===')
r1 = run_supervised_episode(actor, perturb_fn=nudge, supervised=False,
    out_path=str(base/'classical_unsupervised.mp4'), verbose=True,
    model=model, data=data)
print(f'unsupervised — {r1["steps"]} steps, {r1["retries"]} retries\n')

print('=== RUN 2: supervised + nudge ===')
r2 = run_supervised_episode(actor, perturb_fn=nudge, supervised=True,
    out_path=str(base/'classical_supervised.mp4'), verbose=True,
    model=model, data=data)
print(f'supervised — {r2["steps"]} steps, {r2["retries"]} retries')
for e in r2['log']:
    c = e['correction']
    print(f"  step {e['step']}: {c['failure_type']} :: {c['corrected_instruction']}")

with open(base/'classical_log.json','w') as f:
    json.dump({
        'unsupervised_mp4': str(base/'classical_unsupervised.mp4'),
        'supervised_mp4': str(base/'classical_supervised.mp4'),
        'unsupervised_steps': r1['steps'],
        'supervised_steps': r2['steps'],
        'retries': r2['retries'],
        'supervisor_log': r2['log'],
    }, f, indent=2)
print('log -> outputs/episodes/classical_log.json')
