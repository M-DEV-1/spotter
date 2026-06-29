# CLAUDE.md — Spotter

## What this is
Spotter is a 24-hour hackathon project (Cerebras × Google DeepMind Gemma 4
Hackathon). It's an **engineering exploration**, NOT a research contribution or a
novelty claim. We are reproducing-and-exploring an FPC-VLA-style semantic
supervision pattern, citing prior work openly.

**One-line description:** A frozen robot policy performs pick-and-place in MuJoCo
simulation; Gemma 4 31B on Cerebras acts as an external "supervisor" that watches
camera frames + cheap failure signals, diagnoses failures in language, and issues a
corrected instruction. We explore what this supervision feels like when the
supervisor is fast (Cerebras) vs slow.

Framing rule: always use "explored / investigated / evaluated", NEVER "invented /
novel / our new architecture". Prior art we build on: FPC-VLA (arXiv 2509.04018),
RoboFAC (2505.12224), CycleVLA (2601.02295); lineage SayCan (2022), Inner
Monologue (2022), Code as Policies (2023).

## Hard constraints (do not violate)
- **Time:** ~24h total, solo builder. Scope is the enemy. Prefer the smallest thing
  that works over the most impressive thing.
- **Hardware:** NVIDIA DGX Spark, ARM64 (aarch64), headless (no display — render
  via `MUJOCO_GL=egl`, view via mjviser web server on a port). Some x86 wheels won't
  build; prefer packages with ARM wheels.
- **Cerebras API:** model id is exactly `gemma-4-31b`. OpenAI-compatible Chat
  Completions. Limits: 100 RPM, 100K TPM, context 5K MSL / 32K MCL. Image input via
  standard OpenAI `image_url` with base64 data URIs. Tool calling supported, use
  `strict: true` to lock JSON schema. Reasoning off by default (`reasoning_effort`).
  Text+image in, text out — NO video input, NO audio.
- **Rate limit math:** 100 RPM = ~1.6 req/sec. The supervisor loop runs at ~1Hz, NOT
  every physics step. Multi-agent (3 calls/cycle) must stay under 100 RPM — keep the
  realtime demo loop to a single supervisor call unless explicitly told otherwise.

## Tech stack (confirmed working)
- `mujoco` (plain engine, ARM wheel works, EGL headless render confirmed) — primary
  simulator. Genesis 1.2.0 and MuJoCo Playground are also installed and available if
  they make something easier; default to plain MuJoCo unless there's a good reason.
- `robot_descriptions` — loads the Franka Panda model (`panda_mj_description.MJCF_PATH`).
- `mujoco_menagerie` cloned at repo root for scene.xml + mjx_single_cube.xml (arm+cube).
- `mjviser` — web-based viewer on a port, for human dev viewing only.
- `cerebras-cloud-sdk` — the supervisor.
- (later) `lerobot` + SmolVLA — the frozen VLA actor. Treated as a CLIMB-TO goal, not
  a guaranteed dependency.

THE DGX SPARK LIVES ON tailscale. To connect you must do ```sudo tailscale ssh zugzwang@spark-3100```. Try without sudo. if it only works with sudo. Notify the user to help you connect.

## The descope ladder (current position drives all decisions)
Build bottom-up. Ship the highest rung that works. Never skip a rung.
- Rung 0: sim renders a frame. ✅ DONE
- Rung 0.5: arm loads + moves programmatically.  ← current focus
- Rung 1: one successful pick-and-place, CLASSICAL IK/PD control, NO LLM. ← THE FLOOR.
  Once this works we have a guaranteed submission.
- Rung 2: Cerebras call returns a structured corrective instruction from frame + signal.
- Rung 3: supervisor wired into loop, recovers a SCRIPTED failure (classical actor).
- Rung 4: swap classical actor → SmolVLA (the authentic version).
- Rung 5: side-by-side with/without supervision + fast-vs-slow supervisor.

Tell me which rung we're on; don't build rung N+2 while rung N is unproven.

## Your role (Claude Code)
Do the PLUMBING so the human does the architecture and the robotics decisions. That
means: boilerplate, config, CLI scaffolding, logging, file I/O, render-to-mp4
helpers, retry/clip/validate wrappers, prompt templating, glue code. Write small,
runnable, testable pieces. After any sim or API code, give a one-command way to run
and verify it.

Do NOT:
- Add dependencies not listed above without flagging why.
- Build abstraction layers, plugin systems, or "modes" before a single path works
  end-to-end. No premature "playground" framework. One working path first.
- Guess Cerebras request formats — the confirmed shapes are in this file; if
  something's not here, say "confirm in Cerebras docs" rather than inventing it.
- Trust raw model output: always clip/validate waypoints to a safe workspace box and
  guard `if message.tool_calls:` before using a tool call.
- Make the supervisor loop exceed the rate limit.
- Downgrade the quality of a lower rung to save time for a higher rung. The floor
  must be demo-solid before climbing. A floor that "might fail" is not a floor.

## Key design decisions (source of truth for README + demo narration)
- Supervisor outputs a REWRITTEN INSTRUCTION (the "coach"), not raw waypoints — this
  keeps it distinct from SayCan-style waypoint planning. (Until SmolVLA is wired,
  the classical floor version may use waypoints; label that honestly as the fallback.)
- Failure detection uses CHEAP sim ground-truth signals (gripper-closed-but-empty,
  object out of region, pose mismatch, no-progress timeout) — Gemma does SEMANTIC
  interpretation of those signals, it does not guess failure from pixels alone.
- Speed claim is honest and narrow: fast inference enables CONTINUOUS supervision
  vs keyframe-only, catching failures sooner before the scene drifts. NOT "only works
  with Cerebras". Network round-trip is unaffected by Cerebras; only token gen speeds up.

## Demo target (drives priorities)
60-second video. The single memorable moment: reach for object → object nudged out
of place → unsupervised run fails → Gemma observes + rewrites instruction →
supervised run succeeds. Side-by-side fast-vs-slow recommended. Everything serves
making that 60s clip legible.

## Git commits
- Message: 2–3 words, title only, no body, no description
- Always sign off: `git commit -s -m "title"`
- No "Co-Authored-By" lines

## Repo layout
README.md  DESIGN.md  DECISIONS.md  TODO.md
simulator/   # mujoco scene, control loop, render helpers
actor/       # classical controller now; SmolVLA later
supervisor/  # cerebras client, prompts, failure signals
prompts/     # supervisor system prompts
scripts/     # run helpers, demo recording
experiments/ # comparison runs (with/without, fast/slow)
