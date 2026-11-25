# Nemotron controls a Franka Panda directly (robosuite Lift) -- Phase 1

The model sees numbers (end-effector pose, gripper opening, cube pose, contacts) and outputs RAW ACTION CHUNKS. No learned policy sits between it
and the arm. Level L2 = end-effector deltas (default). Level L1 = joint-angle deltas (needs ~40 model calls per episode: only with a big quota or a local model).

## Setup  (no GPU needed)
robosuite on native Windows is unreliable. Use WSL2 (Ubuntu) or Google Colab (CPU is enough).
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt             # mujoco 3.3.5 is required: robosuite breaks on newer MuJoCo
    cp .env.example .env                        # paste a NEW OpenRouter key; revoke the one pasted in chat earlier
Colab: `!pip install mujoco==3.3.5 "numpy<2" robosuite requests`, upload these files, set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL with os.environ, restart the runtime if numpy complains.

## Run
    python ladder_selftest.py                       # expect PASS (a scripted controller lifts the cube through the same chunk interface)
    python nemotron_ladder.py --mock --episodes 3   # pipeline test with a SCRIPTED fake (not an LLM); then delete the results/ folder
    python nemotron_ladder.py --episodes 3          # the real run
Budget: each episode takes roughly 8-20 requests (one per chunk), so 3 episodes fit inside the 45-request cap. Progress is saved after every chunk and the run
is RESUMABLE: when your daily request allowance resets, run the same command again and it continues. Expect about 1 minute per request (reasoning models are slow).

## What to send back
results/summary_L2.json plus results/L2_seed*.json (full conversations; the key is never logged).

## Reference: Claude as the controller, same harness, seeds 100-104
L2: 5/5 success, 32 steps, 5 chunks per episode, 1 grasp attempt each, 0-1 table-contact steps.
L1: 5/5 success, 109-137 steps, ~9 chunks per episode, 0-14 table-contact steps (needed in-context calibration by probing).
The first L2 chunk was miscalibrated: the prompt says 1.0 ~ 5 cm per step but the controller realises ~1 cm. The model has to notice this from feedback.

## Watch the robot (MP4 replays)
    python render_replay.py results/L2_seed100.json results/L2_seed102.json     # -> videos/*.mp4  (headless; set MUJOCO_GL=egl, or osmesa if EGL fails)
    python render_replay.py results/L2_seed101.json --vs other.json --label Nemotron --vs-label Other   # side by side
Replays are deterministic to within about a step in contact-heavy episodes (the log says 83, the replay may say 84).
A LIVE interactive window needs a display (Linux desktop or Windows 11 WSLg) and robosuite's on-screen viewer; Colab can only show videos.

## Phase 2: Stack (pick the red cube, place it on the green one)
    python nemotron_ladder.py --task Stack --episodes 2            # seeds 200+; about 10-20 requests per episode, so 1-2 episodes per daily allowance
    python nemotron_ladder.py --task Stack --episodes 2 --protocol  # with an explicit 8-step protocol in the prompt
Reference (Claude, same harness, seeds 200-204): 5/5 success, 68-70 steps, 9 chunks per episode, 1 grasp attempt each, 1-4 table-contact steps, cube A pushed 7-13 mm.
Tip: re-use `python render_replay.py results/Stack_L2_seed200.json` to make a video (the task is read from the file).

## Variants
    --protocol       adds an explicit approach / close / lift protocol (tests whether failures are about discipline or about capability)
    --honest-scale   tell the model the real scale (1 cm per unit)      --no-hints   remove the grasping hints      --seed0 N   different start seeds

## Caveats (keep in mind when reading results)
- The harness, prompt and hints were written by me, and Lift is robosuite's easiest task. 3 episodes is a tiny sample.
- The prompt deliberately contains the same (slightly wrong) nominal scale I had, so that the comparison is like-for-like.
- Contact counters can flicker by a few steps between replays; success flags are stable.
- Perception is simplified: the model is given exact object positions as numbers, not camera images.
