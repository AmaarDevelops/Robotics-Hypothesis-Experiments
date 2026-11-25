# Nemotron as the prior for berry picking (MuJoCo, real contacts, real rendered-image perception)

## Setup (about 3 minutes)
    python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env        # paste a NEW OpenRouter key; revoke the one pasted in chat; never commit .env
No rendering/GPU is needed: the camera-based perception results are pre-computed in data.pkl / real.pkl.

## 1. Check your physics matches the reference
    python berry_selftest.py                    # expect: ALL 40 MATCH

## 2. Pipeline test with a fake LLM (free; this is NOT Nemotron)
    python nemotron_berry.py --mock --quick --samples 2      # then delete the results/ folder

## 3. The real run
    python nemotron_berry.py                    # 5 samples x (1 prior + 2 feedback rounds) = 15 requests; hard cap 40 in code
    python nemotron_berry.py --transfer         # also runs the sim -> 'reality' patch loop on each trained policy (more compute, no extra requests)
Expect roughly 20-30 min (API latency was ~50 s per call earlier; learning runs ~90 s each). Outputs: results/nemotron_berry_results.json (all numbers) and results/llm_log.jsonl (every reply).

## What gets measured (per independent sample)
1. Zero-shot: held-out success (200 simulated berries) of Nemotron's policy with NO training.
2. After feedback: success after each of 2 revisions, when Nemotron sees outcome counts (picked / crushed / slipped / dropped / wrongly attempted).
3. After learning: success curve of the same CEM learner started from Nemotron's first policy, vs episodes of simulated experience.

## Reference numbers from the earlier experiments (same simulator, same held-out set)
- A policy written by Claude (me): zero-shot 0.70; about 0.86 after 4,800 episodes; 12/12 runs reached >=83%.
- No prior (from scratch): starts ~0.01, ends 0.51-0.91 (mean ~0.67); 6 of 15 runs got stuck at 0.51.
- 'Skip every berry' scores 0.51 (about half the berries are unripe). A confidently WRONG prior stayed at 0.51 in 3/3 runs.

## Honest caveats to keep in mind when reading the result
- The task description in nemotron_berry.py (TASK) is written by me and contains facts I know about the simulator (a leaf blocks one side, squeezing too hard crushes, unripe berries hold on harder, ripe means >= ~0.55). No numbers from my experiments are given, but a different description could change the outcome.
- The earlier 'Claude prior' was written by the same person who built the simulator; Nemotron's is not, which is the point of this test.
- 5 samples is small; sampling temperature is 0.7. Free-tier models may rate-limit (HTTP 429 is retried and counts toward the cap).
- Perception is a fixed hand-written colour segmenter, not a learned model.
