# Local LLM-as-planner grasp experiment

## 1. Setup (about 3 minutes)
    python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env                                   # then edit .env: paste your NEW OpenRouter key
(Use a fresh key, not the one pasted into the chat. Revoke it afterwards. Never commit `.env`.)

## 2. Check the physics matches the original runs
    python selftest.py
Expect: `ALL MATCH`. If you see MISMATCHES, you probably have a different MuJoCo version than 4.0.1
(`pip install mujoco==4.0.1`). Small numeric drift across versions/platforms is possible; the pass/fail outcomes are what matter.

## 3. Pipeline test without the API (optional, free)
    python run.py --mock --baseline-samples 20
(The mock planner is a trivial heuristic, NOT an LLM. Delete the `results/` folder afterwards.)

## 4. The real run (max 50 API requests, enforced in code)
    python run.py
Takes a few minutes. Outputs: `results/summary.json` (headline numbers), `results/llm_log.jsonl` (every prompt/response).
The request counter persists in `results/request_count.json`; free-tier rate limits (HTTP 429) are retried and count toward the cap.

## 5. Optional GUI
    python view_demo.py 7          # watch a grasp of object 7 in MuJoCo's viewer (macOS: mjpython view_demo.py)
The viewer only draws what the headless physics computes; results do not change.

## What the run measures
- Phase A: zero-shot first-try success on 8 named objects (vs fixed 10 N / 20 N grasps).
- Phase B: recovery on 6 hidden-property objects where a generic grasp fails (attempts to success vs a hand-coded retry rule and random search).
- Invalid-reply count and per-call latency.
Each object is a separate conversation (no memory across objects), so this is the *stateless* condition.
