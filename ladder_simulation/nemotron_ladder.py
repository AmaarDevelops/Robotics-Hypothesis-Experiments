"""
nemotron_ladder.py -- Nemotron (via OpenRouter) controls a Franka Panda in robosuite's standard LIFT task by producing ACTION CHUNKS directly.
No learned policy sits between the model and the arm: it sees numbers (end-effector pose, gripper opening, cube pose, contacts) and outputs raw
action vectors; the simulator's built-in arm controller turns end-effector deltas into motion (level L2) or joint deltas (level L1).

  python nemotron_ladder.py --mock --episodes 3        pipeline test with a SCRIPTED fake (NOT an LLM), no API calls
  python nemotron_ladder.py --episodes 3               real run (uses .env; request cap enforced in code; resumable across days)
Options: --level L2|L1   --honest-scale (tell the model the real action scale)   --no-hints   --cap 45   --seed0 100
Per episode it records: success, steps, chunks (= model calls), grasp attempts, table-contact steps, cube-pushed distance, invalid replies, latency.
The API key is never printed or logged.
"""
import os, sys, re, json, time, argparse, math
import numpy as np
import Robot.ladder_simulation.ladder as Lm

MAX_STEPS_PER_CHUNK = 30
SYSTEM = "You are a robot-control expert. Reply with ONLY one JSON object and no other text. Keep any thinking very short."
TASK = {
"L2": """You control a Franka Panda robot arm in a physics simulator (robosuite 'Lift' task). GOAL: grasp the cube on the table and lift it at least 4 cm. You act by sending ACTION CHUNKS and after each chunk you receive the new state, so you can correct yourself.

Each action is [dx, dy, dz, drx, dry, drz, grip], numbers in [-1, 1], applied for `repeat` simulation steps (20 steps = 1 second).
- dx, dy, dz command end-effector translation in the world frame (x forward, y left, z up). Nominal scale: {SCALE}.
- drx, dry, drz rotate the gripper; use 0 unless you need them.
- grip: -1 = open, +1 = close.
A chunk is a list of entries [dx, dy, dz, drx, dry, drz, grip, repeat] (repeat = integer 1-30). Total steps in one chunk must be at most 30.

State you receive: eef = position of the point between the fingertips; gripper opening (m); cube position and yaw; eef->cube vector; how far the cube has been lifted; number of grasp attempts; table-contact steps (fingertips touching the table: avoid). You have at most 300 steps in total.
{HINTS}
Reply with ONLY JSON: {{"chunk": [[dx, dy, dz, 0, 0, 0, grip, repeat], ...], "reason": "one short sentence"}}""",
"L1": """You control a Franka Panda robot arm (7 joints) in a physics simulator (robosuite 'Lift' task). GOAL: grasp the cube on the table and lift it at least 4 cm. You act by sending ACTION CHUNKS and after each chunk you receive the new state.

Each action is [q1, q2, q3, q4, q5, q6, q7, grip], numbers in [-1, 1], applied for `repeat` simulation steps (20 steps = 1 second). q1..q7 command joint-angle changes. Nominal scale: {SCALE}. grip: -1 = open, +1 = close.
A chunk is a list of entries [q1..q7, grip, repeat] (repeat = integer 1-30). Total steps in one chunk must be at most 30.
State you receive: eef position, gripper opening, cube position/yaw, eef->cube vector, lift height, grasp attempts, table-contact steps (fingertips touching the table: avoid), joint angles and velocities. At most 300 steps in total.
{HINTS}
Reply with ONLY JSON: {{"chunk": [[q1, q2, q3, q4, q5, q6, q7, grip, repeat], ...], "reason": "one short sentence"}}"""}
SCALE = {"L2": ("1.0 is about 5 cm per step in each translation axis (the controller may realise less: calibrate from the feedback)", "1.0 is about 1 cm per step (measured)"),
         "L1": ("1.0 is about 0.05 rad per step (the arm may realise less: calibrate from the feedback)", "1.0 is about 0.008 rad per step (measured)")}
HINTS = "Hints: the cube is about 4 cm wide and the open gripper about 8 cm; close only when the fingertips are at the cube's height and centred on it; keep closing for about 10 steps before lifting."

# ------------------------------------------------------------ plumbing
def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
class Budget:
    def __init__(self, cap): os.makedirs("results", exist_ok=True); self.f = "results/request_count_ladder.json"; self.cap = cap; self.n = json.load(open(self.f)) if os.path.exists(self.f) else 0
    def spend(self):
        if self.n >= self.cap: raise SystemExit(f"Request cap {self.cap} reached; progress is saved, rerun later to resume. (Delete {self.f} to reset.)")
        self.n += 1; json.dump(self.n, open(self.f, "w"))
def _candidates(txt):
    txt = txt or ""; out = []
    try: out.append(json.loads(txt))
    except Exception: pass
    for blk in re.findall(r"```(?:json)?\s*(.*?)```", txt, flags=re.S):
        try: out.append(json.loads(blk))
        except Exception: pass
    depth, st = 0, None
    for i, ch in enumerate(txt):
        if ch == "{":
            if depth == 0: st = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and st is not None:
                try: out.append(json.loads(txt[st:i + 1]))
                except Exception: pass
    return [c for c in reversed(out) if isinstance(c, dict)]
def parse_chunk(txt, level):
    """-> (chunk|None, reason, problem). Validates entry length, clips actions to [-1,1], caps repeats, caps total steps."""
    need = 8 if level == "L2" else 9; problem = "no JSON object with a 'chunk' list found"
    for c in _candidates(txt):
        ch = c.get("chunk")
        if not isinstance(ch, list) or not ch: continue
        out, total = [], 0
        for e in ch:
            try:
                e = [float(x) for x in e]
                if len(e) == need - 1: e = e[:-1] + [e[-1]]        # tolerate a missing repeat -> handled below
                if len(e) != need: raise ValueError
                n = int(max(1, min(round(e[-1]), MAX_STEPS_PER_CHUNK))); n = min(n, MAX_STEPS_PER_CHUNK - total)
                if n <= 0: break
                out.append([float(np.clip(x, -1, 1)) for x in e[:-1]] + [n]); total += n
            except Exception: problem = f"each entry must be a list of exactly {need} numbers"; out = None; break
        if out: return out, str(c.get("reason", ""))[:200], ""
    return None, "", problem
def call_llm(bud, messages, temperature):
    import requests
    for attempt in range(3):
        bud.spend(); t0 = time.time()
        body = {"model": os.environ["LLM_MODEL"], "messages": messages, "temperature": temperature, "max_tokens": 12000}
        if not os.environ.get("NO_JSON_MODE"): body["response_format"] = {"type": "json_object"}
        try: r = requests.post(os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions", timeout=300, headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"], "Content-Type": "application/json"}, json=body)
        except Exception as e: print("   network error:", type(e).__name__); time.sleep(3); continue
        if r.status_code == 429: print("   rate-limited; waiting 20 s"); time.sleep(20); continue
        if r.status_code == 400 and "response_format" in r.text and not os.environ.get("NO_JSON_MODE"): print("   model rejects JSON mode -> retrying without it"); os.environ["NO_JSON_MODE"] = "1"; continue
        if r.status_code != 200: print("   HTTP", r.status_code, r.text[:160].replace(os.environ["LLM_API_KEY"], "***")); time.sleep(3); continue
        ch = r.json()["choices"][0]; msg = ch["message"]; txt = msg.get("content") or ""
        if not _candidates(txt): txt += "\n" + (msg.get("reasoning") or "")
        return txt, time.time() - t0
    return "", 0.0
def mock_chunk(ep, level):
    """SCRIPTED stand-in (NOT an LLM): proportional chunk controller, used only to test the pipeline."""
    o = ep.obs; e = o["robot0_eef_pos"]; c = o["cube_pos"]; d = c - e; v = lambda x: float(np.clip(x / 0.0105, -1, 1))
    if ep.attempts == 0 and (np.linalg.norm(d[:2]) > 0.006 or abs(d[2] + 0.0) > 0.012 and e[2] > c[2] + 0.02):
        z = c[2] + 0.12 if np.linalg.norm(d[:2]) > 0.01 and e[2] < c[2] + 0.1 else c[2] + 0.005
        tgt = np.array([c[0], c[1], z]) - e; return [[v(tgt[0]) * .8, v(tgt[1]) * .8, v(tgt[2]) * .8, 0, 0, 0, -1, 2]]
    if ep.attempts == 0: return [[0, 0, 0, 0, 0, 0, 1, 10]]
    return [[0, 0, 1.0, 0, 0, 0, 1, 8]]

# ------------------------------------------------------------ main loop
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--level", default="L2", choices=["L2", "L1"]); ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=100); ap.add_argument("--cap", type=int, default=45); ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--honest-scale", action="store_true"); ap.add_argument("--no-hints", action="store_true"); ap.add_argument("--max-chunks", type=int, default=0)
    a = ap.parse_args(); load_env()
    if not a.mock:
        for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
            if not os.environ.get(k): sys.exit(f"Missing {k}. Copy .env.example to .env and fill it in.")
    maxc = a.max_chunks or (20 if a.level == "L2" else 40); bud = Budget(a.cap); env = Lm.make(a.level)
    task = TASK[a.level].format(SCALE=SCALE[a.level][1 if a.honest_scale else 0], HINTS="" if a.no_hints else HINTS)
    print(("MOCK (scripted, NOT an LLM)" if a.mock else "Model: " + os.environ["LLM_MODEL"]), f"| level {a.level} | request cap {a.cap} | used so far {bud.n}")
    rows = []
    for k in range(a.episodes):
        seed = a.seed0 + k; f = f"results/{'mock_' if a.mock else ''}{a.level}_seed{seed}.json"
        S = json.load(open(f)) if os.path.exists(f) else dict(seed=seed, chunks=[], messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}], invalid=0, latency=[], done=False)
        ep = Lm.Ep(env, seed, S["chunks"])
        if S["done"]: rows.append((S, ep)); print(f"seed {seed}: already finished ({Lm.tag(ep)})"); continue
        print(f"\n=== episode seed {seed} ===")
        if not S["chunks"]: S["messages"].append({"role": "user", "content": "Initial state: " + ep.text(a.level) + "\nGive your first chunk."})
        while ep.succ_step is None and ep.t < Lm.HORIZON and len(S["chunks"]) < maxc:
            if a.mock: chunk, why = mock_chunk(ep, a.level), "mock"; txt = json.dumps(dict(chunk=chunk))
            else:
                chunk = None; msgs = list(S["messages"])
                for attempt in range(2):
                    txt, lat = call_llm(bud, msgs, a.temperature); S["latency"].append(round(lat, 1)); chunk, why, prob = parse_chunk(txt, a.level)
                    if chunk: break
                    S["invalid"] += 1; msgs += [{"role": "assistant", "content": (txt or "")[:300]}, {"role": "user", "content": f"Unusable reply ({prob}). Reply with ONLY the JSON object."}]
                if not chunk: print("   no usable chunk after retry; episode abandoned"); break
            S["messages"].append({"role": "assistant", "content": json.dumps(dict(chunk=chunk))}); ep.run(chunk); S["chunks"].append(chunk)
            state = ep.text(a.level); print(f"   chunk {len(S['chunks'])}: {sum(int(e[-1]) for e in chunk)} steps | {state[:150]}")
            S["messages"].append({"role": "user", "content": "New state: " + state + ("\nThe cube is lifted: SUCCESS." if ep.succ_step else "\nGive your next chunk.")})
            json.dump(S, open(f, "w"))
        S["done"] = True; json.dump(S, open(f, "w")); rows.append((S, ep)); print(f"   -> {Lm.tag(ep)} after {ep.t} steps")
    ok = [e.succ_step is not None for S, e in rows]
    summ = dict(model="mock" if a.mock else os.environ["LLM_MODEL"], level=a.level, episodes=len(rows), success=f"{sum(ok)}/{len(rows)}",
                chunks_per_episode=[len(S["chunks"]) for S, e in rows], steps=[e.t for S, e in rows], grasp_attempts=[e.attempts for S, e in rows],
                table_contact_steps=[e.table for S, e in rows], cube_pushed_m=[round(e.shift, 3) for S, e in rows], invalid_replies=[S["invalid"] for S, e in rows],
                median_latency_s=float(np.median([x for S, e in rows for x in S["latency"]] or [0])), requests_used=bud.n)
    json.dump(summ, open("results/summary_" + ("mock_" if a.mock else "") + a.level + ".json", "w"), indent=1); print("\n==== SUMMARY ====\n" + json.dumps(summ, indent=1))
    print("REFERENCE (Claude as controller, same harness, seeds 100-104): L2 5/5, 32 steps, 5 chunks/episode; L1 5/5, 109-137 steps, ~9 chunks/episode, 0-14 table-contact steps.")
