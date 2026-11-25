"""
nemotron_berry.py -- Nemotron (via OpenRouter) as the 'prior' for berry picking in the MuJoCo simulation.

  python nemotron_berry.py --mock --quick     pipeline test with a fake LLM (NOT Nemotron), no API calls
  python nemotron_berry.py                    real run (uses .env; HARD request cap in code)

For each of N independent samples (separate conversations, no shared memory):
  1. Nemotron is given a plain-language task description (NO simulator code, no numbers from the experiments) and writes a picking policy.
  2. ZERO-SHOT: that policy is scored on 200 held-out simulated berries, with no training at all.
  3. FEEDBACK: Nemotron is shown outcome counts from 60 simulated berries and revises the policy (2 rounds). Each revision is scored on the same held-out berries.
  4. LEARNING: the same CEM learner used in the earlier experiments starts from Nemotron's FIRST policy and improves it from simulated experience.
Optional --transfer: runs the sim -> 'reality' patch loop on the trained policy.

Everything is logged to results/.  The API key is never printed or logged.
"""
import os, sys, re, json, time, math, pickle, random, argparse
import numpy as np
import Robot.berry_simulation.learn as L, Robot.berry_simulation.berry_sim as B

CAP = 40
BOUNDS = dict(pick_if_ripeness_above=(0, 1), approach_offset_from_leaf_deg=(-180, 180), finger_gap_cm=(3.4, 7.5), squeeze_force_N=(0.3, 25),
              pull_offset_from_stem_deg=(-180, 180), pull_elevation_deg=(-90, 60), pull_speed_m_per_s=(0.03, 0.15), twist_rad=(0, 1.6))
SYSTEM = "You are an expert in robotics and agriculture. Reply with ONLY one JSON object and no other text."
TASK = """You are setting the picking policy of a berry-picking robot BEFORE it is trained. The robot works in a physics simulator. Facts about the world:

- A berry (about 2.8 cm across, about 12 g) hangs from a stem. The stem leaves the berry going upward at an angle, toward where it attaches to the plant.
- A flat leaf stands next to the berry, about 3 cm from it, on one side. The gripper cannot pass through the leaf.
- Berries are ripe (red) or unripe (pale green/yellow). A berry counts as ripe if its true ripeness is at least about 0.55 on a 0-1 scale. Unripe berries are fastened to their stem much more firmly than ripe ones.
- Two cameras give NOISY estimates: ripeness r (0 = unripe, 1 = fully red; typical error about 0.1), leaf bearing and stem bearing. Bearings are directions seen from above, in radians, x to the right and y up. Leaf bearing = direction from the berry toward the leaf. Stem bearing = direction from the berry toward where the stem attaches (horizontal part). Bearings are usually right to within a few degrees, but the stem bearing is occasionally far off.
- The hand is a two-finger parallel gripper. It starts 10 cm from the berry at a chosen bearing around the berry, moves horizontally to the berry, closes its fingers with a chosen squeeze force (per finger), then pulls along a chosen direction for 7 cm at a chosen speed while rotating about the vertical axis by a chosen twist angle.
- The berry comes off when the stem is pulled hard enough. A berry squeezed too hard is crushed; squeezed too softly it can slip out. A poor pull direction can fail to detach the berry or lose it.

Scoring: for a ripe berry you succeed only if it is detached, still held at the end, and not crushed. For an unripe berry you succeed only by leaving it alone (the policy skips it); trying to pick it is a failure.

Choose these policy parameters (all numeric, within the stated ranges):
- pick_if_ripeness_above (0 to 1): the robot attempts a pick only if the perceived ripeness is above this.
- approach_offset_from_leaf_deg (-180 to 180): the hand starts at bearing = leaf bearing + this offset.
- finger_gap_cm (3.4 to 7.5): finger opening before closing.
- squeeze_force_N (0.3 to 25): squeeze force per finger.
- pull_offset_from_stem_deg (-180 to 180): the hand moves horizontally in the direction (stem bearing + this offset).
- pull_elevation_deg (-90 to 60): upward angle of the pull; negative means downward.
- pull_speed_m_per_s (0.03 to 0.15).
- twist_rad (0 to 1.6): rotation of the hand about the vertical axis during the pull.

Reply with ONLY JSON: {"pick_if_ripeness_above": ..., "approach_offset_from_leaf_deg": ..., "finger_gap_cm": ..., "squeeze_force_N": ..., "pull_offset_from_stem_deg": ..., "pull_elevation_deg": ..., "pull_speed_m_per_s": ..., "twist_rad": ..., "reason": "one or two sentences"}"""

# ---------------------------------------------------------------- plumbing
def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

class Budget:
    def __init__(self):
        os.makedirs("results", exist_ok=True); self.f = "results/request_count_berry.json"
        self.n = json.load(open(self.f)) if os.path.exists(self.f) else 0
    def spend(self):
        if self.n >= CAP: raise SystemExit(f"Request cap {CAP} reached; stopping. (Delete {self.f} to reset.)")
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

def _num(v):
    if isinstance(v, bool): return None
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", v)
        if m: return float(m.group())
    return None

def parse_policy(txt):
    """-> (dict|None, reason, problem). Tolerates fences/prose/units; clips out-of-range values and says so."""
    problem = "no JSON object found"
    for c in _candidates(txt):
        flat = {re.sub(r"[^a-z0-9_]", "", str(k).lower()): v for k, v in c.items()}
        p, missing = {}, []
        for k in BOUNDS:
            val = _num(flat.get(k.lower()))
            if val is None or val != val or abs(val) == float("inf"): missing.append(k)
            else: p[k] = val
        if missing: problem = "missing/non-numeric: " + ", ".join(missing); continue
        clipped = [k for k, v in p.items() if not (BOUNDS[k][0] <= v <= BOUNDS[k][1])]
        p = {k: float(min(max(v, BOUNDS[k][0]), BOUNDS[k][1])) for k, v in p.items()}
        return p, str(flat.get("reason", ""))[:300], ("clipped: " + ", ".join(clipped)) if clipped else ""
    return None, "", problem

def to_theta(p):
    """readable policy -> the 19-number policy vector used by the learner."""
    def rot(deg):
        a = math.radians(deg); c, s = math.cos(a), math.sin(a); return [c, -s, 0, s, c, 0]
    return np.array([10.0, -10.0 * p["pick_if_ripeness_above"], *rot(p["approach_offset_from_leaf_deg"]), p["finger_gap_cm"] / 100.0, math.log(p["squeeze_force_N"]),
                     *rot(p["pull_offset_from_stem_deg"]), p["pull_elevation_deg"], p["pull_speed_m_per_s"], p["twist_rad"]])

def call_llm(bud, messages, temperature, mock=None):
    if mock is not None: return mock(messages), 0.0
    import requests
    for attempt in range(3):
        bud.spend(); t0 = time.time()
        body = {"model": os.environ["LLM_MODEL"], "messages": messages, "temperature": temperature, "max_tokens": 4000}
        if not os.environ.get("NO_JSON_MODE"): body["response_format"] = {"type": "json_object"}
        try:
            r = requests.post(os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions", timeout=240,
                              headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"], "Content-Type": "application/json"}, json=body)
        except Exception as e:
            print("   network error:", type(e).__name__); time.sleep(3); continue
        if r.status_code == 429: print("   rate-limited; waiting 20 s"); time.sleep(20); continue
        if r.status_code == 400 and "response_format" in r.text and not os.environ.get("NO_JSON_MODE"):
            print("   model rejects JSON mode -> retrying without it"); os.environ["NO_JSON_MODE"] = "1"; continue
        if r.status_code != 200: print("   HTTP", r.status_code, r.text[:200].replace(os.environ["LLM_API_KEY"], "***")); time.sleep(3); continue
        ch = r.json()["choices"][0]; msg = ch["message"]; txt = msg.get("content") or ""
        if not _candidates(txt): txt += "\n" + (msg.get("reasoning") or "")
        return txt, time.time() - t0
    return "", 0.0

# ---------------------------------------------------------------- simulation-side helpers
def outcome_report(theta, eps):
    ripe = [e for e in eps if e["ripe"]]; unripe = [e for e in eps if not e["ripe"]]
    def run(group):
        c = {}
        for e in group:
            o = B.episode_success(e, L.to_act(theta, e["obs"]))[1]["outcome"]; c[o] = c.get(o, 0) + 1
        return c
    cr, cu = run(ripe), run(unripe); g = lambda c, k: c.get(k, 0)
    return (f"Your policy was run once on each of {len(eps)} simulated berries ({len(ripe)} ripe, {len(unripe)} unripe).\n"
            f"Ripe berries: {g(cr,'picked')} picked cleanly, {g(cr,'crushed')} crushed, {g(cr,'slipped_or_stuck')} slipped or the berry stayed attached, "
            f"{g(cr,'dropped')} detached but dropped, {g(cr,'skipped')} skipped (a miss).\n"
            f"Unripe berries: {g(cu,'skipped')} correctly left alone, {len(unripe) - g(cu,'skipped')} wrongly attempted.")

def policy_text(p): return json.dumps({k: round(v, 4) for k, v in p.items()})

# ---------------------------------------------------------------- mock LLM (pipeline test only; NOT Nemotron)
def make_mock(rng):
    def mock(messages):
        n = sum(1 for m in messages if m["role"] == "assistant")
        f = 4.0 * (1 + 0.3 * rng.gauss(0, 1)) if n == 0 else 3.0
        return json.dumps(dict(pick_if_ripeness_above=0.6, approach_offset_from_leaf_deg=180, finger_gap_cm=4.5, squeeze_force_N=max(0.5, f),
                               pull_offset_from_stem_deg=180, pull_elevation_deg=-50, pull_speed_m_per_s=0.06, twist_rad=0.5, reason="mock"))
    return mock

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--quick", action="store_true", help="short learning runs (pipeline test)")
    ap.add_argument("--samples", type=int, default=5); ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-train", action="store_true"); ap.add_argument("--transfer", action="store_true")
    a = ap.parse_args(); load_env()
    if not a.mock:
        for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
            if not os.environ.get(k): sys.exit(f"Missing {k}. Copy .env.example to .env and fill it in.")
    D = pickle.load(open("data.pkl", "rb")); bud = Budget(); os.makedirs("results", exist_ok=True); log = open("results/llm_log.jsonl", "a")
    print(("MOCK LLM (NOT Nemotron)" if a.mock else "Planner model: " + os.environ["LLM_MODEL"]), "| request cap", CAP, "| used so far", bud.n)
    diag = D["train"][:60]; results = []
    for s in range(a.samples):
        mock = make_mock(random.Random(s)) if a.mock else None
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]; policy = None; revs = []
        for attempt in range(2):
            txt, lat = call_llm(bud, msgs, a.temperature, mock); policy, why, prob = parse_policy(txt)
            log.write(json.dumps(dict(sample=s, stage="initial", response=txt, parsed=policy, problem=prob, latency_s=round(lat, 1))) + "\n"); log.flush()
            if policy: break
            msgs += [{"role": "assistant", "content": (txt or "")[:400]}, {"role": "user", "content": f"Unusable reply ({prob}). Reply with ONLY the JSON object with all eight numeric fields."}]
        if not policy: print(f"sample {s}: no usable policy"); results.append(dict(sample=s, failed=True)); continue
        th0 = to_theta(policy); z0 = L.evaluate(th0, D["test"]); print(f"\nsample {s}: zero-shot policy {policy_text(policy)}\n   reason: {why}\n   held-out success with NO training: {z0:.2f}")
        cur, curtheta, hist = policy, th0, [z0]
        msgs.append({"role": "assistant", "content": json.dumps({**{k: round(v, 4) for k, v in policy.items()}, "reason": why})})
        for rd in range(1, a.rounds + 1):
            msgs.append({"role": "user", "content": outcome_report(curtheta, diag) + "\n\nRevise the policy to do better. Reply with ONLY the JSON object (all eight fields + reason)."})
            txt, lat = call_llm(bud, msgs, a.temperature, mock); p2, why2, prob2 = parse_policy(txt)
            log.write(json.dumps(dict(sample=s, stage=f"revision{rd}", response=txt, parsed=p2, problem=prob2, latency_s=round(lat, 1))) + "\n"); log.flush()
            if not p2: print(f"   revision {rd}: unusable reply ({prob2}); keeping previous policy"); hist.append(hist[-1]); msgs.pop(); continue
            cur, curtheta = p2, to_theta(p2); z = L.evaluate(curtheta, D["test"]); hist.append(z)
            msgs.append({"role": "assistant", "content": txt[:1500]}); print(f"   revision {rd}: {policy_text(p2)} -> held-out success {z:.2f} | {why2[:140]}")
        rec = dict(sample=s, initial_policy=policy, final_policy=cur, reason=why, zero_shot_success=z0, success_after_feedback=hist)
        if not a.no_train:
            t = time.time(); iters = 2 if a.quick else 20
            res = L.cem("nemotron", s, D["train"], D["test"], iters=iters, init=(th0, L.STD_LLM)); rec["learning_curve"] = res["curve"]; rec["trained_theta"] = res["theta"]
            print("   learning from Nemotron's first policy (episodes, success):", [(u, round(x, 2)) for u, x in res["curve"]], f"[{time.time()-t:.0f}s]")
            if a.transfer:
                import Robot.berry_simulation.transfer as T
                o = T.loop(res["theta"], "nemotron", s); rec["transfer"] = {k: o[k] for k in ("sim_A", "real_before", "real_after", "fit")}
                print(f"   sim->reality: sim {o['sim_A']:.2f} | reality before {o['real_before']:.2f} -> after patch+retrain {o['real_after']:.2f}")
        results.append(rec); json.dump(results, open("results/nemotron_berry_results.json", "w"), indent=1)
    ok = [r for r in results if not r.get("failed")]
    if ok:
        z = [r["zero_shot_success"] for r in ok]; f = [r["success_after_feedback"][-1] for r in ok]
        print("\n==== SUMMARY ====")
        print(f"usable policies: {len(ok)}/{len(results)} | zero-shot held-out success: mean {np.mean(z):.2f} (min {min(z):.2f}, max {max(z):.2f}) | after {a.rounds} feedback rounds: mean {np.mean(f):.2f}")
        if not a.no_train: print("after learning (final point of each curve):", [round(r["learning_curve"][-1][1], 2) for r in ok])
        print("REFERENCE (from the earlier experiments, same simulator): policy written by Claude: zero-shot 0.70, ~0.86 after 4,800 episodes; no prior: 0.01 start, ~0.67 final, 40% of runs stuck at 0.51; 'skip everything' = 0.51.")
    print("requests used:", bud.n)
