"""
run.py -- LLM-as-planner experiment (physics: MuJoCo). The LLM sees only a TEXT description + sensor-style feedback.

  python run.py              # real run, uses .env, HARD CAP of 50 API requests (persisted in results/request_count.json)
  python run.py --mock       # pipeline test with a trivial fake planner (NOT an LLM); no API calls

Phase A  zero-shot : 8 named objects, up to 2 attempts each            (<= 16 requests)
Phase B  recovery  : 6 hidden-property objects chosen so a generic 15 N grasp FAILS, up to 4 attempts each (<= 24 requests)
Each object is its own conversation (no memory across objects). Everything is logged to results/. The API key is never logged.
"""
import os, sys, re, json, time, random, argparse
import numpy as np
import grasp_sim as S

CAP = 50
K = ("approach_height", "finger_opening", "grip_force", "lift_speed")
SYSTEM = (
 "You control a parallel-jaw robot gripper in a physics simulator. Task: pick up ONE upright cylindrical object from a table and lift it 25 cm. "
 "You choose 4 numbers: approach_height_m (height of the finger-pad centre above the table, range 0-0.20; the pads are 4 cm tall, so values below 0.02 hit the table), "
 "finger_opening_m (gap between the fingers before descending, 0.01-0.15), grip_force_N (squeeze force per finger, 0.5-80), "
 "lift_speed_mps (0.02-0.5; lift acceleration = 10 x speed). You cannot see the object's true mass, friction or fragility; you only get a text description, "
 "and after each attempt you get sensor-style feedback. Reply with ONLY one JSON object, no other text: "
 '{"approach_height_m": ..., "finger_opening_m": ..., "grip_force_N": ..., "lift_speed_mps": ..., "reason": "one short sentence"}')

def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

class Budget:
    def __init__(self): os.makedirs("results", exist_ok=True); self.f = "results/request_count.json"; self.n = json.load(open(self.f)) if os.path.exists(self.f) else 0
    def spend(self):
        if self.n >= CAP: raise SystemExit(f"Request cap {CAP} reached; stopping. (Delete results/request_count.json to reset.)")
        self.n += 1; json.dump(self.n, open(self.f, "w"))

def call_llm(bud, messages, mock_obj=None, history=None):
    if mock_obj is not None:  # MOCK: trivial heuristic, NOT an LLM
        F = 15.0 if not history else (history[-1]["F"] * (0.5 if history[-1]["crushed"] else 1.9))
        d = round(2 * mock_obj["r"] * 100) / 100
        return json.dumps(dict(approach_height_m=round(mock_obj["h"] * 50) / 100, finger_opening_m=d + 0.02, grip_force_N=F, lift_speed_mps=0.05, reason="mock")), 0.0
    import requests
    for attempt in range(3):
        bud.spend(); t0 = time.time()
        try:
            r = requests.post(os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions", timeout=180,
                headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"], "Content-Type": "application/json"},
                json={"model": os.environ["LLM_MODEL"], "messages": messages, "temperature": 0.2, "max_tokens": 2000})
        except Exception as e:
            print("   network error:", type(e).__name__); time.sleep(3); continue
        if r.status_code == 429: print("   rate-limited, waiting 20 s"); time.sleep(20); continue
        if r.status_code != 200: print("   HTTP", r.status_code, r.text[:200].replace(os.environ["LLM_API_KEY"], "***")); time.sleep(3); continue
        msg = r.json()["choices"][0]["message"]; txt = msg.get("content") or msg.get("reasoning") or ""
        return txt, time.time() - t0
    return "", 0.0

def parse(txt):
    for m in reversed(re.findall(r"\{[^{}]*\}", txt or "", flags=re.S)):
        try:
            j = json.loads(m)
            return dict(approach_height=float(j["approach_height_m"]), finger_opening=float(j["finger_opening_m"]),
                        grip_force=float(j["grip_force_N"]), lift_speed=float(j["lift_speed_mps"])), j.get("reason", "")
        except Exception: continue
    return None, ""

def solve_object(bud, o, max_att, log, mock=False):
    hist, notes = [], []
    for att in range(1, max_att + 1):
        user = S.describe(o) + "\n\n"
        if notes: user += "Previous attempts on this object:\n" + "\n".join(notes) + "\n\n"
        user += "Give your next grasp as JSON."
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        p = None; invalid = 0
        for tries in range(2):
            txt, lat = call_llm(bud, msgs, o if mock else None, hist)
            p, why = parse(txt)
            log.write(json.dumps(dict(obj=o["id"], attempt=att, prompt=user, response=txt, latency_s=round(lat, 2), parsed=p)) + "\n"); log.flush()
            if p: break
            invalid += 1; msgs.append({"role": "user", "content": "Invalid. Reply with ONLY the JSON object."})
        if p is None:
            print(f"   obj {o['id']} attempt {att}: no valid JSON (counts as failed attempt)"); hist.append(dict(F=15, crushed=False, ok=False, invalid=invalid)); notes.append(f"{att}. (invalid reply)"); continue
        r = S.run_grasp(o, p); ok = r["success"]
        print(f"   obj {o['id']} attempt {att}: F={p['grip_force']:.1f}N open={p['finger_opening']:.3f} z={p['approach_height']:.3f} v={p['lift_speed']:.2f} -> {'SUCCESS' if ok else 'FAIL'} | {r['feedback'][:110]}")
        notes.append(f"{att}. approach_height={p['approach_height']:.3f}, finger_opening={p['finger_opening']:.3f}, grip_force={p['grip_force']:.1f}, lift_speed={p['lift_speed']:.2f} -> {r['feedback']}")
        hist.append(dict(F=p["grip_force"], crushed=r["tel"]["crushed"], ok=ok, invalid=invalid, lat=lat))
        if ok: return att, hist
    return None, hist

# ---------- objects for phase B (same recipe as the sandbox run) ----------
def mystery(rng):
    return dict(id=99, name="mystery", r=rng.uniform(0.02, 0.045), h=rng.uniform(0.05, 0.12), m=float(np.exp(rng.uniform(np.log(0.1), np.log(2.0)))),
                mu=rng.uniform(0.2, 1.0), crush=None if rng.random() < 0.5 else rng.uniform(5, 40), text="")
def geo(o, F, v):
    d = round(2 * o["r"] * 100) / 100; h = round(o["h"] * 100) / 100
    return dict(approach_height=h / 2, finger_opening=d + 0.02, grip_force=F, lift_speed=v)
def solvable(o):
    return any(S.run_grasp(o, geo(o, float(F), 0.05))["success"] for F in np.geomspace(1, 80, 14))
def hard_set(seed, n=6):
    rng = random.Random(seed); out = []
    while len(out) < n:
        o = mystery(rng); o["id"] = 12 + len(out)
        if solvable(o) and not S.run_grasp(o, geo(o, 15, 0.06))["success"]: out.append(o)
    return out

def baselines_for(objs, nsamp):
    """non-LLM comparators on the SAME objects: fixed grasps, hand-coded retry rule, random search."""
    rng = random.Random(1); res = []
    for o in objs:
        fixed = {n: S.run_grasp(o, geo(o, F, 0.1))["success"] for n, F in (("geo10N", 10), ("geo20N", 20))}
        F, v, lo, hi, att = 15.0, 0.06, None, None, None
        for t in range(1, 9):
            r = S.run_grasp(o, geo(o, F, v))
            if r["success"]: att = t; break
            if r["tel"]["crushed"]: hi = F; F = (lo + hi) / 2 if lo else F / 2
            else: lo = F; F = (lo + hi) / 2 if hi else F * 1.9
            v = 0.04
        p = np.mean([S.run_grasp(o, dict(approach_height=rng.uniform(0, .2), finger_opening=rng.uniform(.01, .15), grip_force=float(np.exp(rng.uniform(np.log(.5), np.log(80)))), lift_speed=rng.uniform(.02, .5)))["success"] for _ in range(nsamp)])
        res.append(dict(id=o["id"], rule_attempts=att, p_rand_in_4=1 - (1 - p) ** 4, **fixed))
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mock", action="store_true"); ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--baseline-samples", type=int, default=100); a = ap.parse_args()
    load_env()
    if not a.mock:
        for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
            if not os.environ.get(k): sys.exit(f"Missing {k}. Copy .env.example to .env and fill it in.")
    bud = Budget(); os.makedirs("results", exist_ok=True)
    print(("MOCK planner (NOT an LLM)" if a.mock else f"Planner model: {os.environ['LLM_MODEL']}"), "| request cap", CAP, "| used so far", bud.n)
    log = open("results/llm_log.jsonl", "a")
    A = S.make_objects(a.seed); B = hard_set(a.seed)
    print("\n== Phase A: zero-shot on 8 named objects =="); resA = []
    for o in A: resA.append(solve_object(bud, o, 2, log, a.mock))
    print("\n== Phase B: recovery on 6 hidden-property objects (a generic 15 N grasp fails on these) =="); resB = []
    for o in B: resB.append(solve_object(bud, o, 4, log, a.mock))
    print("\n== Non-LLM baselines on the same objects (no API) =="); bA = baselines_for(A, a.baseline_samples); bB = baselines_for(B, a.baseline_samples)
    first = sum(1 for att, h in resA if att == 1)
    summary = dict(planner="mock" if a.mock else os.environ["LLM_MODEL"], requests_used=bud.n,
        phaseA=dict(first_try_success=f"{first}/8", eventual_success=f"{sum(att is not None for att,_ in resA)}/8",
                    geo10N_fixed=f"{sum(b['geo10N'] for b in bA)}/8", geo20N_fixed=f"{sum(b['geo20N'] for b in bA)}/8"),
        phaseB=dict(recovered=f"{sum(att is not None for att,_ in resB)}/6", attempts=[att for att, _ in resB],
                    mean_attempts_when_recovered=float(np.mean([att for att, _ in resB if att])) if any(att for att, _ in resB) else None,
                    handcoded_rule_attempts=[b["rule_attempts"] for b in bB], random_search_P_success_within_4=round(float(np.mean([b["p_rand_in_4"] for b in bB])), 2)),
        invalid_replies=sum(h.get("invalid", 0) for _, hs in resA + resB for h in hs),
        mean_latency_s=round(float(np.mean([h.get("lat", 0) for _, hs in resA + resB for h in hs if h.get("lat")] or [0])), 2))
    json.dump(summary, open("results/summary.json", "w"), indent=1); print("\n" + json.dumps(summary, indent=1))
