"""transfer.py -- the sim -> 'reality' loop.  REALITY differs from the simulator in ways the learner is not told:
   pads 25% slipperier, stems 30% tougher, perception noise doubled.  Loop: train in sim -> deploy in reality -> observe outcomes on 60 real berries ->
   fit sim friction/stem-strength to match those outcomes (system identification) -> retrain in the PATCHED sim -> redeploy.
Real episodes used: 60 (diagnosis) per cycle.  Compared with training DIRECTLY in reality."""
import Robot.berry_simulation.berry_sim as B
import Robot.berry_simulation.learn as L
import math, json, pickle, os, sys, time, collections, numpy as np
REAL = dict(pad_friction=0.75, break_scale=1.3); NOISE = 2.0
if not os.path.exists("real.pkl"):
    pickle.dump(dict(diag=B.make_dataset(60, 303, REAL, NOISE), test=B.make_dataset(200, 404, REAL, NOISE), train=B.make_dataset(400, 505, REAL, NOISE)), open("real.pkl", "wb"))
R = pickle.load(open("real.pkl", "rb")); A = pickle.load(open("data.pkl", "rb"))
CATS = ["skipped", "picked", "crushed", "slipped_or_stuck", "dropped"]
def hist(th, eps, phys):
    c = collections.Counter(B.episode_success(e, L.to_act(th, e["obs"]), phys)[1]["outcome"] for e in eps); return np.array([c[k] for k in CATS], float) / len(eps)
def sysid(th, diag, real_hist):
    best = None
    for fr in (0.6, 0.7, 0.8, 0.9, 1.0):
        for bs in (1.0, 1.2, 1.4, 1.6):
            d = np.abs(hist(th, diag, dict(pad_friction=fr, break_scale=bs)) - real_hist).sum()
            if best is None or d < best[0]: best = (d, fr, bs)
    return best
def loop(th0, tag, seed):
    th0 = np.array(th0, float)
    out = dict(tag=tag, seed=seed); out["sim_A"] = L.evaluate(th0, A["test"]); out["real_before"] = L.evaluate(th0, R["test"], REAL)
    out["real_hist_before"] = dict(zip(CATS, hist(th0, R["test"], REAL).round(3).tolist()))
    rh = hist(th0, R["diag"], REAL); d, fr, bs = sysid(th0, R["diag"], rh); out["fit"] = dict(friction=fr, break_scale=bs, hist_gap=round(float(d), 3))
    patched = dict(pad_friction=fr, break_scale=bs)
    res = L.cem("warm", seed, A["train"], R["test"], iters=6, phys=patched, init=(np.array(th0), L.STD_LLM), test_phys=REAL)
    out["real_after"] = res["curve"][-1][1]; out["real_curve_during_retrain"] = res["curve"]; out["theta_after"] = res["theta"]
    out["real_episodes_used"] = len(R["diag"]); return out
if __name__ == "__main__":
    rows = [json.loads(l) for l in open("results_learn.jsonl")]
    pick = [r for r in rows if (r["name"] == "llm_prior" and r["seed"] in (0, 1, 2, 3)) or (r["name"] in ("scratch", "scratch_wide") and r["curve"][-1][1] > 0.8)]
    for r in pick:
        t = time.time(); o = loop(r["theta"], r["name"], r["seed"]); o["secs"] = round(time.time() - t)
        open("transfer_results.jsonl", "a").write(json.dumps(o) + "\n")
        print(f"{o['tag']:12s} seed {o['seed']}: sim {o['sim_A']:.2f} | reality before {o['real_before']:.2f} -> after patch+retrain {o['real_after']:.2f} | fit {o['fit']} | {o['secs']}s", flush=True)
