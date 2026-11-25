"""learn.py -- does an LLM-written prior reduce the SIMULATED EXPERIENCE needed to learn berry picking?
Policy (19 numbers, theta) maps the PERCEIVED observation to an action. CEM (a simple policy-search RL method) improves theta from the sparse
episode reward (ripe: picked cleanly; unripe: left alone). Learners differ ONLY in the initial mean/std of theta.
Budget counted in physical episodes (one berry presented = 1 episode)."""
import Robot.berry_simulation.berry_sim as B
import math, json, random, sys, time, pickle, os
import numpy as np
NAMES = ["w_r", "b_pick"] + [f"Wa{i}{j}" for i in range(2) for j in range(3)] + ["gap", "lnF"] + [f"Wp{i}{j}" for i in range(2) for j in range(3)] + ["pull_el", "speed", "twist"]
LO = np.array([-20, -20] + [-3] * 6 + [0.034, math.log(0.3)] + [-3] * 6 + [-90, 0.03, 0.0]); HI = np.array([20, 20] + [3] * 6 + [0.075, math.log(25)] + [3] * 6 + [60, 0.15, 1.6])
def vec(d): return np.array([d["w_r"], d["b_pick"], *d["Wa"], d["gap"], d["lnF"], *d["Wp"], d["pull_el"], d["speed"], d["twist"]], float)
def to_act(th, obs):
    f = np.array([math.cos(obs["phi_hat"]), math.sin(obs["phi_hat"]), 1.0]); g = np.array([math.cos(obs["psi_hat"]), math.sin(obs["psi_hat"]), 1.0])
    Wa = th[2:8].reshape(2, 3); Wp = th[10:16].reshape(2, 3)
    return dict(pick=bool(th[0] * obs["r_hat"] + th[1] > 0), beta=math.atan2((Wa @ f)[1], (Wa @ f)[0]), gap=th[8], force=math.exp(th[9]),
                pull_az=math.atan2((Wp @ g)[1], (Wp @ g)[0]), pull_el=th[16], speed=th[17], twist=th[18])
def evaluate(th, eps, phys=None):
    return float(np.mean([B.episode_success(e, to_act(th, e["obs"]), phys)[0] for e in eps]))
# ---- initialisations -------------------------------------------------------------------------------------------------------------------
PRIOR_LLM = dict(   # written BEFORE any learning run, from the task description only (what an LLM knows about picking berries)
    w_r=10, b_pick=-6,                                   # pick only if perceived ripeness > 0.6: ripe = red
    Wa=[-1, 0, 0, 0, -1, 0],                             # approach from the side opposite the leaf
    gap=0.045, lnF=math.log(4.0),                        # berries bruise easily: a few newtons, not more
    Wp=[-1, 0, 0, 0, -1, 0],                             # pull away from where the stem attaches
    pull_el=-45, speed=0.06, twist=1.0)                  # down-and-away, slowly, with a twist (standard technique)
STD_LLM = np.array([2, 2] + [0.3] * 6 + [0.005, 0.4] + [0.3] * 6 + [15, 0.015, 0.3])
PRIOR_WRONG = dict(w_r=-10, b_pick=6, Wa=[1, 0, 0, 0, 1, 0], gap=0.045, lnF=math.log(14.0), Wp=[1, 0, 0, 0, 1, 0], pull_el=40, speed=0.12, twist=0.0)  # control: confidently wrong
SCRATCH = dict(w_r=0, b_pick=1, Wa=[0, 0, 1, 0, 0, 0], gap=0.06, lnF=math.log(2.0), Wp=[0, 0, 1, 0, 0, 0], pull_el=0, speed=0.09, twist=0.5)
STD_SCRATCH = np.array([5, 3] + [0.8] * 6 + [0.012, 1.0] + [0.8] * 6 + [40, 0.03, 0.5])
INITS = dict(llm_prior=(vec(PRIOR_LLM), STD_LLM), wrong_prior=(vec(PRIOR_WRONG), STD_LLM), scratch=(vec(SCRATCH), STD_SCRATCH))
HYP = {"scratch_wide": dict(base="scratch", floor=0.25, pop=48), "llm_prior_wide": dict(base="llm_prior", floor=0.25, pop=48, sdm=2.5)}      # fairer exploration for the no-prior learner
def fitness(th, eps, phys, mode):
    """training reward. v1 = task success (skipping an unripe berry is free credit); v2 = skip 0, clean pick +1, failed ripe pick -0.3, unripe pick -1."""
    if mode == "v1": return evaluate(th, eps, phys)
    tot = 0.0
    for e in eps:
        a = to_act(th, e["obs"])
        if not a["pick"]: continue
        ok, _ = B.episode_success(e, a, phys); tot += 1.0 if ok else (-0.3 if e["ripe"] else -1.0)
    return tot / len(eps)
def cem(name, seed, train, test, iters=20, pop=24, batch=10, phys=None, init=None, log=None, test_phys=None):
    mode = "v2" if name.endswith("@v2") else "v1"; name = name.replace("@v2", "")
    h = HYP.get(name, {}); pop = h.get("pop", pop); floor = h.get("floor", 0.05)
    rng = np.random.default_rng(seed); r = random.Random(seed)
    mu, sd0 = (init if init is not None else INITS[h.get("base", name)]); mu = mu.copy(); sd0 = sd0 * h.get('sdm', 1.0); sd = sd0.copy(); used = 0; curve = []
    tp = test_phys if test_phys is not None else phys
    curve.append((0, evaluate(mu, test, tp)))
    for it in range(1, iters + 1):
        bt = [train[r.randrange(len(train))] for _ in range(batch)]
        cand = np.clip(rng.normal(mu, sd, size=(pop, len(mu))), LO, HI)
        sc = np.array([fitness(c, bt, phys, mode) for c in cand]); used += pop * batch
        el = cand[np.argsort(-sc)[: pop // 4]]
        mu = el.mean(0); sd = np.maximum(0.7 * sd + 0.3 * el.std(0), floor * sd0)
        if it % 2 == 0 or it == iters: curve.append((used, evaluate(mu, test, tp)))
    return dict(name=name + ('@v2' if mode == 'v2' else ''), seed=seed, curve=curve, theta=mu.tolist())
if __name__ == "__main__":
    if not os.path.exists("data.pkl"):
        t = time.time(); pickle.dump(dict(train=B.make_dataset(400, 101), test=B.make_dataset(200, 202)), open("data.pkl", "wb")); print("dataset built in %.0fs" % (time.time() - t), flush=True)
    D = pickle.load(open("data.pkl", "rb")); seeds = [int(x) for x in sys.argv[1].split(',')] if ',' in sys.argv[1] else range(int(sys.argv[1])); names = sys.argv[2].split(",")
    for s in seeds:
        for n in names:
            t = time.time(); res = cem(n, s, D["train"], D["test"]); res["secs"] = round(time.time() - t)
            open("results_learn.jsonl", "a").write(json.dumps(res) + "\n"); print(n, s, [(u, round(x, 2)) for u, x in res["curve"]], f"{res['secs']}s", flush=True)
