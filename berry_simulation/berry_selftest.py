"""Run FIRST. Replays 40 berry picks computed in the original sandbox; checks your MuJoCo reproduces the same outcomes."""
import json, pickle, sys, numpy as np
import Robot.berry_simulation.berry_sim as B, Robot.berry_simulation.learn as L, mujoco
ref = json.load(open("reference_berry.json")); D = pickle.load(open("data.pkl", "rb"))
th = L.vec(L.PRIOR_LLM); bad = 0
print("mujoco here:", mujoco.__version__, "| reference:", ref["mujoco"])
for i, want in enumerate(ref["outcomes"]):
    got = B.episode_success(D["test"][i], L.to_act(th, D["test"][i]["obs"]))[1]["outcome"]; bad += (got != want)
    if got != want: print("MISMATCH episode", i, "got", got, "expected", want)
print("SELF-TEST:", "ALL 40 MATCH" if bad == 0 else f"{bad}/40 mismatches (small differences can come from a different MuJoCo version: pip install mujoco==4.0.1)")
sys.exit(1 if bad > 6 else 0)
