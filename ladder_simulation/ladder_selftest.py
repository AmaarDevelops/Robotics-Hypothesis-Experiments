"""Run FIRST: checks robosuite works and a scripted closed-loop controller can lift the cube through the same chunk interface."""
import numpy as np, Robot.ladder_simulation.ladder as Lm, sys
env = Lm.make("L2"); ok = 0
for seed in (100, 101, 102):
    ep = Lm.Ep(env, seed, [])
    def goto(z, grip, tol, n=40):
        for _ in range(n):
            d = np.array(z) - ep.obs["robot0_eef_pos"]
            if np.linalg.norm(d) < tol: return
            v = np.clip(d / 0.0105, -1, 1) * 0.8; ep.run([[v[0], v[1], v[2], 0, 0, 0, grip, 2]])
    c = ep.obs["cube_pos"].copy(); goto(c + [0, 0, 0.12], -1, 0.01); goto(c + [0, 0, 0.005], -1, 0.006); ep.run([[0, 0, 0, 0, 0, 0, 1, 12]]); goto(c + [0, 0, 0.15], 1, 0.01)
    ok += ep.succ_step is not None; print(f"seed {seed}: {'SUCCESS' if ep.succ_step else 'fail'} in {ep.t} steps")
print("SELF-TEST:", "PASS" if ok == 3 else f"FAIL ({ok}/3)"); sys.exit(0 if ok == 3 else 1)
