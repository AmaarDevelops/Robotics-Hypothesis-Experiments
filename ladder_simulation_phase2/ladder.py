"""ladder.py -- an LLM controls a Franka Panda in robosuite's standard Lift task by producing ACTION CHUNKS directly.
Run with the robosuite venv python.  Episodes are REPLAYED deterministically from (seed, action history) on every call.
  init LEVEL N SEED0 | obs | act '<json: {"0": [[a..., grip, repeat], ...], ...}>' | score
LEVELS: L2 = end-effector delta pose  [dx,dy,dz,drx,dry,drz]  (+grip)  input in [-1,1]: 1.0 = 5 cm (or 0.5 rad) per 20 Hz step
        L1 = joint position deltas    [7 joints]               (+grip)  input in [-1,1]: 1.0 = 0.05 rad per step
        L0 = joint torques (gravity-compensated; 0 holds the arm) [7 joints] (+grip)  input in [-1,1]
grip: -1 = open, +1 = close.  Each chunk entry = [action numbers..., grip, repeat_count].   Success = robosuite's own Lift success flag."""
import sys, json, os, math, numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config, load_part_controller_config
ST = "ladder_state.json"; HORIZON = 300; XY_RANGE = 0.06; TORQUE_SCALE = 2.0
def make(level, render=False, task="Lift"):
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    if level in ("L1", "L0"):
        part = load_part_controller_config(default_controller="JOINT_POSITION" if level == "L1" else "JOINT_TORQUE")
        keep = cfg["body_parts"]["right"].get("gripper")
        if keep is not None: part["gripper"] = keep
        if level == "L0": part["output_max"] = [TORQUE_SCALE] * 7; part["output_min"] = [-TORQUE_SCALE] * 7   # default +-0.1 N*m cannot move the arm
        cfg["body_parts"]["right"] = part
    return suite.make(task, robots="Panda", controller_configs=cfg, has_renderer=False, has_offscreen_renderer=render, use_camera_obs=False,
                      control_freq=20, horizon=10**6, ignore_done=True, reward_shaping=False, initialization_noise=None)
def n_table_contacts(env):
    m = env.sim.model; d = env.sim.data; k = 0
    for i in range(d.ncon):
        c = d.contact[i]; names = []
        for g in (c.geom1, c.geom2):
            names.append(((m.geom_id2name(g) or ""), (m.body_id2name(m.geom_bodyid[g]) or "")))
        rob = [n[1].startswith(("robot0", "gripper0")) for n in names]; tab = ["table" in n[0] for n in names]
        if (rob[0] and tab[1]) or (rob[1] and tab[0]): k += 1
    return k
def yaw(q): x, y, z, w = q; return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
class Ep:
    """one deterministic replayed episode"""
    def __init__(self, env, seed, chunks, task="Lift"):
        np.random.seed(seed); self.env = env; self.task = task; self.obs = env.reset(); rng = np.random.default_rng(seed)
        def place(cube, x, y, th):
            z0 = float(env.table_offset[2] + cube.size[2]) + 0.0005
            env.sim.data.set_joint_qpos(cube.joints[0], np.concatenate([[x, y, z0], [math.cos(th / 2), 0, 0, math.sin(th / 2)]]))
        if task == "Stack":                     # OUR seeded randomisation: two cubes at least 11 cm apart, random yaws
            while True:
                ax, ay, bx, by = [rng.uniform(-XY_RANGE, XY_RANGE) for _ in range(4)]
                if math.hypot(ax - bx, ay - by) > 0.11: break
            place(env.cubeA, ax, ay, rng.uniform(0, 2 * math.pi)); place(env.cubeB, bx, by, rng.uniform(0, 2 * math.pi))
        else:
            th = rng.uniform(0, 2 * math.pi); x = rng.uniform(-XY_RANGE, XY_RANGE); y = rng.uniform(-XY_RANGE, XY_RANGE); place(env.cube, x, y, th)
        env.sim.forward(); self.obs = env._get_observations(force_update=True)
        self.key = "cube_pos" if task == "Lift" else "cubeA_pos"; self.start = self.obs[self.key].copy()
        self.frame_cb = None; self.t = 0; self.grip_prev = -1; self.attempts = 0; self.table = 0; self.shift = 0.0; self.maxh = 0.0; self.succ_step = None
        for ch in chunks: self.run(ch)
    def run(self, chunk):
        for entry in chunk:
            a = np.array(entry[:-1], float); n = int(entry[-1])
            for _ in range(max(1, min(n, 30))):
                if self.succ_step is not None or self.t >= HORIZON: return
                a = np.clip(a, -1, 1); self.obs, r, d, i = self.env.step(a); self.t += 1
                g = a[-1]
                if g > 0 and self.grip_prev <= 0: self.attempts += 1
                self.grip_prev = g; self.table += n_table_contacts(self.env) > 0
                c = self.obs[self.key]; h = c[2] - self.start[2]; self.maxh = max(self.maxh, h)
                if h < 0.01: self.shift = max(self.shift, float(np.linalg.norm(c[:2] - self.start[:2])))
                if self.env._check_success(): self.succ_step = self.t
                if self.frame_cb: self.frame_cb(self)
    def text(self, level):
        o = self.obs; e = o["robot0_eef_pos"]; gq = o["robot0_gripper_qpos"]; tail = f"grasp attempts {self.attempts}, table-contact steps {self.table}"
        if self.task == "Stack":
            a = o["cubeA_pos"]; b = o["cubeB_pos"]
            s = (f"step {self.t}/{HORIZON} | eef ({e[0]:.3f},{e[1]:.3f},{e[2]:.3f}) | gripper opening {abs(gq[0]-gq[1]):.3f} m | cubeA RED, to pick ({a[0]:.3f},{a[1]:.3f},{a[2]:.3f}) yaw {yaw(o['cubeA_quat'])%90:.0f}deg"
                 f" | cubeB GREEN, target ({b[0]:.3f},{b[1]:.3f},{b[2]:.3f}) | eef->cubeA ({a[0]-e[0]:+.3f},{a[1]-e[1]:+.3f},{a[2]-e[2]:+.3f}) | cubeA->cubeB ({b[0]-a[0]:+.3f},{b[1]-a[1]:+.3f},{b[2]-a[2]:+.3f}) | {tail}")
        else:
            c = o["cube_pos"]
            s = (f"step {self.t}/{HORIZON} | eef ({e[0]:.3f},{e[1]:.3f},{e[2]:.3f}) | gripper opening {abs(gq[0]-gq[1]):.3f} m | cube ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) yaw {yaw(o['cube_quat'])%90:.0f}deg"
                 f" | eef->cube ({c[0]-e[0]:+.3f},{c[1]-e[1]:+.3f},{c[2]-e[2]:+.3f}) | cube lifted {c[2]-self.start[2]:+.3f} m | {tail}")
        if level in ("L1", "L0"): s += f" | joints {np.round(o['robot0_joint_pos'],2).tolist()} vel {np.round(o['robot0_joint_vel'],2).tolist()}"
        return s
def load(): return json.load(open(ST))
def tag(ep): return "SUCCESS" if ep.succ_step else ("TIME UP" if ep.t >= HORIZON else "running")
if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "init": json.dump(dict(level=sys.argv[2], task=(sys.argv[5] if len(sys.argv) > 5 else "Lift"), eps=[dict(seed=int(sys.argv[4]) + k, chunks=[]) for k in range(int(sys.argv[3]))]), open(ST, "w")); print("ok")
    else:
        st = load(); env = make(st["level"], task=st.get("task", "Lift")); new = json.loads(sys.argv[2]) if cmd == "act" else {}; S = []
        for k, e in enumerate(st["eps"]):            # each episode is replayed ALONE (never interleaved) on the shared simulator
            ep = Ep(env, e["seed"], e["chunks"], st.get("task", "Lift"))
            if str(k) in new and ep.succ_step is None and ep.t < HORIZON: ep.run(new[str(k)]); e["chunks"].append(new[str(k)])
            S.append(ep)
            if cmd in ("obs", "act"): print(f"ep{k} [{tag(ep)}] " + ep.text(st["level"]))
        if cmd == "act": json.dump(st, open(ST, "w"))
        if cmd == "score":
            print(f"level {st['level']}: success {sum(e.succ_step is not None for e in S)}/{len(S)} | grasp attempts {[e.attempts for e in S]} | table-contact steps {[e.table for e in S]} | steps used {[e.t for e in S]} | cube pushed (max shift m) {[round(e.shift,3) for e in S]}")
