"""
berry_sim.py -- berry picking with REAL contact physics (MuJoCo) and REAL rendered camera images.

Scene: one berry hangs from a stem (a rigid-length link to an anchor above/behind it). A leaf (static box) sits beside it and blocks the
approach from its side. A parallel-jaw hand picks it. Ripe berries detach at low stem tension, unripe ones hold on hard; twisting lowers the
break tension; squeezing too little lets the berry slip, too hard crushes it.

Perception (FIXED, hand-written, NOT learned): colour segmentation of two rendered images (front + top) -> perceived ripeness, berry position,
leaf bearing, stem bearing. Noise comes from lighting, shading and leaf occlusion, plus small calibration noise.

Hidden from the policy: true ripeness, stem break tension, true positions, physics constants.
"""
import os, sys
if sys.platform.startswith("linux"): os.environ.setdefault("MUJOCO_GL", "egl")   # rendering is only needed to BUILD datasets; shipped .pkl files avoid it
import mujoco, numpy as np, math, random

R_BERRY = 0.014; GOOD_R = 0.55
B0 = np.array([0.0, 0.0, 0.30]); STEM_L = 0.05; STEM_EL = math.radians(60)
LEAF_DIST = 0.034; CRUSH_N = 8.0
PULL_DIST = 0.07; MAXHALF = 0.04
F_HAND = 5000.0

XML = f"""<mujoco><option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 -9.81"/>
<default><geom solref="0.004 1"/></default>
<visual><headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6" specular="0 0 0"/><global offwidth="256" offheight="256"/></visual>
<asset><material name="matte" specular="0" shininess="0"/></asset>
<worldbody>
 <light name="key" pos="0.3 -0.4 0.9" dir="-0.3 0.4 -0.9" diffuse="0.5 0.5 0.5" specular="0 0 0"/>
 <geom name="floor" type="plane" size="1 1 .1" pos="0 0 0" rgba="0.55 0.55 0.55 1" contype="0" conaffinity="0"/>
 <geom name="wall" type="box" size="0.4 0.01 0.4" pos="0 0.40 0.3" rgba="0.8 0.78 0.7 1" contype="0" conaffinity="0"/>
 <geom name="leaf" type="box" size="0.003 0.028 0.03" pos="0.034 0 0.3" rgba="0.1 0.45 0.1 1" friction="0.3 0.005 0.0001"/>
 <geom name="stemvis" type="capsule" size="0.0015 0.025" pos="0 0 0.33" rgba="0.4 0.25 0.1 1" contype="0" conaffinity="0"/>
 <body name="anchor" pos="0 0 0.35"/>
 <body name="berry" pos="0 0 0.3"><freejoint name="bj"/>
   <geom name="bg" type="sphere" size="{R_BERRY}" mass="0.012" friction="1.0 0.005 0.0001" rgba="0.8 0.1 0.1 1" priority="1"/></body>
 <body name="hand" pos="0 0 0" gravcomp="1">
   <joint name="hx" type="slide" axis="1 0 0"/><joint name="hy" type="slide" axis="0 1 0"/><joint name="hz" type="slide" axis="0 0 1"/>
   <joint name="hyaw" type="hinge" axis="0 0 1" damping="0.01"/>
   <geom name="palm" type="box" size="0.005 0.03 0.012" pos="-0.03 0 0" mass="0.2" friction="0.3 0.005 0.0001" rgba="0.3 0.3 0.8 1"/>
   <body name="fl" pos="0 {MAXHALF+0.002} 0" gravcomp="1"><joint name="flj" type="slide" axis="0 -1 0" range="0 {MAXHALF}" damping="5"/>
     <geom name="flg" type="box" size="0.012 0.002 0.008" mass="0.03" friction="1.0 0.005 0.0001" rgba="0.3 0.3 0.8 1"/></body>
   <body name="fr" pos="0 {-(MAXHALF+0.002)} 0" gravcomp="1"><joint name="frj" type="slide" axis="0 1 0" range="0 {MAXHALF}" damping="5"/>
     <geom name="frg" type="box" size="0.012 0.002 0.008" mass="0.03" friction="1.0 0.005 0.0001" rgba="0.3 0.3 0.8 1"/></body>
 </body>
 <camera name="front" pos="0 -0.38 0.30" xyaxes="1 0 0 0 0 1" fovy="30"/>
 <camera name="top" pos="0 0 0.62" xyaxes="1 0 0 0 1 0" fovy="40"/>
</worldbody>
<equality><weld name="stem" body1="anchor" body2="berry" solref="0.004 1"/></equality>
<actuator>
 <position name="ax" joint="hx" kp="{F_HAND}" kv="200" forcerange="-200 200"/>
 <position name="ay" joint="hy" kp="{F_HAND}" kv="200" forcerange="-200 200"/>
 <position name="az" joint="hz" kp="{F_HAND}" kv="200" forcerange="-200 200"/>
 <position name="ayaw" joint="hyaw" kp="40" kv="1.5" forcerange="-5 5"/>
 <motor name="afl" joint="flj" gear="1" ctrlrange="-100 100"/>
 <motor name="afr" joint="frj" gear="1" ctrlrange="-100 100"/>
</actuator></mujoco>"""

_M = mujoco.MjModel.from_xml_string(XML)
_D = mujoco.MjData(_M)
_ID = lambda t, n: mujoco.mj_name2id(_M, t, n)
GID = {n: _ID(mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("floor", "wall", "leaf", "stemvis", "bg", "palm", "flg", "frg")}
BID = {n: _ID(mujoco.mjtObj.mjOBJ_BODY, n) for n in ("anchor", "berry", "hand")}
JQ = {n: _M.jnt_qposadr[_ID(mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("hx", "hy", "hz", "hyaw", "flj", "frj", "bj")}
JV = {n: _M.jnt_dofadr[_ID(mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("flj", "frj", "hyaw")}
AID = {n: _ID(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("ax", "ay", "az", "ayaw", "afl", "afr")}
EQ = _ID(mujoco.mjtObj.mjOBJ_EQUALITY, "stem")
LIGHT = _ID(mujoco.mjtObj.mjOBJ_LIGHT, "key")
_REND = {}

def _quat_z_to(v):
    v = v / np.linalg.norm(v); z = np.array([0, 0, 1.0]); ax = np.cross(z, v); s = np.linalg.norm(ax)
    if s < 1e-9: return np.array([1.0, 0, 0, 0]) if v[2] > 0 else np.array([0, 1.0, 0, 0])
    ax /= s; ang = math.acos(np.clip(v[2], -1, 1)); return np.array([math.cos(ang / 2), *(ax * math.sin(ang / 2))])

def make_episode(rng):
    ripe = rng.random() < 0.55
    r = rng.uniform(0.65, 1.0) if ripe else rng.uniform(0.0, 0.45)
    return dict(r=r, ripe=bool(r >= GOOD_R), psi=rng.uniform(-math.pi, math.pi), phi=rng.uniform(-math.pi, math.pi),
                light=rng.uniform(0.5, 1.2), jit=[rng.uniform(-.012, .012), rng.uniform(-.012, .012), rng.uniform(-.008, .008)])

def apply_episode(ep, phys=None):
    """write the episode into the (shared) model/data and reset. phys = optional dict of 'reality' modifications."""
    phys = phys or {}
    m, d = _M, _D
    mujoco.mj_resetData(m, d)
    B = B0 + np.array(ep["jit"]); u = np.array([math.cos(ep["psi"]) * math.cos(STEM_EL), math.sin(ep["psi"]) * math.cos(STEM_EL), math.sin(STEM_EL)])
    A = B + STEM_L * u
    m.body_pos[BID["anchor"]] = A
    m.eq_data[EQ, :3] = 0.0; m.eq_data[EQ, 3:6] = B - A; m.eq_data[EQ, 6:10] = [1, 0, 0, 0]; m.eq_data[EQ, 10] = 1.0   # weld: berry pose relative to anchor frame
    m.geom_pos[GID["stemvis"]] = (A + B) / 2; m.geom_quat[GID["stemvis"]] = _quat_z_to(u); m.geom_size[GID["stemvis"]][1] = STEM_L / 2
    lp = B + LEAF_DIST * np.array([math.cos(ep["phi"]), math.sin(ep["phi"]), 0])
    m.geom_pos[GID["leaf"]] = lp; m.geom_quat[GID["leaf"]] = np.array([math.cos(ep["phi"] / 2), 0, 0, math.sin(ep["phi"] / 2)])
    r = ep["r"]; pale = np.array([0.80, 0.88, 0.40]); red = np.array([0.80, 0.08, 0.08])
    m.geom_rgba[GID["bg"]] = [*(pale * (1 - r) + red * r), 1]
    I = ep["light"]; m.light_diffuse[LIGHT] = [0.5 * I] * 3; m.vis.headlight.diffuse[:] = [0.6 * I] * 3; m.vis.headlight.ambient[:] = [0.35 * I] * 3
    m.geom_friction[GID["flg"], 0] = m.geom_friction[GID["frg"], 0] = 1.0 * phys.get("pad_friction", 1.0)
    m.geom_friction[GID["bg"], 0] = 1.0 * phys.get("pad_friction", 1.0)
    d.qpos[JQ["bj"]:JQ["bj"] + 3] = B; d.qpos[JQ["bj"] + 3:JQ["bj"] + 7] = [1, 0, 0, 0]
    d.eq_active[EQ] = 1
    mujoco.mj_forward(m, d)
    return dict(B=B, u=u, A=A)

def render_views(size_front=128, size_top=160):
    out = {}
    for cam, sz in (("front", size_front), ("top", size_top)):
        if (cam, sz) not in _REND: _REND[(cam, sz)] = mujoco.Renderer(_M, sz, sz)
        rd = _REND[(cam, sz)]; rd.update_scene(_D, cam); out[cam] = rd.render().astype(float)
    return out

def perceive(ep, geo, noise_scale=1.0, rng=None):
    """FIXED colour-segmentation perception from rendered pixels -> dict(r_hat, B_hat, phi_hat, psi_hat, ok)."""
    rng = rng or random.Random(0)
    im = render_views(); T, Fv = im["top"], im["front"]
    def masks(I):
        R, G, Bc = I[..., 0], I[..., 1], I[..., 2]
        red = (R > 1.6 * G) & (R > 1.6 * Bc) & (R > 60)
        pale = (G > 100) & (R > 0.75 * G) & (Bc < 0.6 * G) & (G > 1.1 * Bc) & (R > 1.35 * Bc)
        brown = (R > 1.25 * G) & (G > 1.1 * Bc) & (R < 150) & ~red
        leaf = (G > 1.5 * R) & (G > 1.5 * Bc) & (G < 150)
        return red | pale, red, brown, leaf
    bm_t, red_t, brown_t, leaf_t = masks(T); bm_f, red_f, _, _ = masks(Fv)
    if bm_t.sum() < 6 or bm_f.sum() < 6: return dict(ok=False, r_hat=0.5, B_hat=B0.copy(), phi_hat=0.0, psi_hat=0.0)
    mpp_t = 2 * (0.62 - B0[2]) * math.tan(math.radians(20)) / T.shape[0]
    mpp_f = 2 * 0.38 * math.tan(math.radians(15)) / Fv.shape[0]
    def cen(mask): ys, xs = np.nonzero(mask); return xs.mean(), ys.mean()
    cx, cy = cen(bm_t); fx, fy = cen(bm_f)
    nT = T.shape[0] / 2; nF = Fv.shape[0] / 2
    x = (cx - nT) * mpp_t; y = -(cy - nT) * mpp_t; z = B0[2] - (fy - nF) * mpp_f
    def redness(I, m_): R, G = I[..., 0][m_], I[..., 1][m_]; return float(np.mean((R - G) / (R + G + 1.0)))
    rho = 0.5 * (redness(T, bm_t) + redness(Fv, bm_f))
    r_hat = float(np.clip((rho + 0.10) / 0.80, 0, 1))
    def bearing(mask):
        if mask.sum() < 3: return None
        ys, xs = np.nonzero(mask); return math.atan2(-(ys.mean() - cy), xs.mean() - cx)
    phi_hat = bearing(leaf_t); psi_hat = bearing(brown_t)
    ok = phi_hat is not None and psi_hat is not None
    ns = noise_scale
    return dict(ok=ok, r_hat=r_hat + rng.gauss(0, 0.03 * ns), B_hat=np.array([x, y, z]) + np.array([rng.gauss(0, .002 * ns), rng.gauss(0, .002 * ns), rng.gauss(0, .002 * ns)]),
                phi_hat=(phi_hat if phi_hat is not None else 0.0) + rng.gauss(0, 0.05 * ns), psi_hat=(psi_hat if psi_hat is not None else 0.0) + rng.gauss(0, 0.05 * ns))

def make_dataset(n, seed, phys=None, noise_scale=1.0):
    rng = random.Random(seed); out = []
    for _ in range(n):
        ep = make_episode(rng); geo = apply_episode(ep, phys); ep["obs"] = perceive(ep, geo, noise_scale, rng); out.append(ep)
    return out

def pick_attempt(ep, act, phys=None):
    """act: dict(beta, gap, force, pull_az, pull_el, speed, twist). Runs the physical pick. Returns dict(outcome=..., detail)."""
    m, d = _M, _D; phys = phys or {}
    geo = apply_episode(ep, phys); dt = m.opt.timestep
    Bh = ep["obs"]["B_hat"]; beta = act["beta"]; yaw0 = beta + math.pi
    F = float(np.clip(act["force"], 0.3, 30)); gap = float(np.clip(act["gap"], 0.032, 0.080)); v = float(np.clip(act["speed"], 0.03, 0.15))
    twist = float(np.clip(act["twist"], 0, 1.6)); tb_scale = phys.get("break_scale", 1.0)
    qopen = (2 * MAXHALF - gap) / 2
    d.qpos[JQ["flj"]] = qopen; d.qpos[JQ["frj"]] = qopen
    start = Bh + 0.10 * np.array([math.cos(beta), math.sin(beta), 0.0])
    d.qpos[JQ["hx"]:JQ["hz"] + 1] = start; d.qpos[JQ["hyaw"]] = yaw0
    mujoco.mj_forward(m, d)
    info = dict(crushed=False, detached=False, max_force=0.0, tension_max=0.0, lateral_max=0.0)
    ripe_r = ep["r"]; Tb0 = (22.0 if ripe_r < GOOD_R else 2.0 + 6.0 * (1 - ripe_r)) * tb_scale
    u = geo["u"]
    def ctrl_hand(p, yaw): d.ctrl[AID["ax"]], d.ctrl[AID["ay"]], d.ctrl[AID["az"]] = p; d.ctrl[AID["ayaw"]] = yaw
    def hold_open():
        for j, a in (("flj", "afl"), ("frj", "afr")): d.ctrl[AID[a]] = float(np.clip(300 * (qopen - d.qpos[JQ[j]]), -30, 30))
    def sense():
        fmax = 0.0; T = 0.0
        for i in range(d.ncon):
            c = d.contact[i]; g = {c.geom1, c.geom2}
            if GID["bg"] in g and (GID["flg"] in g or GID["frg"] in g):
                f6 = np.zeros(6); mujoco.mj_contactForce(m, d, i, f6); fmax = max(fmax, abs(f6[0]))
        for k in range(d.nefc):
            pass
        T = (0.0, 0.0)
        if d.eq_active[EQ]:
            rows = [k for k in range(d.nefc) if d.efc_type[k] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[k] == EQ]
            if len(rows) >= 3:
                Fv = np.array([d.efc_force[k] for k in rows[:3]]); ax = float(abs(Fv @ u)); T = (ax, float(np.linalg.norm(Fv - (Fv @ u) * u)))
        return fmax, T
    def twist_now():
        rv = np.zeros(3); mujoco.mju_quat2Vel(rv, d.qpos[JQ["bj"] + 3:JQ["bj"] + 7].copy(), 1.0); return abs(float(rv @ u))
    # A: approach
    n = int(0.10 / 0.25 / dt)
    for k in range(n + 100):
        p = start + (Bh - start) * min(1.0, k / n); ctrl_hand(p, yaw0); hold_open(); mujoco.mj_step(m, d)
    # B: close
    m.dof_damping[JV["flj"]] = m.dof_damping[JV["frj"]] = F / 0.05
    stall = 0
    for k in range(int(1.0 / dt)):
        ctrl_hand(Bh, yaw0); d.ctrl[AID["afl"]] = F; d.ctrl[AID["afr"]] = F; mujoco.mj_step(m, d)
        if abs(d.qvel[JV["flj"]]) < 2e-3 and abs(d.qvel[JV["frj"]]) < 2e-3: stall += 1
        else: stall = 0
        if stall > 80: break
        if k % 5 == 0:
            f, _T = sense(); info["max_force"] = max(info["max_force"], f)
    closed_gap = 2 * MAXHALF - (d.qpos[JQ["flj"]] + d.qpos[JQ["frj"]])
    # C: pull (+ twist) along direction d_pull
    el = math.radians(act["pull_el"]); az = act["pull_az"]
    dirv = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
    t_pull = PULL_DIST / v; npull = int(t_pull / dt) + 1
    for k in range(npull + int(0.3 / dt)):
        s = min(1.0, k / npull); ctrl_hand(Bh + dirv * PULL_DIST * s, yaw0 + twist * s)
        d.ctrl[AID["afl"]] = F; d.ctrl[AID["afr"]] = F; mujoco.mj_step(m, d)
        if k % 4 == 0:
            f, (Tax, Tlat) = sense(); info["max_force"] = max(info["max_force"], f)
            info["tension_max"] = max(info["tension_max"], Tax); info["lateral_max"] = max(info["lateral_max"], Tlat)
            if d.eq_active[EQ]:
                tw = min(1.0, abs(d.qpos[JQ['hyaw']] - yaw0) / 1.2) if info['max_force'] > 0.3 else 0.0; thr = Tb0 * (1 - 0.45 * tw)
                if Tax > thr or Tlat > 4.0 * thr: d.eq_active[EQ] = 0; info["detached"] = True
    info["crushed"] = info["max_force"] > CRUSH_N
    hand_pos = np.array([d.qpos[JQ["hx"]], d.qpos[JQ["hy"]], d.qpos[JQ["hz"]]]); bpos = d.xpos[BID["berry"]].copy()
    held = np.linalg.norm(bpos - hand_pos) < 0.025
    info["held"] = bool(held); info["closed_gap_cm"] = closed_gap * 100
    if info["crushed"]: out = "crushed"
    elif not info["detached"]: out = "slipped_or_stuck"
    elif not held: out = "dropped"
    else: out = "picked"
    info["outcome"] = out
    return info

def episode_success(ep, act, phys=None):
    """sparse task reward: ripe -> must be picked cleanly; unripe -> must be left alone. act['pick']=False means skip."""
    if not act.get("pick", True): return (not ep["ripe"]), dict(outcome="skipped")
    info = pick_attempt(ep, act, phys)
    return (ep["ripe"] and info["outcome"] == "picked"), info
