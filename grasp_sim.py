"""
grasp_sim.py -- MuJoCo parallel-jaw grasp with CONTINUOUS skill parameters.

The planner (an LLM, a random searcher, ...) outputs 4 numbers:
    approach_height  z of finger-pad centre when closing (m).   pad half-height = 2 cm, so z<0.02 hits the table
    finger_opening   gap between finger inner faces before descent (m)
    grip_force       per-finger squeeze force limit (N)
    lift_speed       peak lift speed (m/s); acceleration = 10*speed (m/s^2)
Everything else (contacts, friction cones, inertia, slip) is simulated by MuJoCo.

HIDDEN from the planner: true mass, friction, exact size, crush limit.
VISIBLE to the planner: a text description (material, rounded size, fragility cue).
Feedback after a trial is physical telemetry (like sensors), never the fix.
"""
import mujoco, numpy as np, random, json, os, time

G = 9.81
FH, FT, FX = 0.02, 0.006, 0.012       # finger pad half-height / half-thickness / half-width
MAXHALF = 0.075                       # fully open: gap = 0.15 m
Z_START, Z_LIFT = 0.20, 0.25
BOUNDS = dict(approach_height=(0.0, 0.20), finger_opening=(0.01, 0.15),
              grip_force=(0.5, 80.0), lift_speed=(0.02, 0.5))

# nominal object templates: mass kg, friction mu (surface), radius, height, crush limit N (None = rigid)
TEMPLATES = [
    dict(name="steel soup can",        m=0.40, mu=0.35, r=0.033, h=0.10, crush=None,
         text="hollow steel can with food inside, smooth painted metal"),
    dict(name="rubber puck",           m=0.17, mu=1.20, r=0.040, h=0.040, crush=None,
         text="solid rubber cylinder, grippy surface"),
    dict(name="cardboard tube",        m=0.04, mu=0.55, r=0.025, h=0.12, crush=6.0,
         text="empty thin-walled cardboard tube, fragile (dents easily)"),
    dict(name="thin glass tumbler",    m=0.25, mu=0.25, r=0.035, h=0.11, crush=15.0,
         text="thin-walled glass tumbler, smooth and fragile"),
    dict(name="water-filled bottle",   m=0.55, mu=0.40, r=0.034, h=0.12, crush=35.0,
         text="plastic bottle full of water, slightly flexible walls"),
    dict(name="porcelain teacup",      m=0.12, mu=0.30, r=0.040, h=0.060, crush=12.0,
         text="thin porcelain teacup, glazed and fragile"),
    dict(name="foam cylinder",         m=0.02, mu=0.90, r=0.040, h=0.080, crush=4.0,
         text="soft compressible foam cylinder"),
    dict(name="solid iron weight",     m=1.50, mu=0.30, r=0.030, h=0.080, crush=None,
         text="small solid iron cylinder, smooth machined surface"),
]


def make_objects(seed):
    """Hidden truth = template * random jitter (so the text description is informative but not exact)."""
    r = random.Random(seed)
    out = []
    for i, t in enumerate(TEMPLATES):
        o = dict(id=i, name=t["name"], m=t["m"] * r.uniform(0.8, 1.2), mu=t["mu"] * r.uniform(0.75, 1.25),
                 r=t["r"] * r.uniform(0.92, 1.08), h=t["h"] * r.uniform(0.92, 1.08),
                 crush=None if t["crush"] is None else t["crush"] * r.uniform(0.8, 1.25), text=t["text"])
        out.append(o)
    return out


def describe(o):
    """What the planner is allowed to see."""
    d = round(2 * o["r"] * 100)       # whole cm
    h = round(o["h"] * 100)
    return f"[{o['id']}] {o['name']}: {o['text']}. About {d} cm diameter, {h} cm tall, standing upright on a table."


def _xml(o, fmax):
    r, h, m, mu = o["r"], o["h"], o["m"], o["mu"]
    def finger(name, y, ax):
        return f"""<body name="{name}" pos="0 {y} 0" gravcomp="1"><joint name="{name}j" type="slide" axis="{ax}" range="0 {MAXHALF}" damping="5"/>
          <geom name="{name}g" type="box" size="{FX} {FT} {FH}" mass="0.05" friction="1.0 0.005 0.0001"/></body>"""
    return f"""<mujoco><option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/><default><geom solref="0.005 1"/></default>
<worldbody>
 <geom name="table" type="plane" size="1 1 0.1" friction="0.8 0.005 0.0001"/>
 <body name="obj" pos="0 0 {h/2+1e-4}"><freejoint/>
   <geom name="objg" type="cylinder" size="{r} {h/2}" mass="{m}" friction="{mu} 0.005 0.0001" priority="1"/></body>
 <body name="hand" pos="0 0 {Z_START}" gravcomp="1">
   <joint name="hx" type="slide" axis="1 0 0"/><joint name="hy" type="slide" axis="0 1 0"/><joint name="hz" type="slide" axis="0 0 1"/>
   <geom type="box" size="0.03 0.03 0.01" pos="0 0 0.05" mass="0.5" contype="0" conaffinity="0"/>
   {finger("fl", MAXHALF+FT, "0 -1 0")}{finger("fr", -(MAXHALF+FT), "0 1 0")}
 </body></worldbody>
<actuator>
 <position name="ax" joint="hx" kp="6000" kv="250" forcerange="-300 300"/>
 <position name="ay" joint="hy" kp="6000" kv="250" forcerange="-300 300"/>
 <position name="az" joint="hz" kp="6000" kv="250" forcerange="-300 300"/>
 <motor name="afl" joint="fll" gear="1" ctrlrange="-100 100"/>
 <motor name="afr" joint="frr" gear="1" ctrlrange="-100 100"/>
</actuator></mujoco>""".replace('joint="fll"', 'joint="flj"').replace('joint="frr"', 'joint="frj"')


def run_grasp(o, p, show=False, slowmo=1.0):
    ah = float(np.clip(p["approach_height"], *BOUNDS["approach_height"]))
    gap = float(np.clip(p["finger_opening"], *BOUNDS["finger_opening"]))
    F = float(np.clip(p["grip_force"], *BOUNDS["grip_force"]))
    v = float(np.clip(p["lift_speed"], *BOUNDS["lift_speed"]))
    m = mujoco.MjModel.from_xml_string(_xml(o, F)); d = mujoco.MjData(m)
    gid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("fl" + "g", "frg", "objg", "table")}
    a = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("ax", "ay", "az", "afl", "afr")}
    jq = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("hx", "hy", "hz", "flj", "frj")}
    jv = {n: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("flj", "frj")}
    qopen = (2 * MAXHALF - gap) / 2
    d.qpos[jq["flj"]] = qopen; d.qpos[jq["frj"]] = qopen
    d.qpos[jq["hz"]] = 0.0     # hand slide joint offset from Z_START
    mujoco.mj_forward(m, d)
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj")
    obj_z = lambda: d.xpos[obj_bid][2]
    hand_z_world = lambda: Z_START + d.qpos[jq["hz"]]
    tel = dict(table_contact=False, knocked_cm=0.0, gap_at_close_cm=None, force_L=0.0, force_R=0.0,
               slip_cm=0.0, peak_obj_z_cm=0.0, crushed=False, fingers_closed_on_nothing=False)
    obj_xy0 = d.xpos[obj_bid][:2].copy()
    dt = m.opt.timestep
    viewer = None
    if show:                      # optional GUI: needs a display; on macOS run with `mjpython`
        from mujoco import viewer as _mjv
        viewer = _mjv.launch_passive(m, d)
    _n = [0]
    def step():
        mujoco.mj_step(m, d)
        if viewer is not None:
            _n[0] += 1
            if _n[0] % 10 == 0:
                viewer.sync(); time.sleep(dt * 10 * slowmo)

    def hand_to(z_world):
        d.ctrl[a["ax"]] = 0.0; d.ctrl[a["ay"]] = 0.0; d.ctrl[a["az"]] = z_world - Z_START

    def finger_vel(vl):
        d.ctrl[a["afl"]] = vl; d.ctrl[a["afr"]] = vl

    def contacts():
        fl = fr = 0.0; tc = False
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]; g = {c.geom1, c.geom2}
            if gid["table"] in g and (gid["flg"] in g if "flg" in gid else (gid["fl" + "g"] in g or gid["frg"] in g)):
                tc = True
            if gid["objg"] in g and (gid["fl" + "g"] in g or gid["frg"] in g):
                mujoco.mj_contactForce(m, d, i, f6)
                if gid["fl" + "g"] in g: fl += abs(f6[0])
                else: fr += abs(f6[0])
        return fl, fr, tc

    # --- phase 1: hold fingers at opening, descend to approach height (hard table stop at z=FH)
    m.dof_damping[jv["flj"]] = m.dof_damping[jv["frj"]] = 5.0
    zt = ah
    if zt < FH - 1e-6:
        tel["table_contact"] = True; zt = FH
    def hold_open():
        d.ctrl[a["afl"]] = float(np.clip(500 * (qopen - d.qpos[jq["flj"]]), -50, 50))
        d.ctrl[a["afr"]] = float(np.clip(500 * (qopen - d.qpos[jq["frj"]]), -50, 50))
    n_desc = int(abs(Z_START - zt) / 0.15 / dt) + 1
    for k in range(n_desc + 100):
        z = Z_START + (zt - Z_START) * min(1.0, k / max(1, n_desc))
        hand_to(z); hold_open(); step()
        if k % 10 == 0:
            tel["table_contact"] |= contacts()[2]
    tel["knocked_cm"] = float(np.linalg.norm(d.xpos[obj_bid][:2] - obj_xy0) * 100)
    # --- phase 2: close: constant squeeze force F, closing speed ~ F/damping = 5 cm/s
    m.dof_damping[jv["flj"]] = m.dof_damping[jv["frj"]] = F / 0.05
    def squeeze():
        d.ctrl[a["afl"]] = F; d.ctrl[a["afr"]] = F
    stall = 0
    for k in range(int(2.0 / dt)):
        squeeze(); step()
        if abs(d.qvel[jv["flj"]]) < 2e-3 and abs(d.qvel[jv["frj"]]) < 2e-3: stall += 1
        else: stall = 0
        if stall > 100: break
    for k in range(int(0.3 / dt)):
        squeeze(); step()
    fl, fr, tc = contacts(); tel["table_contact"] |= tc
    gapm = 2 * MAXHALF - (d.qpos[jq["flj"]] + d.qpos[jq["frj"]])
    tel["gap_at_close_cm"] = float(gapm * 100); tel["force_L"], tel["force_R"] = float(fl), float(fr)
    tel["fingers_closed_on_nothing"] = bool(gapm < 0.012 and fl + fr < 0.1)
    lim = o["crush"]
    if lim is not None and max(fl, fr) > lim:
        tel["crushed"] = True
    # --- phase 3: lift (trapezoid velocity profile, accel = 10*v)
    rel0 = obj_z() - hand_z_world()
    acc = 10.0 * v; dist = Z_LIFT - zt
    ts = 0.0; zc = zt; vc = 0.0; max_rel_drop = 0.0
    dropped_at = None
    n = 0
    while n * dt < 12.0:
        remaining = Z_LIFT - zc
        if remaining <= 1e-4 and vc <= 1e-3: break
        vstop = np.sqrt(2 * acc * max(remaining, 0))
        vc = min(v, vc + acc * dt, vstop) if remaining > 0 else 0.0
        zc += vc * dt
        hand_to(zc); squeeze()
        step(); n += 1
        if n % 10 == 0:
            rel = obj_z() - hand_z_world()
            max_rel_drop = max(max_rel_drop, rel0 - rel)
            tel["peak_obj_z_cm"] = max(tel["peak_obj_z_cm"], obj_z() * 100)
            if dropped_at is None and (rel0 - rel) > 0.04: dropped_at = n * dt
    for k in range(int(0.5 / dt)):
        hand_to(Z_LIFT); squeeze(); step()
    ez, hz_ = obj_z(), hand_z_world()
    tel["slip_cm"] = float(max_rel_drop * 100); tel["final_obj_z_cm"] = float(ez * 100)
    in_hand = ez > 0.15 and abs(ez - hz_) < 0.08
    success = bool(in_hand and not tel["crushed"])
    # --- feedback text (sensor-like symptoms only)
    msgs = []
    if tel["table_contact"]: msgs.append("fingertips touched the table during descent")
    if tel["knocked_cm"] > 0.5: msgs.append(f"object was pushed {tel['knocked_cm']:.1f} cm during descent (fingers hit it)")
    if tel["fingers_closed_on_nothing"]: msgs.append("fingers closed fully on nothing (no object between them)")
    else: msgs.append(f"fingers stopped at a gap of {tel['gap_at_close_cm']:.1f} cm; measured squeeze L={fl:.1f} N R={fr:.1f} N")
    if tel["crushed"]: msgs.append("OBJECT DAMAGED: squeeze force exceeded what the object tolerates")
    elif success: msgs.append(f"lift OK: object held at {ez*100:.0f} cm after hold (slid {tel['slip_cm']:.1f} cm in grip)")
    else:
        if dropped_at is not None: msgs.append(f"object slipped out during lift (at t={dropped_at:.2f} s of lift); it slid {tel['slip_cm']:.1f} cm relative to the fingers")
        msgs.append(f"object final height {ez*100:.0f} cm (needed >15 cm in hand)")
    if viewer is not None:
        time.sleep(1.5); viewer.close()
    return dict(success=success, tel=tel, feedback="; ".join(msgs), params=dict(approach_height=ah, finger_opening=gap, grip_force=F, lift_speed=v))


if __name__ == "__main__":
    import time
    objs = make_objects(0)
    t = time.time()
    for o in objs[:3]:
        r = run_grasp(o, dict(approach_height=0.05, finger_opening=0.10, grip_force=20, lift_speed=0.1))
        print(o["name"], r["success"], r["feedback"])
    print("time/trial %.2fs" % ((time.time() - t) / 3))
