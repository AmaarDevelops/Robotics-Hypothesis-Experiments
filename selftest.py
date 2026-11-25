"""Run this FIRST. Replays 13 reference grasps computed in the original sandbox and checks your machine reproduces them."""
import json, sys, grasp_sim as S, mujoco
ref = json.load(open("reference_outputs.json")); objs = S.make_objects(0); bad = 0
print(f"mujoco here: {mujoco.__version__} | mujoco in reference: {ref['mujoco']}")
for c in ref["cases"]:
    r = S.run_grasp(objs[c["obj"]], dict(zip(("approach_height","finger_opening","grip_force","lift_speed"), c["params"]))); t = r["tel"]
    ok_succ = r["success"] == c["success"]
    ok_num = abs(t["gap_at_close_cm"] - c["gap_cm"]) < 0.3 and abs(t["force_L"] - c["force_L"]) < max(0.05 * c["force_L"], 0.5)
    ok = ok_succ and ok_num; bad += (not ok)
    print(("PASS " if ok else "FAIL "), f"obj{c['obj']} params={c['params']} success={r['success']} (ref {c['success']}) gap={t['gap_at_close_cm']:.2f} (ref {c['gap_cm']:.2f})")
print("\nSELF-TEST:", "ALL MATCH -- your physics reproduces the reference." if bad == 0 else f"{bad} MISMATCHES -- see README (usually a MuJoCo version difference).")
sys.exit(1 if bad else 0)
