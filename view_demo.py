"""Optional GUI: watch one grasp in MuJoCo's viewer. Physics identical to headless runs.  macOS: run with `mjpython view_demo.py`."""
import sys, grasp_sim as S
oid = int(sys.argv[1]) if len(sys.argv) > 1 else 7
o = S.make_objects(0)[oid]; print(S.describe(o))
r = S.run_grasp(o, dict(approach_height=0.04, finger_opening=0.08, grip_force=45, lift_speed=0.08), show=True, slowmo=2.0)
print(r["feedback"])
