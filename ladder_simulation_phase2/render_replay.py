"""render_replay.py -- watch the robot. Replays a saved episode (results/L2_seed*.json from nemotron_ladder.py) deterministically and writes an MP4.
  python render_replay.py results/L2_seed100.json results/L2_seed102.json          -> videos/L2_seed100.mp4 ...
  python render_replay.py results/L2_seed101.json --vs other_run/L2_seed101.json    -> side-by-side comparison video
Options: --camera agentview|frontview|sideview|birdview   --size 320
Headless (no display needed). On Linux/WSL set MUJOCO_GL=egl (default here); if EGL fails try MUJOCO_GL=osmesa.
LIVE window (needs a display, e.g. Linux desktop or Windows 11 WSLg): robosuite's own viewer: Lm.make('L2') with has_renderer=True and env.render() each step."""
import os, sys, json, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, imageio
from PIL import Image, ImageDraw
import ladder as Lm
def load_chunks(path):
    d = json.load(open(path)); return d["seed"], d["chunks"], d.get("task", "Lift")
def episode_frames(env, seed, chunks, label, size, cam, task="Lift"):
    frames, meta = [], []
    ep = Lm.Ep(env, seed, [], task)
    def grab(e): frames.append(e.env.sim.render(width=size, height=size, camera_name=cam)[::-1].copy()); meta.append((e.t, e.succ_step is not None))
    grab(ep); ep.frame_cb = grab
    for ch in chunks: ep.run(ch)
    out = []
    for img, (t, ok) in zip(frames, meta):
        im = Image.fromarray(img); d = ImageDraw.Draw(im)
        d.rectangle([0, 0, size, 34], fill=(0, 0, 0)); d.text((6, 4), f"{label}", fill=(255, 255, 255)); d.text((6, 18), f"step {t}/300   seed {seed}", fill=(200, 200, 200))
        if ok: d.rectangle([0, size - 24, size, size], fill=(20, 120, 20)); d.text((6, size - 18), "SUCCESS: cube lifted", fill=(255, 255, 255))
        out.append(np.array(im))
    final = Image.fromarray(out[-1]); d = ImageDraw.Draw(final)
    if not meta[-1][1]: d.rectangle([0, size - 24, size, size], fill=(150, 30, 30)); d.text((6, size - 18), f"NOT lifted ({ep.attempts} grasp attempts, {ep.t} steps)", fill=(255, 255, 255)); out[-1] = np.array(final)
    return out, ep
def pad(frames, n, hold=20):
    return frames + [frames[-1]] * (n - len(frames) + hold)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--vs"); ap.add_argument("--label", default="model"); ap.add_argument("--vs-label", default="other"); ap.add_argument("--camera", default="agentview"); ap.add_argument("--size", type=int, default=320)
    a = ap.parse_args(); os.makedirs("videos", exist_ok=True); envs = {}
    for f in a.files:
        seed, chunks, task = load_chunks(f); env = envs.setdefault(task, Lm.make("L2", render=True, task=task)); name = os.path.splitext(os.path.basename(f))[0]
        fr, ep = episode_frames(env, seed, chunks, a.label, a.size, a.camera, task)
        if a.vs:
            s2, c2, t2 = load_chunks(a.vs); fr2, ep2 = episode_frames(env, s2, c2, a.vs_label, a.size, a.camera, t2); n = max(len(fr), len(fr2))
            fr = [np.concatenate([x, y], axis=1) for x, y in zip(pad(fr, n), pad(fr2, n))]
        else: fr = pad(fr, len(fr))
        out = f"videos/{name}{'_vs' if a.vs else ''}.mp4"; imageio.mimsave(out, fr, fps=20, macro_block_size=1); print(out, f"{len(fr)} frames | {Lm.tag(ep)} at step {ep.t}")
