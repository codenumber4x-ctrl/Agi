"""Two virtual rats living in a realistic home cage (MuJoCo).

Usage
-----
    MUJOCO_GL=egl python simulate.py --seconds 40 --out rat_cage.mp4
    python simulate.py --viewer          # interactive MuJoCo viewer

The script builds the cage scene (build_scene.py), attaches two copies of the
DeepMind virtual-rodent body, runs a CPG/IK locomotion controller plus a
behaviour state machine for each rat (rat_controller.py) and renders a video
with an automatic "camera director" (wide shots, follow shots, close-ups of
the water bottle and the pellet bowl).
"""

import argparse
import os
import time

import mujoco
import numpy as np

import build_scene as scene
from rat_controller import Behaviour, GaitLibrary, RatController

CTRL_DT = 0.01          # 100 Hz controller


class World:
    POSES = {"rat1": ([0.12, -0.10, 0.0], 2.4), "rat2": ([-0.02, 0.02, 0.0], -0.6)}

    def __init__(self, seed=0, assist=1.0):
        self.spec, self.m = scene.compile_scene(seed, self.POSES)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        gait = GaitLibrary()
        self.rats = [RatController(self.m, "rat1/", gait, seed=seed + 1, assist=assist),
                     RatController(self.m, "rat2/", gait, seed=seed + 2, assist=assist)]
        self.beh = [Behaviour(self.rats[0], self.rats[1], seed=seed + 1, start="explore"),
                    Behaviour(self.rats[1], self.rats[0], seed=seed + 2, start="explore")]
        # make the opening minute interesting: one rat is thirsty, one is hungry
        self.beh[0].thirst, self.beh[0].hunger = 0.9, 0.5
        self.beh[1].thirst, self.beh[1].hunger = 0.3, 0.85
        self.nsub = int(round(CTRL_DT / self.m.opt.timestep))
        # let the rats settle on the bedding
        for _ in range(int(0.5 / CTRL_DT)):
            for r in self.rats:
                r.step(self.d, CTRL_DT, dict(speed=0.0, heading=r.pose(self.d)[1]))
            for _ in range(self.nsub):
                mujoco.mj_step(self.m, self.d)

    def step(self):
        for r, b in zip(self.rats, self.beh):
            r.step(self.d, CTRL_DT, b.update(self.d, CTRL_DT))
        for _ in range(self.nsub):
            mujoco.mj_step(self.m, self.d)


class Director:
    """Picks camera shots based on what the rats are doing."""

    def __init__(self, world, shot_len=5.0):
        self.w = world
        self.shot_len = shot_len
        self.t_shot = 1e9
        self.shot = None
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.look = np.array([0.0, 0.0, 0.12])
        self.k = 0
        self.history = []

    def _choose(self):
        states = [b.state for b in self.w.beh]
        self.k += 1
        last = self.shot[0] if self.shot else None
        if "drink" in states and last != "water":
            return ("water", states.index("drink"))
        if "eat" in states and last != "food":
            return ("food", states.index("eat"))
        if last in (None, "wide2") or (last == "follow" and self.k % 3 == 0):
            return ("wide", None)
        if last == "wide":
            return ("follow", self.k % 2)
        if last in ("water", "food"):
            return ("wide2", None)
        return ("follow", self.k % 2)

    def update(self, t, dt):
        self.t_shot += dt
        if self.t_shot > self.shot_len:
            self.shot = self._choose()
            self.t_shot = 0.0
            self.history.append((round(t, 1), self.shot[0]))
            snap = True
        else:
            snap = False
        kind, idx = self.shot
        c = self.cam
        s = self.t_shot
        if kind == "wide":
            target = np.array([0.0, 0.0, 0.14])
            c.distance, c.azimuth, c.elevation = 1.25 - 0.02 * s, 62 + 3.0 * s, -30
        elif kind == "wide2":
            target = np.array([0.0, 0.0, 0.12])
            c.distance, c.azimuth, c.elevation = 1.10, 125 - 3.0 * s, -38
        elif kind == "follow":
            # camera inside the cage, on the cage-centre side of the rat, looking outwards
            p = self.w.d.xpos[self.w.rats[idx].torso]
            target = np.array([p[0], p[1], 0.05])
            out = np.degrees(np.arctan2(p[1] + 0.04, p[0]))
            c.distance, c.elevation = 0.30, -30
            c.azimuth = out + 25 * np.sin(0.25 * s)
        elif kind == "water":
            # from inside the cage, looking at the sipper tube on the right wall
            target = np.array([scene.HX - 0.06, scene.WATER_SPOUT[1], 0.07])
            c.distance, c.azimuth, c.elevation = 0.30, 38 - 2.0 * s, -18
        elif kind == "food":
            # from inside the cage, looking at the pellet bowl in the front-left corner
            target = np.array([scene.FOOD_BOWL[0] + 0.03, scene.FOOD_BOWL[1] + 0.02, 0.04])
            c.distance, c.azimuth, c.elevation = 0.31, 218 + 2.0 * s, -26
        if snap:
            self.look = target.copy()
        else:
            self.look += (1 - np.exp(-dt / 0.35)) * (target - self.look)
        c.lookat[:] = self.look
        return snap


def simulate(seconds, fps, seed):
    """Run the physics + controllers and record one state per video frame."""
    w = World(seed)
    director = Director(w)
    steps_per_frame = int(round(1.0 / fps / CTRL_DT))
    nframes = int(seconds * fps)
    rec = dict(qpos=[], cam=[], cut=[], shot=[], states=[])
    t0 = time.time()
    for f in range(nframes):
        for _ in range(steps_per_frame):
            w.step()
        t = f / fps
        cut = director.update(t, 1.0 / fps)
        c = director.cam
        rec["qpos"].append(w.d.qpos.copy())
        rec["cam"].append([*c.lookat, c.distance, c.azimuth, c.elevation])
        rec["cut"].append(cut)
        rec["shot"].append(director.shot[0])
        rec["states"].append(tuple(b.state for b in w.beh))
        if f % (fps * 5) == 0:
            print(f"  sim t={t:5.1f}s  states={rec['states'][-1]}  wall={time.time() - t0:5.1f}s",
                  flush=True)
    rec = {k: np.array(v) if k in ("qpos", "cam") else v for k, v in rec.items()}
    return w, rec


# ---------------------------------------------------------------- rendering
_R = {}


def _render_init(seed, width, height, qpos, cams):
    import warnings
    warnings.simplefilter("ignore")
    _, m = scene.compile_scene(seed, World.POSES)
    _R.update(m=m, d=mujoco.MjData(m), r=mujoco.Renderer(m, height, width),
              qpos=qpos, cams=cams)
    opt = mujoco.MjvOption()
    opt.geomgroup[2] = 0          # hide collision capsules
    opt.geomgroup[3] = 0          # hide invisible collision walls
    _R["opt"] = opt
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    _R["cam"] = cam


def _render_frame(i):
    m, d, cam = _R["m"], _R["d"], _R["cam"]
    d.qpos[:] = _R["qpos"][i]
    mujoco.mj_forward(m, d)
    c = _R["cams"][i]
    cam.lookat[:] = c[:3]
    cam.distance, cam.azimuth, cam.elevation = c[3], c[4], c[5]
    _R["r"].update_scene(d, cam, _R["opt"])
    return _R["r"].render()


def render_video(seconds, out, width, height, fps, seed, shot_dir, workers=4):
    import multiprocessing as mp
    import imageio.v2 as imageio
    w, rec = simulate(seconds, fps, seed)
    n = len(rec["qpos"])
    t0 = time.time()
    ctx = mp.get_context("fork")
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8,
                                macro_block_size=8, pixelformat="yuv420p")
    with ctx.Pool(workers, initializer=_render_init,
                  initargs=(seed, width, height, rec["qpos"], rec["cam"])) as pool:
        for i, img in enumerate(pool.imap(_render_frame, range(n), chunksize=4)):
            writer.append_data(img)
            if shot_dir and i > 0 and (rec["cut"][i] or i == n - 1):
                imageio.imwrite(os.path.join(
                    shot_dir, f"video_t{i / fps:05.1f}_{rec['shot'][i - 1]}.png"), prev)
            prev = img
            if i % (fps * 5) == 0:
                print(f"  render frame {i}/{n}  wall={time.time() - t0:5.1f}s", flush=True)
    writer.close()
    print(f"wrote {out}  ({n} frames, render {time.time() - t0:.0f}s)")
    print("behaviour log:")
    last = None
    for i, st in enumerate(rec["states"]):
        if st != last:
            print(f"  {i / fps:6.1f}s  rat1={st[0]:9s} rat2={st[1]}")
            last = st
    return w, rec


def run_viewer(seed):
    import mujoco.viewer
    w = World(seed)
    with mujoco.viewer.launch_passive(w.m, w.d) as v:
        while v.is_running():
            t = time.time()
            w.step()
            v.sync()
            time.sleep(max(0.0, CTRL_DT - (time.time() - t)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--out", default="rat_cage.mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", default="screenshots", help="dir for still frames at each cut")
    ap.add_argument("--workers", type=int, default=4, help="parallel render processes")
    ap.add_argument("--viewer", action="store_true")
    a = ap.parse_args()
    if a.viewer:
        run_viewer(a.seed)
    else:
        if a.shots:
            os.makedirs(a.shots, exist_ok=True)
        render_video(a.seconds, a.out, a.width, a.height, a.fps, a.seed, a.shots, a.workers)
