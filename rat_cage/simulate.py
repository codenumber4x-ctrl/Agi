"""Run the two-rat cage simulation and render a video.

    python simulate.py --seconds 40 --out rat_cage.mp4

Use ``--viewer`` to open the interactive MuJoCo viewer instead (needs a
display).
"""
import argparse
import math
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

import cage
import rat_controller as rc

HERE = os.path.dirname(os.path.abspath(__file__))
RATS = ("rat1", "rat2")


def build():
    spec = cage.build_spec()
    pellet_names = [b.name for b in spec.bodies if b.name.startswith("pellet")]
    model = spec.compile()
    return spec, model, pellet_names


class Sim:
    def __init__(self, seed=1):
        self.spec, self.m, pellet_names = build()
        self.d = mujoco.MjData(self.m)
        rng = np.random.default_rng(seed)
        self.ik = rc.LegIK()
        self.rats = [rc.Rat(self.m, f"{n}/", self.ik, np.random.default_rng(seed * 10 + i))
                     for i, n in enumerate(RATS)]
        self.pellets = rc.Pellets(self.m, pellet_names)
        self.beh = [rc.Behaviour(r, [o for o in self.rats if o is not r], self.pellets, self.m)
                    for r in self.rats]
        mujoco.mj_forward(self.m, self.d)
        for r in self.rats:
            r.heading = r.yaw(self.d)
        # stagger initial behaviour
        self.beh[0].set_mode("idle", 0.8)
        self.beh[1].set_mode("idle", 1.6)
        # scripted opening so both feeding stations are visited early on
        self.beh[0].plan = ["explore", "drink", "rear", "eat"]
        self.beh[1].plan = ["eat", "explore", "groom", "drink"]
        self.t = 0.0
        self.rng = rng
        self._spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state = np.zeros(mujoco.mj_stateSize(self.m, self._spec))
        self.recoveries = 0
        # dofs of pellets + bowl (light free bodies)
        self.small_dofs = np.concatenate([np.arange(self.m.jnt_dofadr[j], self.m.jnt_dofadr[j] + 6)
                                          for j in range(self.m.njnt)
                                          if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                                          and not self.m.joint(j).name.endswith("root")])

    def step(self):
        m, d = self.m, self.d
        dt = m.opt.timestep
        for r, b in zip(self.rats, self.beh):
            b.update(d, self.t, dt)
            legs = r.gait_offsets(dt)
            r.apply_pose(d, self.t, dt, legs)
            r.assist(d, dt)
        self.pellets.carry(d)
        # stability guard: MuJoCo resets mjData if accelerations blow up
        # (e.g. a 1.4 g pellet squeezed between paw and bowl). Roll back one
        # step, calm the small free bodies down and continue instead.
        mujoco.mj_getState(m, d, self._state, self._spec)
        nbad = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
        mujoco.mj_step(m, d)
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number != nbad:
            mujoco.mj_setState(m, d, self._state, self._spec)
            d.qvel[self.small_dofs] = 0.0
            d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = nbad
            mujoco.mj_step(m, d)
            self.recoveries += 1
        self.t += dt


class Director:
    """Simple camera director: wide shots, close-ups on drinking / eating."""

    def __init__(self, sim):
        self.sim = sim
        self.cam = mujoco.MjvCamera()
        self.shot = "main"
        self.shot_t = 0.0
        self.free_cam = mujoco.MjvCamera()
        self.free_cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def choose(self, dt):
        self.shot_t += dt
        modes = [r.mode for r in self.sim.rats]
        want = self.shot
        if self.shot_t > 4.0:
            if "drink" in modes:
                want = "water_close"
            elif "eat" in modes:
                want = "food_close"
            else:
                want = self.sim.rng.choice(["main", "orbit", "follow", "front_low"])
            if want != self.shot:
                self.shot, self.shot_t = want, 0.0
        return self.shot

    def camera(self, dt):
        shot = self.choose(dt)
        m, d = self.sim.m, self.sim.d
        if shot == "orbit":
            c = self.free_cam
            c.lookat[:] = [0, 0, 0.12]
            c.distance = 1.35
            c.azimuth = -110 + 12 * self.sim.t
            c.elevation = -32
            return c
        if shot == "follow":
            r = self.sim.rats[int(self.sim.t // 8) % 2]
            c = self.free_cam
            target = r.pos(d).copy()
            target[2] = 0.06
            c.lookat[:] = 0.9 * c.lookat + 0.1 * target if self.shot_t > 0.05 else target
            c.distance = 0.62
            c.azimuth = -95 + 20 * math.sin(0.2 * self.sim.t)
            c.elevation = -30
            return c
        c = self.cam
        c.type = mujoco.mjtCamera.mjCAMERA_FIXED
        c.fixedcamid = m.camera(shot).id
        return c


def render_video(seconds, out, width, height, fps, seed, shots_dir=None):
    sim = Sim(seed)
    m, d = sim.m, sim.d
    renderer = mujoco.Renderer(m, height, width)
    opt = mujoco.MjvOption()
    opt.geomgroup[2] = opt.geomgroup[3] = 0
    director = Director(sim)
    steps_per_frame = int(round(1.0 / (fps * m.opt.timestep)))
    n_frames = int(seconds * fps)
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=8,
                                ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "medium"])
    t0 = time.time()
    shot_times = {2.0, 5.0, 8.5, 12.0, 15.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0}
    for f in range(n_frames):
        for _ in range(steps_per_frame):
            sim.step()
        cam = director.camera(1.0 / fps)
        renderer.update_scene(d, camera=cam, scene_option=opt)
        frame = renderer.render()
        writer.append_data(frame)
        if shots_dir and any(abs(sim.t - s) < 0.5 / fps for s in shot_times):
            p = os.path.join(shots_dir, f"sim_t{sim.t:04.1f}s_{director.shot}.png")
            imageio.imwrite(p, frame)
        if f % (fps * 5) == 0:
            modes = ", ".join(f"{r.p[:-1]}={r.mode}" for r in sim.rats)
            print(f"t={sim.t:5.1f}s frame {f}/{n_frames} [{modes}] recoveries={sim.recoveries} {time.time() - t0:.0f}s elapsed", flush=True)
    writer.close()
    renderer.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "rat_cage.mp4"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shots", default=None, help="directory for periodic screenshots")
    ap.add_argument("--viewer", action="store_true")
    a = ap.parse_args()
    if a.viewer:
        import mujoco.viewer
        sim = Sim(a.seed)
        with mujoco.viewer.launch_passive(sim.m, sim.d) as v:
            while v.is_running():
                t = time.time()
                for _ in range(8):
                    sim.step()
                v.sync()
                time.sleep(max(0, 8 * sim.m.opt.timestep - (time.time() - t)))
        return
    if a.shots:
        os.makedirs(a.shots, exist_ok=True)
    print(render_video(a.seconds, a.out, a.width, a.height, a.fps, a.seed, a.shots))


if __name__ == "__main__":
    main()
