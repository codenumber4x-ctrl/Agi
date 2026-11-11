"""Gymnasium environment: one virtual rat learning to move in a 3x-larger cage.

Nothing about the behaviour is scripted.  There is no gait generator, no
steering logic and no assist force: the policy outputs the 38 actuator
commands of the DeepMind virtual-rodent body directly, 50 times per second,
and has to discover walking, steering and approaching the food bowl and the
water spout purely from reward.

Reward (per 20 ms control step)
-------------------------------
* moving      : +MOVE_W * forward speed of the torso (m/s), clipped to
                [-0.1, 0.3] m/s.  Moving backwards gives a small penalty.
* food        : +FOOD_REWARD (very high) the first time the rat's head is
                in the food bowl ("eating") in an episode
* water       : +WATER_REWARD (high) the first time the rat's head reaches
                the sipper tube ("drinking") in an episode
* wall impact : -IMPACT_W * (normal force above IMPACT_FREE N), capped at
                IMPACT_MAX per step, for contacts
                between the rat and the cage walls, so gently brushing a
                wall (thigmotaxis) is fine but crashing into one costs.
* effort      : a tiny -CTRL_W * |a|^2 to discourage pointless thrashing

Episodes end when the rat flips over (torso up-axis below horizontal by
more than ~70 deg) or after EPISODE_SECONDS.

Observation (all egocentric, what a rat could sense)
-----------------------------------------------------
joint angles + velocities (proprioception), actuator activations, torso
orientation w.r.t. gravity, torso linear/angular velocity (vestibular),
paw touch sensors, 9 whisker/vision rays from the head (distance to walls
and objects up to 30 cm), egocentric direction + distance to the food bowl
and to the water spout (smell / memory of the home cage), and whether it
has already eaten / drunk this episode.
"""

import os
import sys
import warnings

import gymnasium as gym
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import build_scene as scene  # noqa: E402

# ---------------------------------------------------------------- settings
CAGE_SCALE = 3.0                 # 3x the original 80 x 50 cm floor -> 2.4 x 1.5 m
CTRL_DT = 0.02                   # 50 Hz control
EPISODE_SECONDS = 30.0

MOVE_W = 2.0
FOOD_REWARD = 200.0              # "very high"
WATER_REWARD = 100.0             # "high"
IMPACT_W = 0.1
IMPACT_FREE = 3.5                # N: above ~body weight (3.3 N) counts as a hard hit
IMPACT_MAX = 1.5                # cap per step so crashing never beats flipping over
CTRL_W = 0.002

FOOD_RADIUS = 0.075              # head within the bowl rim
FOOD_HEAD_Z = 0.09
WATER_RADIUS = 0.04              # head (skull centre) at the sipper ball
N_RAYS = 9
RAY_RANGE = 0.30
P_SPAWN_NEAR = 0.3               # fraction of episodes starting 20-50 cm from food/water


def configure_cage():
    scene.set_cage_size(0.80 * CAGE_SCALE, 0.50 * CAGE_SCALE)


def build_model(visual=False):
    """Physics model (visual=False) or the same model with all the cage
    decorations for rendering (visual=True).  Both have identical qpos layout."""
    configure_cage()
    tex_ok = all(os.path.exists(os.path.join(scene.TEX_DIR, n + ".png"))
                 for n in ("bedding", "wood", "floor", "wall", "glaze", "pellet"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, m = scene.compile_scene(rat_poses={"rat1": ([0.0, 0.0, 0.0], 0.0)},
                                   regen_textures=not tex_ok, visual=visual,
                                   furniture=False, pellets=False, strengthen=False)
    return m


class RatCageEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / CTRL_DT)}

    def __init__(self, seed=None, spawn_near=P_SPAWN_NEAR, episode_seconds=EPISODE_SECONDS):
        super().__init__()
        self.m = build_model(visual=False)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.nsub = int(round(CTRL_DT / m.opt.timestep))
        self.max_steps = int(episode_seconds / CTRL_DT)
        self.spawn_near = spawn_near
        self.rng = np.random.default_rng(seed)

        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rat1/torso")
        self.skull = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rat1/skull")
        self.rat_geoms = np.array([g for g in range(m.ngeom)
                                   if m.body_rootid[m.geom_bodyid[g]] == self.torso])
        self.is_rat_geom = np.zeros(m.ngeom, bool)
        self.is_rat_geom[self.rat_geoms] = True
        wall_names = ["pan_xp", "pan_xn", "pan_yp", "pan_yn",
                      "wire_col_xp", "wire_col_xn", "wire_col_yp", "wire_col_yn"]
        self.is_wall = np.zeros(m.ngeom, bool)
        for n in wall_names:
            self.is_wall[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)] = True
        self.touch_adr = [m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"rat1/{n}")]
                          for n in ("palm_L", "palm_R", "sole_L", "sole_R")]
        # rays see the cage (groups 0 and 3), not the rat's own collision capsules (group 2)
        self.ray_groups = np.array([1, 0, 0, 1, 0, 0], np.uint8)
        self.ray_angles = np.linspace(-np.pi / 2, np.pi / 2, N_RAYS)
        self.food = scene.FOOD_BOWL.copy()
        self.water = scene.WATER_SPOUT.copy()

        mujoco.mj_resetData(m, self.d)
        mujoco.mj_forward(m, self.d)
        self.q0 = self.d.qpos.copy()
        self.torso_z0 = self.q0[2]
        self.ate = self.drank = False
        nobs = len(self._obs())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (nobs,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (m.nu,), np.float32)

    # ------------------------------------------------------------------ utils
    def _torso_frame(self):
        R = self.d.xmat[self.torso].reshape(3, 3)
        return R

    def _rays(self):
        d, m = self.d, self.m
        R = self.d.xmat[self.skull].reshape(3, 3)
        fwd = R[:, 0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd) + 1e-9
        left = np.array([-fwd[1], fwd[0], 0.0])
        origin = d.xpos[self.skull].copy()
        geomid = np.zeros(1, np.int32)
        out = np.empty(N_RAYS)
        for i, a in enumerate(self.ray_angles):
            v = np.cos(a) * fwd + np.sin(a) * left
            dist = mujoco.mj_ray(m, d, origin, v, self.ray_groups, 1, self.torso, geomid)
            out[i] = RAY_RANGE if (dist < 0 or dist > RAY_RANGE) else dist
        return out / RAY_RANGE

    def _target_obs(self, target_xy):
        R = self._torso_frame()
        head = self.d.xpos[self.skull]
        v = np.array([target_xy[0] - head[0], target_xy[1] - head[1], 0.0])
        local = R.T @ v
        dist = np.linalg.norm(v[:2])
        u = local[:2] / (np.linalg.norm(local[:2]) + 1e-9)
        return np.array([u[0], u[1], np.tanh(dist / 0.5)])

    def _obs(self):
        d = self.d
        R = self._torso_frame()
        lin = R.T @ d.cvel[self.torso][3:]
        ang = R.T @ d.cvel[self.torso][:3]
        touch = np.log1p(d.sensordata[self.touch_adr])
        obs = np.concatenate([
            d.qpos[7:],                      # 67 joint angles
            np.clip(d.qvel[6:], -50, 50) * 0.1,
            d.act,                           # 38 actuator activations
            R[2, :],                         # gravity direction in torso frame
            lin, 0.2 * ang,
            [d.qpos[2] / self.torso_z0],
            touch,
            self._rays(),
            self._target_obs(self.food),
            self._target_obs(self.water[:2]),
            [float(self.ate), float(self.drank)],
        ])
        return obs.astype(np.float32)

    def _wall_impact(self):
        d, m = self.d, self.m
        g = d.contact.geom[:d.ncon]
        hit = ((self.is_rat_geom[g[:, 0]] & self.is_wall[g[:, 1]]) |
               (self.is_rat_geom[g[:, 1]] & self.is_wall[g[:, 0]]))
        if not hit.any():
            return 0.0
        f6 = np.zeros(6)
        total = 0.0
        for i in np.flatnonzero(hit):
            mujoco.mj_contactForce(m, d, int(i), f6)
            total += abs(f6[0])
        return total

    def _sample_spawn(self):
        HX, HY = scene.HX, scene.HY
        margin = 0.15
        for _ in range(200):
            if self.rng.random() < self.spawn_near:
                tgt = self.food if self.rng.random() < 0.5 else self.water[:2]
                a = self.rng.uniform(0, 2 * np.pi)
                r = self.rng.uniform(0.20, 0.50)
                p = np.array([tgt[0] + r * np.cos(a), tgt[1] + r * np.sin(a)])
            else:
                p = np.array([self.rng.uniform(-HX + margin, HX - margin),
                              self.rng.uniform(-HY + margin, HY - margin)])
            if abs(p[0]) > HX - margin or abs(p[1]) > HY - margin:
                continue
            if np.linalg.norm(p - self.food) < 0.15:
                continue
            return p, self.rng.uniform(-np.pi, np.pi)
        return np.zeros(2), 0.0

    # -------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None or not hasattr(self, "_seeded"):
            self.rng = self.np_random
            self._seeded = True
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.q0
        p, yaw = self._sample_spawn()
        d.qpos[0:2] = p
        d.qpos[2] = self.torso_z0 + 0.005
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[7:] += self.rng.normal(0, 0.02, m.nq - 7)
        # clamp joint noise into joint limits
        for j in range(1, m.njnt):
            if m.jnt_limited[j]:
                a = m.jnt_qposadr[j]
                d.qpos[a] = np.clip(d.qpos[a], *m.jnt_range[j])
        mujoco.mj_forward(m, d)
        self.steps = 0
        self.ate = False
        self.drank = False
        self.prev_xy = d.xpos[self.torso][:2].copy()
        return self._obs(), {}

    def step(self, action):
        m, d = self.m, self.d
        a = np.clip(action, -1, 1)
        d.ctrl[:] = a
        impact = 0.0
        nbad = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
        for _ in range(self.nsub):
            mujoco.mj_step(m, d)
            if d.ncon:
                impact = max(impact, self._wall_impact())
        self.steps += 1

        # --- moving: forward speed of the torso over the control step
        xy = d.xpos[self.torso][:2].copy()
        vel = (xy - self.prev_xy) / CTRL_DT
        self.prev_xy = xy
        R = self._torso_frame()
        fwd = R[:2, 0] / (np.linalg.norm(R[:2, 0]) + 1e-9)
        v_fwd = float(vel @ fwd)
        r_move = MOVE_W * np.clip(v_fwd, -0.1, 0.3)

        # --- food / water
        head = d.xpos[self.skull]
        r_food = r_water = 0.0
        if not self.ate and np.linalg.norm(head[:2] - self.food) < FOOD_RADIUS and head[2] < FOOD_HEAD_Z:
            self.ate = True
            r_food = FOOD_REWARD
        if not self.drank and np.linalg.norm(head - self.water) < WATER_RADIUS:
            self.drank = True
            r_water = WATER_REWARD

        # --- hitting walls hard
        r_impact = -min(IMPACT_MAX, IMPACT_W * max(0.0, impact - IMPACT_FREE))
        r_ctrl = -CTRL_W * float(a @ a)
        reward = r_move + r_food + r_water + r_impact + r_ctrl

        up_z = R[2, 2]
        unstable = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > nbad
        fallen = up_z < 0.35 or unstable or not np.isfinite(d.qpos).all()
        truncated = self.steps >= self.max_steps
        info = dict(r_move=r_move, r_food=r_food, r_water=r_water, r_impact=r_impact,
                    v_fwd=v_fwd, impact=impact, ate=self.ate, drank=self.drank)
        return self._obs(), float(reward), bool(fallen), bool(truncated), info
