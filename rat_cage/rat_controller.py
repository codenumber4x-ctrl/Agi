"""Locomotion + behaviour controller for the MuJoCo virtual rodent.

Layers
------
1. ``GaitLibrary``  - pre-computes, by inverse kinematics on the real rat
   skeleton, joint-angle trajectories for each leg over one gait cycle
   (foot sweeps backwards on the ground during stance, lifts and swings
   forward during swing).  Rats walk with a lateral-sequence gait
   (LH -> LF -> RH -> RF) and switch to a trot when faster.
2. ``RatController`` - per-rat central pattern generator (CPG).  Every
   control step it advances the gait phase, mixes in steering (asymmetric
   stride + spine/neck bending), head/whisker sniffing, licking or chewing,
   and converts the desired joint angles to actuator controls of the
   38 position actuators of the rodent model.
3. ``Behaviour`` - a small ethogram-style state machine: explore (wall
   following / random waypoints), go-to-water + drink (licking at ~7 Hz),
   go-to-food + eat (head in bowl, chewing at ~5 Hz), groom/rest and
   sniff, with avoidance of walls, furniture and the cage-mate.

A light "postural assist" (external wrench on the torso: upright torque,
partial weight support, small heading/speed correction) stands in for the
balance reflexes a trained neural policy would provide.  Its strength is
configurable (``assist`` argument); legs, feet contacts, tail, head and
pellets are all fully simulated by MuJoCo.
"""

import mujoco
import numpy as np

import build_scene as scene

LEGS = {
    # name : (end-effector body, joints used for IK, phase offset)
    "LH": ("toe_L", ["hip_L_extend", "knee_L", "ankle_L"], 0.00),
    "LF": ("finger_L", ["shoulder_L", "elbow_L", "wrist_L"], 0.25),
    "RH": ("toe_R", ["hip_R_extend", "knee_R", "ankle_R"], 0.50),
    "RF": ("finger_R", ["shoulder_R", "elbow_R", "wrist_R"], 0.75),
}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(xmat):
    R = xmat.reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


class GaitLibrary:
    """IK-derived per-leg joint trajectories, built once on a single rat."""

    NPH = 64

    def __init__(self, stride=0.045, lift=0.016, duty=0.62):
        import os
        import warnings
        spec = mujoco.MjSpec()
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
        rat = mujoco.MjSpec.from_file(os.path.join(scene.ASSETS, "rodent.xml"))
        f = spec.worldbody.add_frame()
        f.attach_body(rat.body("torso"), "", "").add_freejoint()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = spec.compile()
        d = mujoco.MjData(m)
        # reference posture = the model's default (mocap-derived) standing pose
        mujoco.mj_kinematics(m, d)
        self.qref = d.qpos.copy()
        self.model = m
        self.duty = duty
        # Two IK libraries per leg: horizontal sweep only, and vertical lift
        # only.  Mixing them in joint space lets stride length scale with speed
        # while the paws always clear the bedding during swing.
        self.traj, self.lift = {}, {}
        self.qref_leg = {}
        for leg, (body, joints, _) in LEGS.items():
            self.traj[leg] = self._leg_traj(m, d, body, joints, stride, 0.0, duty)
            self.lift[leg] = self._leg_traj(m, d, body, joints, 0.0, lift, duty)
            self.qref_leg[leg] = np.array([self.qref[m.jnt_qposadr[
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in joints])

    @staticmethod
    def foot_path(ph, stride, lift, duty):
        """Foot offset (dx, dz) in the torso frame at phase ph in [0,1)."""
        if ph < duty:                      # stance: sweep front -> back
            s = ph / duty
            return stride * (0.5 - s), 0.0
        s = (ph - duty) / (1 - duty)       # swing: back -> front, lifted
        s_smooth = 0.5 - 0.5 * np.cos(np.pi * s)
        return stride * (-0.5 + s_smooth), lift * np.sin(np.pi * s) ** 1.5

    def _leg_traj(self, m, d, body, joints, stride, lift, duty):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        dofs = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints]
        qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints]
        rng = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints])
        d.qpos[:] = self.qref
        mujoco.mj_kinematics(m, d)
        R0 = d.xmat[1].reshape(3, 3).copy()
        p_home = R0.T @ (d.xpos[bid] - d.xpos[1])
        out = np.zeros((self.NPH, len(joints)))
        q = self.qref[qadr].copy()
        jacp = np.zeros((3, m.nv))
        for k in range(self.NPH):
            dx, dz = self.foot_path(k / self.NPH, stride, lift, duty)
            target = p_home + np.array([dx, 0.0, dz])
            for _ in range(60):
                d.qpos[:] = self.qref
                d.qpos[qadr] = q
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                p = R0.T @ (d.xpos[bid] - d.xpos[1])
                err = target - p
                if np.linalg.norm(err) < 1e-5:
                    break
                mujoco.mj_jacBody(m, d, jacp, None, bid)
                J = (R0.T @ jacp)[:, dofs][[0, 2]]          # x/z only (sagittal)
                lam = 1e-4
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(2), err[[0, 2]])
                # null-space pull toward the reference posture
                N = np.eye(len(joints)) - np.linalg.pinv(J) @ J
                dq += N @ (0.1 * (self.qref[qadr] - q))
                q = np.clip(q + dq, rng[:, 0], rng[:, 1])
            out[k] = q
        return out

    def _interp(self, table, ph):
        x = (ph % 1.0) * self.NPH
        i = int(x)
        f = x - i
        return table[i % self.NPH] * (1 - f) + table[(i + 1) % self.NPH] * f

    def sample(self, leg, ph, stride_scale=1.0, lift_scale=1.0):
        q0 = self.qref_leg[leg]
        return (q0 + stride_scale * (self._interp(self.traj[leg], ph) - q0)
                + lift_scale * (self._interp(self.lift[leg], ph) - q0))


class RatController:
    """Drives one prefixed rat (e.g. 'rat1/') inside the full cage model."""

    def __init__(self, model, prefix, gait, seed=0, assist=1.0):
        self.m = model
        self.p = prefix
        self.gait = gait
        self.rng = np.random.default_rng(seed)
        self.assist = assist
        m = model
        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + "torso")
        self.skull = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + "skull")
        self.mass = m.body_subtreemass[self.torso]
        self.act = {}
        for a in range(m.nu):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            if n.startswith(prefix):
                self.act[n[len(prefix):]] = a
        self.jq = {}
        for j in range(m.njnt):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if n and n.startswith(prefix):
                self.jq[n[len(prefix):]] = m.jnt_qposadr[j]
        # actuator gain / bias: force = g*act + b0 + b1*length  =>  target = (g*u + b0)/(-b1)
        self.g = m.actuator_gainprm[:, 0].copy()
        self.b0 = m.actuator_biasprm[:, 0].copy()
        self.kp = -m.actuator_biasprm[:, 1].copy()
        self.phase = self.rng.uniform(0, 1)
        self.freq = 3.5            # stride frequency (Hz); rats walk at ~3-4 Hz
        self.speed = 0.0           # 0..1 (fraction of nominal stride)
        self.turn = 0.0            # -1..1 (positive = turn left)
        self.head = {"pitch": 0.0, "yaw": 0.0}
        self.mouth = 0.0
        self.t = 0.0
        self.sniff_amp = 1.0
        # postural-assist components (scaled by `assist`)
        self.support = 0.35        # fraction of body weight carried (balance reflex proxy)
        self.kh = 0.08             # heading-correction torque gain (N m / rad)
        self.push = 0.0            # N per (m/s) forward-speed error (0: legs alone propel)

    # -- helpers ----------------------------------------------------------
    def pose(self, data):
        p = data.xpos[self.torso]
        return p[:2].copy(), yaw_of(data.xmat[self.torso])

    def head_pos(self, data):
        return data.xpos[self.skull].copy()

    def set_target(self, data, name, q):
        """Joint-angle (or tendon-length) target -> actuator ctrl in [-1, 1]."""
        a = self.act[name]
        u = (self.kp[a] * q - self.b0[a]) / self.g[a]
        data.ctrl[a] = np.clip(u, -1.0, 1.0)

    def neutral(self, name):
        # the reference standing posture is the model's default pose (q = 0)
        return 0.0

    # -- main update ------------------------------------------------------
    def step(self, data, dt, cmd):
        """cmd: dict(speed, turn, head_pitch, head_yaw, mouth, heading, rear)."""
        self.t += dt
        # smooth command changes (muscle / motor-pattern inertia)
        a = 1 - np.exp(-dt / 0.15)
        self.speed += a * (cmd.get("speed", 0.0) - self.speed)
        self.turn += a * (cmd.get("turn", 0.0) - self.turn)
        for k in ("pitch", "yaw"):
            self.head[k] += a * (cmd.get("head_" + k, 0.0) - self.head[k])

        moving = self.speed > 0.05
        freq = self.freq if moving else 0.0
        self.phase = (self.phase + freq * dt) % 1.0

        # ---------------- legs (CPG + IK library)
        for leg, (_, joints, off) in LEGS.items():
            left = leg[0] == "L"
            side_scale = 1.0 + 0.8 * self.turn if left else 1.0 - 0.8 * self.turn
            amp = np.clip(self.speed * side_scale, -0.3, 1.4)
            if moving:
                q = self.gait.sample(leg, self.phase + off, amp, 1.0)
            else:
                q = self.gait.qref_leg[leg]
            for j, qj in zip(joints, q):
                self.set_target(data, j, qj)
        # abduct the legs a little more when turning (wider base)
        for s in ("L", "R"):
            self.set_target(data, f"hip_{s}_abduct", self.neutral(f"hip_{s}_abduct") + 0.1 * abs(self.turn))

        # ---------------- spine & neck: steering, head orientation, sniffing
        t = self.t
        sniff = self.sniff_amp * (0.04 * np.sin(2 * np.pi * 7.5 * t) + 0.05 * np.sin(2 * np.pi * 0.6 * t))
        sway = 0.025 * np.sin(2 * np.pi * self.phase) * self.speed
        self.set_target(data, "lumbar_bend", -0.25 * self.turn + sway)
        self.set_target(data, "lumbar_extend", self.neutral("lumbar_extend") + cmd.get("rear", 0.0) * 0.6)
        self.set_target(data, "lumbar_twist", 0.0)
        self.set_target(data, "cervical_bend", 0.5 * self.turn + self.head["yaw"])
        # (negative cervical extension raises the head)
        self.set_target(data, "cervical_extend",
                        self.neutral("cervical_extend") - self.head["pitch"] + sniff)
        self.set_target(data, "cervical_twist", 0.0)
        self.set_target(data, "atlas", self.neutral("atlas") + 0.3 * sniff)
        # tail: passive-looking sway counter-phase to the body
        self.set_target(data, "caudal_bend", -0.6 * self.turn - 2.0 * sway
                        + 0.15 * np.sin(2 * np.pi * 0.3 * t + 1.0))
        self.set_target(data, "caudal_extend", self.neutral("caudal_extend"))
        # mouth: licking (~7 Hz) or chewing (~5 Hz)
        mouth = cmd.get("mouth", 0.0)
        hz = 7.0 if cmd.get("mode") == "drink" else 5.0
        # mandible: negative = closed, positive = open
        self.set_target(data, "mandible",
                        -0.24 + mouth * 0.30 * (0.5 + 0.5 * np.sin(2 * np.pi * hz * t)))

        if cmd.get("forepaws_up", 0.0) > 0:
            k = cmd["forepaws_up"]
            for s in ("L", "R"):
                self.set_target(data, f"shoulder_{s}", self.neutral(f"shoulder_{s}") - 0.6 * k)
                self.set_target(data, f"elbow_{s}", self.neutral(f"elbow_{s}") - 0.6 * k)

        self._assist(data, cmd)

    # -- postural assist ---------------------------------------------------
    def _assist(self, data, cmd):
        if self.assist <= 0:
            return
        k = self.assist
        b = self.torso
        R = data.xmat[b].reshape(3, 3)
        up = R[:, 2]
        w = data.cvel[b][:3]                      # angular velocity (world)
        v = data.cvel[b][3:]
        # uprighting torque: align torso z with world z, tilted nose-up when rearing
        fwd_h = R[:, 0].copy(); fwd_h[2] = 0; fwd_h /= np.linalg.norm(fwd_h) + 1e-9
        phi = 0.55 * cmd.get("rear", 0.0)
        up_des = np.cos(phi) * np.array([0, 0, 1.0]) - np.sin(phi) * fwd_h
        err = np.cross(up, up_des)
        torque = k * (0.10 * err - 0.004 * w)
        # heading
        if "heading" in cmd and cmd["heading"] is not None:
            yaw = yaw_of(data.xmat[b])
            torque[2] += k * self.kh * wrap(cmd["heading"] - yaw)
        torque[2] -= k * 0.0015 * w[2]
        # partial weight support (balance reflexes) and height regulation
        z = data.xpos[b][2]
        z_des = 0.050 + 0.03 * cmd.get("rear", 0.0)
        fz = k * (self.support * self.mass * 9.81 + 8.0 * (z_des - z) - 0.4 * v[2])
        fz = max(fz, 0.0)
        # mild horizontal speed tracking along the body axis
        fwd = R[:, 0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd) + 1e-9
        v_des = 0.17 * self.speed
        f_xy = k * self.push * (v_des - fwd @ v[:]) * fwd
        lat = np.array([-fwd[1], fwd[0], 0])
        f_xy -= k * 0.6 * (lat @ v) * lat
        data.xfrc_applied[b, :3] = [f_xy[0], f_xy[1], fz]
        data.xfrc_applied[b, 3:] = torque


class Behaviour:
    """Ethogram-like state machine for one rat."""

    def __init__(self, ctrl, other=None, seed=0, start="explore"):
        self.c = ctrl
        self.other = other
        self.rng = np.random.default_rng(seed + 100)
        self.state = start
        self.t_state = 0.0
        self.goal = None
        self.dur = 0.0
        self.thirst = self.rng.uniform(0.3, 0.7)
        self.hunger = self.rng.uniform(0.3, 0.7)
        self._new_state(start)

    # obstacles inside the cage as (cx, cy, radius) for repulsion
    OBST = [
        (scene.HIDE_HOUSE[0] + 0.01, scene.HIDE_HOUSE[1], 0.13),
        (scene.FOOD_BOWL[0], scene.FOOD_BOWL[1], 0.085),
        (-0.07, scene.HY - 0.05, 0.10),           # ramp foot / body
        (-0.16, scene.HY - 0.05, 0.08),
    ]

    def _random_waypoint(self):
        for _ in range(100):
            p = np.array([self.rng.uniform(-scene.HX + 0.10, scene.HX - 0.10),
                          self.rng.uniform(-scene.HY + 0.09, scene.HY - 0.09)])
            if all(np.hypot(*(p - o[:2])) > o[2] + 0.04 for o in map(np.array, self.OBST)):
                return p
        return np.zeros(2)

    def _new_state(self, s):
        self.state = s
        self.t_state = 0.0
        self.t_feed = 0.0
        if s == "explore":
            self.goal = self._random_waypoint()
            self.dur = self.rng.uniform(4, 8)
        elif s == "to_water":
            self.goal = scene.WATER_DRINK_POS.copy()
        elif s == "drink":
            self.dur = self.rng.uniform(4.0, 6.0)
        elif s == "to_food":
            self.goal = scene.FOOD_EAT_POS.copy()
        elif s == "eat":
            self.dur = self.rng.uniform(5.0, 7.0)
        elif s == "sniff":
            self.dur = self.rng.uniform(1.5, 3.0)

    def _steer(self, data, goal, arrive=0.03):
        pos, yaw = self.c.pose(data)
        to = goal - pos
        dist = np.linalg.norm(to)
        desired = to / (dist + 1e-9)
        # repulsion from walls, furniture and the other rat
        rep = np.zeros(2)
        for ax, lim in ((0, scene.HX), (1, scene.HY)):
            for sgn in (1, -1):
                gap = lim - sgn * pos[ax]
                if gap < 0.10:
                    v = np.zeros(2); v[ax] = -sgn
                    rep += v * (0.10 - gap) / 0.10 * 1.5
        for (ox, oy, r) in self.OBST:
            dv = pos - np.array([ox, oy])
            dd = np.linalg.norm(dv)
            if dd < r + 0.06 and np.linalg.norm(goal - np.array([ox, oy])) > r + 0.02:
                rep += dv / (dd + 1e-9) * (r + 0.06 - dd) / 0.06 * 1.2
        if self.other is not None:
            op, _ = self.other.pose(data)
            dv = pos - op
            dd = np.linalg.norm(dv)
            if dd < 0.16:
                rep += dv / (dd + 1e-9) * (0.16 - dd) / 0.16 * 2.0
        v = desired + rep
        heading = np.arctan2(v[1], v[0])
        err = wrap(heading - yaw)
        turn = np.clip(1.6 * err, -1, 1)
        # slow down (but keep stepping) while turning; feet in stance resist yaw
        speed = np.clip(dist / 0.08, 0.5, 1.0) * np.clip(1.2 - abs(err) / 1.6, 0.5, 1.0)
        return dict(speed=speed if dist > arrive else 0.0, turn=turn, heading=heading), dist

    def update(self, data, dt):
        self.t_state += dt
        self.thirst += dt * 0.025
        self.hunger += dt * 0.02
        s = self.state
        cmd = dict(speed=0.0, turn=0.0)
        if s == "explore":
            cmd, dist = self._steer(data, self.goal)
            cmd["head_pitch"] = 0.05
            if dist < 0.04:
                self._new_state("sniff")
            elif self.t_state > self.dur:
                self._new_state("sniff")
        elif s == "sniff":
            _, yaw = self.c.pose(data)
            cmd = dict(speed=0.0, turn=0.0, heading=yaw,
                       head_yaw=0.35 * np.sin(self.t_state * 2.2),
                       head_pitch=-0.15 + 0.2 * np.sin(self.t_state * 1.3),
                       rear=0.35 * np.clip(np.sin(self.t_state * 1.5), 0, 1))
            if self.t_state > self.dur:
                if self.thirst > 1.0:
                    self._new_state("to_water")
                elif self.hunger > 1.0:
                    self._new_state("to_food")
                else:
                    self._new_state("explore")
        elif s in ("to_water", "to_food"):
            cmd, dist = self._steer(data, self.goal, arrive=0.02)
            if dist < 0.035 or self.t_state > 25:
                self._new_state("drink" if s == "to_water" else "eat")
        elif s in ("drink", "eat"):
            pos, yaw = self.c.pose(data)
            nose = self.c.head_pos(data)[:2]
            if s == "drink":
                # face the sipper tube, stretch up and lick at ~7 Hz
                target, reach = scene.WATER_SPOUT[:2], 0.03
                cmd = dict(head_pitch=0.30, mouth=1.0, mode="drink", rear=0.45)
            else:
                # head down into the bowl, gnaw pellets (~5 Hz chewing)
                target, reach = scene.FOOD_BOWL, 0.07
                cmd = dict(head_pitch=-0.35, mouth=1.0, mode="eat", rear=0.0,
                           head_yaw=0.12 * np.sin(self.t_state * 0.9))
            heading = np.arctan2(*(target - pos)[::-1])
            cmd["heading"] = heading
            cmd["turn"] = np.clip(1.5 * wrap(heading - yaw), -1, 1) * 0.4
            in_place = np.linalg.norm(nose - target) < reach
            if not in_place:
                # still approaching: walk with a neutral head posture
                cmd = dict(heading=heading, speed=0.55, turn=cmd["turn"])
            self.c.sniff_amp = 0.25
            if in_place:
                self.t_feed += dt
            if self.t_feed > self.dur or self.t_state > self.dur + 10:
                if s == "drink":
                    self.thirst = 0.0
                else:
                    self.hunger = 0.0
                self.c.sniff_amp = 1.0
                self._new_state("explore")
        cmd.setdefault("heading", None)
        return cmd
