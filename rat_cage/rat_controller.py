"""Locomotion + behaviour controller for the MuJoCo virtual rats.

The rat is a fully physical, articulated body (DeepMind virtual rodent:
anatomical skeleton, 38 position-servo "muscles" with activation dynamics).
There is no trained neural policy here; instead the controller is a classic
hand-designed hierarchy:

1. Behaviour state machine (per rat): explore / walk-to-goal, drink at the
   sipper, eat pellets at the bowl (pick one up, sit back on the haunches and
   nibble it held in the forepaws), rear up and sniff, groom, rest.
2. Gait central pattern generator (CPG): rat walk (lateral sequence) at low
   speed, trot (diagonal pairs) at higher speed. Stride length and frequency
   follow measured rat gait scaling (stride ~ speed / frequency, 2.5-6 Hz).
3. Leg inverse kinematics: paw trajectories (stance / swing with lift) are
   converted into hip/knee/ankle and shoulder/elbow/wrist angles using an IK
   lookup table solved on the model itself at start-up.
4. Actuator command: joint targets -> the rodent's position-servo controls,
   with feed-forward compensation of the 40 ms muscle activation filter.
5. Postural assistance: a weak "virtual spring" (external force/torque on
   the torso through ``xfrc_applied``) helps keep balance and heading, the
   role played by a learned balance policy in the original work.  Feet still
   carry a large part of the body weight and make real contacts.
"""
import math

import mujoco
import numpy as np

import cage

G = 9.81
HIND = ("hip_{s}_extend", "knee_{s}", "ankle_{s}")
FORE = ("shoulder_{s}", "elbow_{s}", "wrist_{s}")


# --------------------------------------------------------------------- math
def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def euler_zyx_to_mat(yaw, pitch, roll=0.0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def rotvec(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ----------------------------------------------------------------- leg IK
class LegIK:
    """IK lookup tables: paw offset (dx, dz) in torso frame -> 3 joint angles.

    Solved once on a standalone copy of the rodent with damped least squares.
    """
    DX = np.linspace(-0.035, 0.035, 15)
    DZ = np.linspace(-0.006, 0.026, 9)
    # crouched nominal stance (joints near mid-range so paws can lift/reach)
    NOMINAL = {"hind": np.array([0.0, 0.4, 0.6]), "fore": np.array([-0.1, 0.5, 0.1])}

    def __init__(self):
        m = mujoco.MjModel.from_xml_path(cage.os.path.join(cage.ASSETS, "rodent.xml"))
        d = mujoco.MjData(m)
        self.tables = {}
        for kind, names, site in (("hind", HIND, "sole_L"), ("fore", FORE, "palm_L")):
            jids = [m.joint(n.format(s="L")).id for n in names]
            qadr = [m.jnt_qposadr[j] for j in jids]
            dadr = [m.jnt_dofadr[j] for j in jids]
            sid = m.site(site).id
            qn = self.NOMINAL[kind]
            d.qpos[:] = 0
            d.qpos[qadr] = qn
            mujoco.mj_kinematics(m, d)
            p0 = d.site_xpos[sid].copy()
            tab = np.zeros((len(self.DX), len(self.DZ), 3))
            jac = np.zeros((3, m.nv))
            lo, hi = m.jnt_range[jids, 0] + 0.02, m.jnt_range[jids, 1] - 0.02
            for i, dx in enumerate(self.DX):
                for k, dz in enumerate(self.DZ):
                    target = p0 + np.array([dx, 0, dz])
                    q = qn.copy()
                    for _ in range(60):
                        d.qpos[:] = 0
                        d.qpos[qadr] = q
                        mujoco.mj_kinematics(m, d)
                        mujoco.mj_comPos(m, d)
                        e = (target - d.site_xpos[sid])[[0, 2]]
                        mujoco.mj_jacSite(m, d, jac, None, sid)
                        J = jac[:, dadr][[0, 2]]
                        lam = 3e-4
                        dq = np.linalg.solve(J.T @ J + lam * np.eye(3), J.T @ e - 0.5 * lam * (q - qn))
                        q = np.clip(q + dq, lo, hi)
                    tab[i, k] = q
            self.tables[kind] = tab

    def __call__(self, kind, dx, dz):
        tab = self.tables[kind]
        fx = np.interp(dx, self.DX, np.arange(len(self.DX)))
        fz = np.interp(dz, self.DZ, np.arange(len(self.DZ)))
        i0, k0 = int(min(fx, len(self.DX) - 2)), int(min(fz, len(self.DZ) - 2))
        ax, az = fx - i0, fz - k0
        return ((1 - ax) * (1 - az) * tab[i0, k0] + ax * (1 - az) * tab[i0 + 1, k0]
                + (1 - ax) * az * tab[i0, k0 + 1] + ax * az * tab[i0 + 1, k0 + 1])


# ------------------------------------------------------------------ the rat
class Rat:
    STAND_H = 0.073          # torso height above bedding while walking
    MASS_SUPPORT = 0.80      # fraction of body weight assisted
    LEAD = 0.045             # s, phase lead compensating muscle activation lag

    def __init__(self, model, prefix, ik, rng, other_names=()):
        self.m, self.p, self.ik, self.rng = model, prefix, ik, rng
        m = model
        self.torso = m.body(prefix + "torso").id
        self.skull = m.body(prefix + "skull").id
        self.jaw = m.body(prefix + "jaw").id
        self.pelvis = m.body(prefix + "pelvis").id
        self.hand = {s: m.body(prefix + f"hand_{s}").id for s in "LR"}
        self.root_dof = m.jnt_dofadr[m.joint(prefix + "root").id]
        self.mass = m.body_subtreemass[self.torso]
        # actuator bookkeeping: ctrl = (target * -b1 - b0) / gain
        self.act = {}
        for i in range(m.nu):
            n = m.actuator(i).name
            if n.startswith(prefix):
                self.act[n[len(prefix):]] = i
        self.gain = m.actuator_gainprm[:, 0].copy()
        self.b0 = m.actuator_biasprm[:, 0].copy()
        self.b1 = m.actuator_biasprm[:, 1].copy()
        self.tau = m.actuator_dynprm[:, 0].copy()
        self.prev_target = {}
        # state
        self.phase = rng.random()
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.heading = None
        self.mode = "idle"
        self.mode_t = 0.0
        self.mode_dur = 1.0
        self.goal = None
        self.goal_heading = None
        self.after_goal = None
        self.posture = dict(head_pitch=0.0, head_yaw=0.0, rear=0.0, sit=0.0, chew=0.0,
                            lick=0.0, groom=0.0, crouch=0.0, paws_up=0.0)
        self.post_s = dict(self.posture)
        self.held = None
        self.history = []
        self.hunger = rng.uniform(0.3, 0.8)
        self.thirst = rng.uniform(0.3, 0.8)

    # ---------------------------------------------------------- state access
    def pos(self, d):
        return d.xpos[self.torso]

    def yaw(self, d):
        R = d.xmat[self.torso].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    def nose(self, d):
        R = d.xmat[self.skull].reshape(3, 3)
        return d.xpos[self.skull] + R @ np.array([0.035, 0, -0.005])

    # ------------------------------------------------------------- commands
    def set_ctrl(self, d, name, target, dt):
        i = self.act[name]
        self.prev_target[name] = target
        u = (target * -self.b1[i] - self.b0[i]) / self.gain[i]
        d.ctrl[i] = np.clip(u, -1, 1)

    # -------------------------------------------------------------- gait
    def gait_offsets(self, dt):
        """Returns per-leg (dx, dz) paw offsets from the CPG."""
        v = self.speed
        turn = self.yaw_rate
        moving = abs(v) > 0.02 or abs(turn) > 0.4
        if moving:
            freq = np.clip(2.6 + 9.0 * abs(v), 2.6, 6.0)
            self.phase = (self.phase + freq * dt) % 1.0
        trot = abs(v) > 0.16
        duty = 0.5 if trot else 0.65
        # leg phase offsets: trot = diagonal pairs, walk = lateral sequence
        offs = ({"hind_L": 0.0, "fore_R": 0.0, "hind_R": 0.5, "fore_L": 0.5} if trot else
                {"hind_L": 0.0, "fore_L": 0.25, "hind_R": 0.5, "fore_R": 0.75})
        freq = np.clip(2.6 + 9.0 * abs(v), 2.6, 6.0)
        stride = np.clip(v / freq, -0.05, 0.06)
        out = {}
        amp_gate = smoothstep((max(abs(v), abs(turn) * 0.03) - 0.01) / 0.03)
        for leg, off in offs.items():
            side = 1 if leg.endswith("L") else -1
            s_leg = stride - side * turn * 0.012          # turn: inside legs shorter
            ph = (self.phase + off + freq * self.LEAD) % 1.0
            if ph < duty:
                x = s_leg * (0.5 - ph / duty)
                z = 0.0
            else:
                sw = (ph - duty) / (1 - duty)
                x = s_leg * (-0.5 + 0.5 * (1 - math.cos(math.pi * sw)))
                z = (0.016 if leg.startswith("hind") else 0.016) * math.sin(math.pi * sw)
            out[leg] = (x * amp_gate, z * amp_gate)
        return out

    # ------------------------------------------------------------ posture
    def apply_pose(self, d, t, dt, legs):
        ps = self.post_s
        k = 1 - math.exp(-dt / 0.18)
        for key, val in self.posture.items():
            ps[key] += (val - ps[key]) * k
        sit, rear, groom, paws = ps["sit"], ps["rear"], ps["groom"], ps["paws_up"]
        lift_fore = max(sit, rear, groom, paws)
        for s in "LR":
            dx, dz = legs[f"hind_{s}"]
            # sitting / rearing: hind paws tucked forward under the body
            dx += 0.022 * max(sit, groom) + 0.012 * rear
            dz += 0.006 * max(sit, groom)
            qh = self.ik("hind", dx, dz)
            for n, q in zip(HIND, qh):
                self.set_ctrl(d, n.format(s=s), q, dt)
            dxf, dzf = legs[f"fore_{s}"]
            qf = self.ik("fore", dxf, dzf)
            # raised forepaws: hands brought up to the mouth (eating/grooming)
            raised = np.array([-0.45, -0.54, 0.84])   # IK: palms just under the mouth
            if groom > 0.05:
                wob = 0.25 * math.sin(2 * math.pi * 3.5 * t + (0 if s == "L" else 0.6))
                raised = raised + np.array([wob, 0.3 * wob, 0])
            q = (1 - lift_fore) * qf + lift_fore * raised
            for n, qq in zip(FORE, q):
                self.set_ctrl(d, n.format(s=s), qq, dt)
            self.set_ctrl(d, f"shoulder_sup_{s}", 0.25 + 0.53 * lift_fore, dt)
            self.set_ctrl(d, f"finger_{s}", 0.6 * lift_fore, dt)
            self.set_ctrl(d, f"toe_{s}", 0.1, dt)
            self.set_ctrl(d, f"hip_{s}_abduct", 0.05 + 0.1 * sit, dt)
            self.set_ctrl(d, f"hip_{s}_supinate", 0.0, dt)
            self.set_ctrl(d, f"scapula_{s}_abduct", -0.19 * lift_fore, dt)
            self.set_ctrl(d, f"scapula_{s}_supinate", 0.05, dt)
            self.set_ctrl(d, f"scapula_{s}_extend", 0.0, dt)
        # spine: bend into turns, flex when sitting
        self.set_ctrl(d, "lumbar_bend", np.clip(0.18 * self.yaw_rate, -0.35, 0.35), dt)
        self.set_ctrl(d, "lumbar_extend", 0.25 * max(sit, groom) - 0.05 * rear, dt)
        self.set_ctrl(d, "lumbar_twist", 0.0, dt)
        # head: pitch (+ is down for the cervical tendon), yaw, sniffing bobs
        sniff = 0.04 * math.sin(2 * math.pi * 7.0 * t) if self.mode in ("explore", "rear", "idle") else 0.0
        self.set_ctrl(d, "cervical_extend", np.clip(ps["head_pitch"] + sniff + 0.45 * groom, -0.4, 0.6), dt)
        self.set_ctrl(d, "cervical_bend", np.clip(ps["head_yaw"] + 0.12 * self.yaw_rate, -0.6, 0.6), dt)
        self.set_ctrl(d, "cervical_twist", 0.0, dt)
        self.set_ctrl(d, "atlas", 0.2 + 0.2 * max(ps["lick"], ps["chew"]), dt)
        # jaw: chewing ~5 Hz, licking ~7 Hz
        jaw = (-0.05 + ps["chew"] * 0.12 * (0.5 + 0.5 * math.sin(2 * math.pi * 5.2 * t))
               + ps["lick"] * 0.14 * (0.5 + 0.5 * math.sin(2 * math.pi * 7.0 * t)))
        self.set_ctrl(d, "mandible", -jaw, dt)
        # tail: slow sway counter to turning
        self.set_ctrl(d, "caudal_bend", np.clip(-0.08 * self.yaw_rate + 0.05 * math.sin(0.9 * t + self.phase * 6), -0.2, 0.2), dt)
        self.set_ctrl(d, "caudal_extend", 0.02 * rear, dt)

    # ----------------------------------------------------- postural assist
    def assist(self, d, dt):
        """Balance/heading assistance applied to the torso (see module doc)."""
        ps = self.post_s
        p = d.xpos[self.torso]
        R = d.xmat[self.torso].reshape(3, 3)
        v = d.qvel[self.root_dof:self.root_dof + 3]
        w = d.qvel[self.root_dof + 3:self.root_dof + 6]  # world-frame ang vel
        cur_yaw = math.atan2(R[1, 0], R[0, 0])
        if self.heading is None:
            self.heading = cur_yaw
        self.heading = wrap(self.heading + self.yaw_rate * dt)
        lag = wrap(self.heading - cur_yaw)
        if abs(lag) > 0.45:                      # don't wind up if the body is blocked
            self.heading = wrap(cur_yaw + math.copysign(0.45, lag))
        pitch = -0.95 * ps["rear"] - 0.55 * max(ps["sit"], ps["groom"]) + 0.15 * ps["head_pitch"] * (1 - ps["sit"])
        h = (self.STAND_H + 0.055 * ps["rear"] + 0.014 * max(ps["sit"], ps["groom"])
             - 0.008 * ps["crouch"] + cage.BED_Z)
        Rt = euler_zyx_to_mat(self.heading, pitch, 0.0)
        err = rotvec(Rt @ R.T)
        I = 2.5e-4
        torque = I * (900 * err - 60 * w)
        fwd = np.array([math.cos(self.heading), math.sin(self.heading), 0])
        vdes = fwd * self.speed
        fxy = self.mass * 25.0 * (vdes[:2] - v[:2])
        support = self.MASS_SUPPORT + 0.35 * max(ps["rear"], ps["sit"], ps["groom"])
        # vertical support split between shoulders (torso) and hips (pelvis)
        fz = self.mass * (0.5 * support * G + 300 * (h - p[2]) - 25 * v[2])
        pp = d.xpos[self.pelvis]
        hp = h + 0.062 * math.sin(pitch) + 0.003
        vp = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, vp, 0)
        fzp = self.mass * (0.5 * support * G + 300 * (hp - pp[2]) - 25 * vp[5])
        fz = np.clip(fz, 0, 0.9 * self.mass * G)
        fzp = np.clip(fzp, 0, 0.9 * self.mass * G)
        d.xfrc_applied[self.torso, :3] = [fxy[0], fxy[1], fz]
        d.xfrc_applied[self.pelvis, :3] = [0.0, 0.0, fzp]
        d.xfrc_applied[self.torso, 3:] = np.clip(torque, -0.035, 0.035)


# ------------------------------------------------------------ behaviour
class Behaviour:
    """Goal selection, navigation with obstacle avoidance and activity timing."""

    def __init__(self, rat, others, pellets, model):
        self.r = rat
        self.others = others
        self.pellets = pellets
        self.m = model
        self.water_spot = np.array([cage.SIPPER_TIP[0], cage.SIPPER_TIP[1] + 0.100])
        self.water_heading = -math.pi / 2
        self.plan = []          # optional scripted list of first activities

    # obstacles: (center, radius) circles + cage walls
    def obstacles(self, d):
        obs = [(cage.BOWL_POS, cage.BOWL_R + 0.07)]
        hx, hy = cage.HOUSE_SIZE[:2] / 2
        c = cage.HOUSE_POS
        for ox in np.linspace(-hx, hx, 4):
            for oy in (-hy, hy):
                obs.append((c + [ox, oy], 0.09))
        for oy in np.linspace(-hy, hy, 3):
            obs.append((c + [hx, oy], 0.09))
            obs.append((c + [-hx, oy], 0.09))
        for o in self.others:
            obs.append((o.pos(d)[:2].copy(), 0.13))
            obs.append((o.nose(d)[:2].copy(), 0.06))
        return obs

    def free_point(self, d, margin=0.12):
        for _ in range(100):
            p = np.array([self.r.rng.uniform(-cage.L / 2 + margin, cage.L / 2 - margin),
                          self.r.rng.uniform(-cage.W / 2 + margin, cage.W / 2 - margin)])
            if all(np.linalg.norm(p - c) > rad + 0.02 for c, rad in self.obstacles(d)):
                return p
        return np.zeros(2)

    def choose_next(self, d, t):
        r = self.r
        r.thirst += 0.0
        if self.plan:
            self.launch(d, self.plan.pop(0))
            return
        opts = ["explore", "explore", "rear", "groom", "drink", "eat"]
        weights = np.array([2.0, 1.0, 1.0, 0.6, 1.4 * r.thirst, 1.6 * r.hunger])
        if r.history and r.history[-1] in ("drink", "eat"):
            weights[opts.index(r.history[-1])] *= 0.15
        weights /= weights.sum()
        act = r.rng.choice(opts, p=weights)
        # don't queue at a resource the other rat is using
        for o in self.others:
            if act == "drink" and o.mode in ("drink", "to_drink"):
                act = "explore"
            if act == "eat" and o.mode in ("eat", "to_eat", "pick"):
                act = "explore"
        self.launch(d, act)

    def launch(self, d, act):
        r = self.r
        r.history.append(act)
        if act == "drink":
            self.go(d, self.water_spot, "drink", self.water_heading)
        elif act == "eat":
            a = math.atan2(-0.6 + r.rng.uniform(-0.5, 0.5), 1.0)
            spot = cage.BOWL_POS + 0.125 * np.array([math.cos(a), math.sin(a)])
            self.go(d, spot, "eat", math.atan2(*(cage.BOWL_POS - spot)[::-1]))
        elif act == "rear":
            self.go(d, self.free_point(d), "rear", None)
        elif act == "groom":
            self.set_mode("groom", r.rng.uniform(2.5, 4.0))
        else:
            self.go(d, self.free_point(d), "explore", None)

    def set_mode(self, mode, dur):
        r = self.r
        r.mode, r.mode_t, r.mode_dur = mode, 0.0, dur
        for k in r.posture:
            r.posture[k] = 0.0
        r.speed = 0.0
        r.yaw_rate = 0.0

    def go(self, d, goal, then, heading):
        self.progress = []
        self.set_mode("walk", 12.0)
        self.r.goal, self.r.after_goal, self.r.goal_heading = np.asarray(goal, float), then, heading
        self.r.cruise = self.r.rng.uniform(0.13, 0.26)

    def steer(self, d, dt):
        r = self.r
        p = r.pos(d)[:2]
        to = r.goal - p
        dist = np.linalg.norm(to)
        desired = to / max(dist, 1e-6)
        rep = np.zeros(2)
        for c, rad in self.obstacles(d):
            v = p - c
            dd = np.linalg.norm(v)
            if dd < rad + 0.06 and np.linalg.norm(r.goal - c) > rad * 0.6:
                rep += v / max(dd, 1e-3) * (rad + 0.06 - dd) / 0.06 * 1.6
        hx, hy = cage.L / 2 - 0.09, cage.W / 2 - 0.09
        rep[0] += max(0, p[0] - hx) * -25 + max(0, -hx - p[0]) * 25
        rep[1] += max(0, p[1] - hy) * -25 + max(0, -hy - p[1]) * 25
        dirv = desired + rep
        target_yaw = math.atan2(dirv[1], dirv[0])
        yaw = r.heading if r.heading is not None else r.yaw(d)
        e = wrap(target_yaw - yaw)
        r.yaw_rate = np.clip(4.0 * e, -3.5, 3.5)
        slow = max(smoothstep(dist / 0.12), 0.45) * (0.35 + 0.65 * smoothstep(1 - abs(e) / 1.6))
        r.speed = r.cruise * slow
        # stuck detection: little progress for a while -> back off and re-plan
        self.progress = getattr(self, "progress", [])
        self.progress.append(dist)
        if len(self.progress) > 1000:
            self.progress.pop(0)
            if self.progress[0] - dist < 0.015:
                self.stuck = True
        return dist

    def update(self, d, t, dt):
        r = self.r
        r.mode_t += dt
        P = r.posture
        r.thirst = min(1.0, r.thirst + dt * 0.01)
        r.hunger = min(1.0, r.hunger + dt * 0.008)
        if r.mode == "idle":
            r.speed = 0.0
            r.yaw_rate = 0.0
            P["head_yaw"] = 0.35 * math.sin(0.8 * t + r.phase * 5)
            if r.mode_t > r.mode_dur:
                self.choose_next(d, t)
        elif r.mode == "walk":
            self.stuck = False
            dist = self.steer(d, dt)
            P["head_pitch"] = 0.12
            if self.stuck and dist > 0.06:
                self.progress = []
                self.set_mode("backoff", 0.9)
                return
            if dist < 0.035 or r.mode_t > r.mode_dur:
                r.speed = 0.0
                nxt = r.after_goal
                if r.goal_heading is not None:
                    self.set_mode("align", 2.0)
                    r.after_goal = nxt
                else:
                    self.start(nxt)
        elif r.mode == "align":
            r.speed = 0.0
            yaw = r.heading
            e = wrap(r.goal_heading - yaw)
            r.yaw_rate = np.clip(3.0 * e, -2.5, 2.5)
            # small positional correction toward the goal spot
            p = r.pos(d)[:2]
            to = r.goal - p
            fwd = np.array([math.cos(yaw), math.sin(yaw)])
            r.speed = np.clip(6 * float(to @ fwd), -0.05, 0.08)
            if (abs(e) < 0.08 and np.linalg.norm(to) < 0.02) or r.mode_t > r.mode_dur:
                self.start(r.after_goal)
        elif r.mode == "drink":
            # stretch up to the sipper ball and lick
            P["head_pitch"] = -0.35
            P["lick"] = 1.0
            P["rear"] = 0.22
            P["head_yaw"] = 0.0
            r.thirst = max(0.0, r.thirst - dt * 0.15)
            if r.mode_t > r.mode_dur:
                self.back_off(d)
        elif r.mode == "eat":
            # phase 1: head down into the bowl, pick a pellet with the teeth
            if r.mode_t < 1.4:
                P["head_pitch"] = 0.55
                P["crouch"] = 1.0
                P["chew"] = 0.3
            else:
                if r.held is None:
                    self.grab_pellet(d)
                # phase 2: sit back on the haunches, hold pellet in forepaws, nibble
                P["head_pitch"] = 0.30
                P["crouch"] = 0.0
                P["sit"] = 1.0
                P["chew"] = 1.0
                if r.held is not None:
                    self.nibble(d, dt)
            r.hunger = max(0.0, r.hunger - dt * 0.08)
            if r.mode_t > r.mode_dur:
                self.drop_pellet(d)
                self.back_off(d)
        elif r.mode == "rear":
            P["rear"] = 1.0
            P["paws_up"] = 0.6
            P["head_pitch"] = -0.25
            P["head_yaw"] = 0.4 * math.sin(1.7 * t)
            if r.mode_t > r.mode_dur:
                self.set_mode("idle", r.rng.uniform(0.3, 1.0))
        elif r.mode == "groom":
            P["groom"] = 1.0
            P["chew"] = 0.4
            P["head_pitch"] = 0.3
            if r.mode_t > r.mode_dur:
                self.set_mode("idle", r.rng.uniform(0.3, 1.2))
        elif r.mode == "explore":
            P["head_yaw"] = 0.45 * math.sin(1.3 * t + r.phase * 3)
            P["head_pitch"] = 0.25 + 0.15 * math.sin(0.7 * t)
            if r.mode_t > r.mode_dur:
                self.set_mode("idle", r.rng.uniform(0.2, 0.8))
        elif r.mode == "backoff":
            r.speed = -0.06
            r.yaw_rate = 0.0
            if r.mode_t > r.mode_dur:
                self.set_mode("idle", r.rng.uniform(0.2, 0.6))

    def start(self, mode):
        r = self.r
        if mode == "drink":
            self.set_mode("drink", r.rng.uniform(3.5, 5.0))
        elif mode == "eat":
            self.set_mode("eat", r.rng.uniform(6.0, 8.0))
        elif mode == "rear":
            self.set_mode("rear", r.rng.uniform(1.8, 3.0))
        elif mode == "explore":
            self.set_mode("explore", r.rng.uniform(1.0, 2.5))
        else:
            self.set_mode("idle", 0.5)

    def back_off(self, d):
        self.set_mode("backoff", 0.6)

    # ----------------------------------------------- pellet manipulation
    def grab_pellet(self, d):
        r = self.r
        nose = r.nose(d)
        best, bd = None, 0.09
        for b in self.pellets.bodies:
            if b in self.pellets.taken:
                continue
            dist = np.linalg.norm(d.xpos[b] - nose)
            if dist < bd:
                best, bd = b, dist
        if best is None:
            return
        self.pellets.attach(d, best, r)
        r.held = best

    def nibble(self, d, dt):
        self.pellets.shrink(d, self.r.held, dt)

    def drop_pellet(self, d):
        if self.r.held is not None:
            self.pellets.release(d, self.r.held)
            self.r.held = None


class Pellets:
    """Lab-chow pellets: free rigid bodies resting in the bowl.

    While a rat holds a pellet, the pellet is carried kinematically between
    its forepaws (collisions off) and nibbled down; when released it drops
    back into the physics simulation.
    """

    def __init__(self, model, names):
        self.m = model
        self.bodies = [model.body(n).id for n in names]
        self.taken = set()
        self.holder = {}
        self.geom = {b: model.body_geomadr[b] for b in self.bodies}
        self.qadr = {b: model.jnt_qposadr[model.body_jntadr[b]] for b in self.bodies}
        self.dadr = {b: model.jnt_dofadr[model.body_jntadr[b]] for b in self.bodies}

    def attach(self, d, body, rat):
        g = self.geom[body]
        self.m.geom_contype[g] = 0
        self.m.geom_conaffinity[g] = 0
        self.taken.add(body)
        self.holder[body] = rat

    def carry(self, d):
        """Place every held pellet between its holder's forepaws."""
        for body, rat in self.holder.items():
            hl, hr = d.xpos[rat.hand["L"]], d.xpos[rat.hand["R"]]
            R = d.xmat[rat.torso].reshape(3, 3)
            p = 0.5 * (hl + hr) + R @ np.array([0.006, 0.0, 0.006])
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, (R @ euler_zyx_to_mat(math.pi / 2, 0.0, 0.0)).flatten())
            a, v = self.qadr[body], self.dadr[body]
            d.qpos[a:a + 3] = p
            d.qpos[a + 3:a + 7] = q
            d.qvel[v:v + 6] = 0

    def shrink(self, d, body, dt):
        g = self.geom[body]
        s = self.m.geom_size[g]
        s[1] = max(0.0008, s[1] - dt * 0.0006)
        s[0] = max(0.0032, s[0] - dt * 0.00012)
        self.m.geom_rbound[g] = s[0] + s[1]

    def release(self, d, body):
        self.holder.pop(body, None)
        g = self.geom[body]
        self.m.geom_contype[g] = 1
        self.m.geom_conaffinity[g] = 1
        self.taken.discard(body)
