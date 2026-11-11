"""Evaluate a trained rat policy and render a video in the decorated 3x cage.

    MUJOCO_GL=egl python rl/evaluate.py rl/runs/ppo_rat/final.zip --seconds 30 \
        --out media/rl_rat.mp4 --episodes 3

The policy runs in the physics-only training environment; the recorded joint
states are replayed in the visual model (bars, bedding, bottle, bowl...),
which has exactly the same degrees of freedom.
"""

import argparse
import os
import pickle
import sys
import warnings

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.simplefilter("ignore")

import rat_env  # noqa: E402
import build_scene as scene  # noqa: E402


def load_policy(path):
    from stable_baselines3 import PPO
    model = PPO.load(path, device="cpu")
    with open(path.replace(".zip", "_vecnorm.pkl"), "rb") as f:
        vn = pickle.load(f)
    mean, var, clip = vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs

    def act(obs):
        o = np.clip((obs - mean) / np.sqrt(var + vn.epsilon), -clip, clip)
        a, _ = model.predict(o, deterministic=True)
        return a
    return act


def rollout(policy, seconds, seed, spawn_near):
    env = rat_env.RatCageEnv(seed=seed, spawn_near=spawn_near, episode_seconds=seconds)
    obs, _ = env.reset(seed=seed)
    frames, stats = [], dict(R=0.0, dist=0.0, food_t=None, water_t=None, hard_hits=0, fell=False)
    start = env.d.xpos[env.torso][:2].copy()
    path = [start.copy()]
    for t in range(env.max_steps):
        obs, r, term, trunc, info = env.step(policy(obs))
        stats["R"] += r
        frames.append(env.d.qpos.copy())
        path.append(env.d.xpos[env.torso][:2].copy())
        if info["r_food"] > 0:
            stats["food_t"] = round(t * rat_env.CTRL_DT, 1)
        if info["r_water"] > 0:
            stats["water_t"] = round(t * rat_env.CTRL_DT, 1)
        stats["hard_hits"] += info["r_impact"] < 0
        if term:
            stats["fell"] = True
            break
        if trunc:
            break
    path = np.array(path)
    stats["dist"] = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    stats["net_disp"] = float(np.linalg.norm(path[-1] - path[0]))
    stats["seconds"] = round(len(frames) * rat_env.CTRL_DT, 1)
    return np.array(frames), path, stats


def render(qpos_seq, out, width=960, height=544, fps=25):
    import imageio.v2 as imageio
    m = rat_env.build_model(visual=True)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height, width)
    opt = mujoco.MjvOption()
    opt.geomgroup[2] = 0
    opt.geomgroup[3] = 0
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rat1/torso")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    step = max(1, int(round(1 / fps / rat_env.CTRL_DT)))
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=8,
                           pixelformat="yuv420p")
    look = None
    stills = []
    for i in range(0, len(qpos_seq), step):
        d.qpos[:] = qpos_seq[i]
        mujoco.mj_forward(m, d)
        p = d.xpos[torso].copy()
        tgt = np.array([p[0], p[1], 0.04])
        look = tgt if look is None else look + 0.15 * (tgt - look)
        cam.lookat[:] = look
        # chase camera from the cage-centre side, slowly orbiting
        cam.distance = 0.55
        cam.elevation = -32
        cam.azimuth = np.degrees(np.arctan2(p[1], p[0])) + 20 * np.sin(i * 0.004)
        r.update_scene(d, cam, opt)
        img = r.render()
        w.append_data(img)
        if i % (step * fps * 5) == 0:
            stills.append(img)
    w.close()
    return stills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--spawn-near", type=float, default=0.0)
    ap.add_argument("--out", default=None, help="mp4 of the best episode")
    a = ap.parse_args()
    policy = load_policy(a.checkpoint)
    best = None
    for ep in range(a.episodes):
        q, path, st = rollout(policy, a.seconds, a.seed + ep, a.spawn_near)
        print(f"episode {ep}: {st}", flush=True)
        if best is None or st["R"] > best[2]["R"]:
            best = (q, path, st)
    if a.out:
        stills = render(best[0], a.out)
        import imageio.v2 as imageio
        base = os.path.splitext(a.out)[0]
        imageio.imwrite(base + "_frames.png", np.concatenate(stills[:4], 1)[::2, ::2])
        np.save(base + "_path.npy", best[1])
        print("wrote", a.out)


if __name__ == "__main__":
    main()
