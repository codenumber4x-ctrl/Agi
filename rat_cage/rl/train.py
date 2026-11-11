"""Train a PPO policy for the virtual rat in the 3x cage (Stable-Baselines3).

    python rl/train.py --timesteps 12_000_000 --envs 4 --out rl/runs/ppo_rat

Saves checkpoints (policy + observation-normalisation statistics) every
--save-every steps and writes a progress CSV with, per rollout:
mean episode return, length, forward speed, food / water discovery rate,
and wall-impact penalty.  Resume with --resume <checkpoint.zip>.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize  # noqa: E402


def make_env(rank, seed):
    def _f():
        os.environ.setdefault("MUJOCO_GL", "egl")
        torch.set_num_threads(1)
        import warnings
        warnings.simplefilter("ignore")
        from rat_env import RatCageEnv
        env = RatCageEnv(seed=seed + rank)
        env.reset(seed=seed + rank)
        return env
    return _f


class ProgressLogger(BaseCallback):
    """Episode statistics -> CSV + stdout, periodic checkpoints."""

    def __init__(self, out, save_every):
        super().__init__()
        self.out = out
        self.save_every = save_every
        self.next_save = save_every
        self.ep = []
        self.t0 = time.time()
        self.f = open(os.path.join(out, "progress.csv"), "a", newline="")
        self.w = csv.writer(self.f)
        if self.f.tell() == 0:
            self.w.writerow(["timesteps", "wall_min", "episodes", "return", "length_s",
                             "v_fwd", "food_rate", "water_rate", "impact_pen", "move_rew"])
        self.acc = {}

    def _on_step(self):
        infos = self.locals["infos"]
        for k, info in enumerate(infos):
            a = self.acc.setdefault(k, dict(v=0.0, n=0, imp=0.0, mv=0.0))
            a["v"] += info.get("v_fwd", 0.0)
            a["imp"] += info.get("r_impact", 0.0)
            a["mv"] += info.get("r_move", 0.0)
            a["n"] += 1
            if "episode" in info:
                self.ep.append(dict(R=info["episode"]["r"], L=info["episode"]["l"],
                                    v=a["v"] / max(a["n"], 1), food=float(info.get("ate", 0)),
                                    water=float(info.get("drank", 0)), imp=a["imp"], mv=a["mv"]))
                self.acc[k] = dict(v=0.0, n=0, imp=0.0, mv=0.0)
        if self.num_timesteps >= self.next_save:
            self.next_save += self.save_every
            self._save(f"ckpt_{self.num_timesteps // 1000:06d}k")
        return True

    def _save(self, name):
        path = os.path.join(self.out, name)
        self.model.save(path + ".zip")
        self.model.get_env().save(path + "_vecnorm.pkl")
        print(f"  saved {path}", flush=True)

    def _on_rollout_end(self):
        if not self.ep:
            return
        e = self.ep
        row = [self.num_timesteps, round((time.time() - self.t0) / 60, 1), len(e),
               np.mean([x["R"] for x in e]), np.mean([x["L"] for x in e]) * 0.02,
               np.mean([x["v"] for x in e]), np.mean([x["food"] for x in e]),
               np.mean([x["water"] for x in e]), np.mean([x["imp"] for x in e]),
               np.mean([x["mv"] for x in e])]
        self.w.writerow([f"{v:.4g}" if isinstance(v, float) else v for v in row])
        self.f.flush()
        print(f"[{row[0]:>9d} steps | {row[1]:6.1f} min] eps={row[2]:3d} return={row[3]:8.2f} "
              f"len={row[4]:5.1f}s v_fwd={row[5]:+.3f} m/s food={row[6]:.2f} "
              f"water={row[7]:.2f} impact={row[8]:7.2f}", flush=True)
        self.ep = []

    def _on_training_end(self):
        self._save("final")
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=12_000_000)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "ppo_rat"))
    ap.add_argument("--save-every", type=int, default=500_000)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    venv = SubprocVecEnv([make_env(i, a.seed) for i in range(a.envs)], start_method="fork")
    venv = VecMonitor(venv, info_keywords=("ate", "drank"))
    gamma = 0.995                        # ~4 s horizon at 50 Hz: food/water are far apart
    if a.resume:
        venv = VecNormalize.load(a.resume.replace(".zip", "_vecnorm.pkl"), venv)
        model = PPO.load(a.resume, env=venv, device="cpu")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=gamma)
        model = PPO(
            "MlpPolicy", venv, device="cpu", seed=a.seed, verbose=0,
            n_steps=2048, batch_size=2048, n_epochs=5, learning_rate=3e-4,
            gamma=gamma, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
            max_grad_norm=0.5, target_kl=0.05,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               activation_fn=torch.nn.ELU, log_std_init=-1.0),
        )
    torch.set_num_threads(4)
    print(f"obs={venv.observation_space.shape} act={venv.action_space.shape} envs={a.envs}",
          flush=True)
    model.learn(total_timesteps=a.timesteps, callback=ProgressLogger(a.out, a.save_every),
                reset_num_timesteps=a.resume is None)
    venv.close()


if __name__ == "__main__":
    main()
