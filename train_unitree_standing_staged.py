# file: train_unitree_wave.py
from stable_baselines3.common.callbacks import BaseCallback
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from unitree_standing_env import UnitreeWaveEnv
import torch
# -----------------------------------------------
# Environment factory for vectorized training
# -----------------------------------------------
control_joints = [
    # Legs (L)
    'left_knee_joint',

    # Legs (R)
    'right_knee_joint',

    # Waist
    'waist_pitch_joint'
]


class StopTrainingOnUpdateCount(BaseCallback):
    def __init__(self, max_updates, n_steps, n_envs, verbose=1):
        super().__init__(verbose)
        self.max_updates = max_updates
        self.n_steps = n_steps
        self.n_envs = n_envs

    def _on_step(self):
        # PPO update = collecting n_steps * n_envs transitions
        updates = self.model.num_timesteps // (self.n_steps * self.n_envs)

        if self.verbose:
            print(f"Updates completed: {updates}/{self.max_updates}", end="\r")

        if updates >= self.max_updates:
            print(f"\nReached {self.max_updates} updates → stopping training.")
            return False

        return True


def make_env(rank=0, seed=0, render_mode="none"):

    def _init():
        env = UnitreeWaveEnv(render_mode=render_mode,
                             control_joints=control_joints)
        env = Monitor(env, filename=None)  # wraps environment for SB3 logging
        # env.seed(seed + rank)
        return env
    return _init


# -----------------------------------------------
# Training
# -----------------------------------------------
if __name__ == "__main__":
    n_envs = 1
    vec_env = SubprocVecEnv(
        [make_env(i, seed=42) for i in range(n_envs)])
    # vec_env = VecNormalize(vec_env, norm_obs=True,
    #                        norm_reward=False, clip_obs=10.0, clip_reward=10.0)

    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    # Evaluation environment
    eval_env = DummyVecEnv([lambda: Monitor(UnitreeWaveEnv(
        render_mode="none", control_joints=control_joints))])
    eval_env = VecNormalize(eval_env, norm_obs=True,
                            norm_reward=False)
    eval_env.training = False
    eval_env.norm_reward = False

    # Policy network configuration
    policy_kwargs = dict(
        net_arch=dict(
            # Actor network: slightly deeper to capture complex action mapping
            pi=[512, 512, 256, 256],
            # Critic network: larger to accurately estimate returns (explained variance)
            vf=[1024, 512, 256],
        ),
        activation_fn=torch.nn.ReLU
    )

    # ---------------- Callbacks ----------------
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./models/",
        name_prefix="ppo_unitree_standing",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/",
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True
    )
    update_callback = StopTrainingOnUpdateCount(
        max_updates=500,     # ← STOP AFTER 500 PPO UPDATES
        n_steps=2048,        # must match your PPO config
        n_envs=n_envs
    )
    # ---------------- Train ----------------
    model = PPO.load(
        "./backup/stage0_standing.zip", env=vec_env)
    model.learn(
        total_timesteps=int(1e12),
        callback=[checkpoint_callback, eval_callback, update_callback],
    )
    # Save final model
    model.save("./models/ppo_unitree_standing_final")
    # Save VecNormalize statistics
    vec_env.save("./models/vecnormalize.pkl")

    print("Training complete!")
