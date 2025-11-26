# file: train_unitree_wave.py
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from unitree_standing_env import UnitreeWaveEnv

# -----------------------------------------------
# Environment factory for vectorized training
# -----------------------------------------------


def make_env(rank=0, seed=0, render_mode="none"):
    def _init():
        env = UnitreeWaveEnv(render_mode=render_mode)
        env = Monitor(env, filename=None)  # wraps environment for SB3 logging
        # env.seed(seed + rank)
        return env
    return _init


# -----------------------------------------------
# Training
# -----------------------------------------------
if __name__ == "__main__":
    n_envs = 8
    vec_env = SubprocVecEnv([make_env(i, seed=42) for i in range(n_envs)])
    # vec_env = VecNormalize(vec_env, norm_obs=True,
    #                        norm_reward=False, clip_obs=10.0, clip_reward=10.0)

    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    # Evaluation environment
    eval_env = DummyVecEnv([lambda: Monitor(UnitreeWaveEnv())])
    eval_env = VecNormalize(eval_env, norm_obs=True,
                            norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    eval_env.training = False
    eval_env.norm_reward = False

    # Policy network configuration
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],   # actor network
            vf=[512, 512]    # critic network (larger for explained variance)
        )
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=1e-4,
        ent_coef=0.005,
        vf_coef=1.0,
        gae_lambda=0.90,
        n_epochs=20,
        # cliprange_vf=None,  # allow value function to update freely
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard/unitree_standing/",
    )

    # ---------------- Callbacks ----------------
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./models/",
        name_prefix="ppo_unitree_standing",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/",
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # ---------------- Train ----------------
    model.learn(
        total_timesteps=500_000,
        callback=[checkpoint_callback, eval_callback],
    )

    # Save final model
    model.save("./models/ppo_unitree_standing_final")
    # Save VecNormalize statistics
    vec_env.save("./models/vecnormalize.pkl")

    print("Training complete!")
