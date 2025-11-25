# file: train_unitree_wave.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from unitree_standing_env import UnitreeWaveEnv


def make_env(rank=0):
    def _init():
        env = UnitreeWaveEnv(
            model_path="g1/scene_29dof.xml")
        return env
    return _init


if __name__ == "__main__":
    n_envs = 8
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    model = PPO("MlpPolicy", vec_env,
                verbose=1,
                n_steps=2048,
                batch_size=64,
                learning_rate=5e-5,  # 3e-4
                ent_coef=0.01,
                tensorboard_log="./tensorboard/unitree_standing/")

    # Checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, save_path="./models/", name_prefix="ppo_unitree_standing")
    # Optional EvalCallback: evaluate every N steps
    eval_env = UnitreeWaveEnv(
        model_path="g1/scene_29dof.xml")
    eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best/',
                                 log_path='./logs/', eval_freq=20_000, n_eval_episodes=5, deterministic=True)

    model.learn(total_timesteps=500_000, callback=[
                checkpoint_callback, eval_callback])
    model.save("./models/ppo_unitree_standing_final")
