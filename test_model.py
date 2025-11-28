from stable_baselines3 import SAC, PPO
from unitree_standing_env import UnitreeWaveEnv
import threading
import time
import numpy as np
control_joints = [
    # Legs (L)
    'left_hip_pitch_joint',
    'left_knee_joint', 'left_ankle_pitch_joint',

    # Legs (R)
    'right_hip_pitch_joint',
    'right_knee_joint', 'right_ankle_pitch_joint',

    # Waist
    'waist_pitch_joint',

    # Left arm
    'left_shoulder_pitch_joint', 'left_elbow_joint',

    # Right arm
    'right_shoulder_pitch_joint',
    'right_elbow_joint',
]

# These lists should match the number of joints
upper_limits = [2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618, 2.618, 0.52, 0.52, 2.6704,
                2.2515, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558, 2.6704, 1.5882, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558]
lower_limits = [-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618, -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618, -2.618, -0.52, -0.52, -
                3.0892, -1.5882, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558, -3.0892, -2.2515, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558]


env = UnitreeWaveEnv(render_mode="human", control_joints=control_joints)
path_final = "/home/timur/git/rl_wavestand_unitree/models/ppo_unitree_standing_final"
path_temp = "/home/timur/git/rl_wavestand_unitree/models/ppo_unitree_standing_300000_steps"
path_best = "/home/timur/git/rl_wavestand_unitree/models/best/best_model"
path_stage = "/home/timur/git/rl_wavestand_unitree/backup/stage1_standing"
model = PPO.load(path_best, env=env)

obs, info = env.reset()
done = False
while 1:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()
