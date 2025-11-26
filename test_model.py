from stable_baselines3 import PPO
from unitree_standing_env import UnitreeWaveEnv
import threading
import time

env = UnitreeWaveEnv(render_mode="human")
model = PPO.load(
    "/home/timur/git/rl_wavestand_unitree/models/ppo_unitree_standing_final", env=env)

obs, info = env.reset()
done = False
while 1:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(action)
    env.render()

env.close()
