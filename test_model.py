from stable_baselines3 import PPO
from unitree_standing_env import UnitreeWaveEnv
import threading
import time

env = UnitreeWaveEnv()
model = PPO.load(
    "/home/timur/git/rl_wavestand_unitree/models/best/best_model", env=env)

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    print(action)
    time.sleep(0.1)

env.close()
