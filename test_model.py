from stable_baselines3 import PPO
from unitree_standing_env import UnitreeWaveEnv

env = UnitreeWaveEnv()
model = PPO.load(
    "/home/timur/git/rl_wavestand_unitree/models/ppo_unitree_standing_final", env=env)

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
