import gym
import importlib
from stable_baselines3 import PPO
import logging
best_env_path = "results/tasks/env_iter0_response1.py"
# convert file path to module name
module_name = best_env_path.replace('/','.').replace('.py','').lstrip()
env_module = importlib.import_module(module_name)
try : 
    env = env_module.Walker2dEnv()
except :
    logging.info("loading the environment failed !")
    exit()
model_path = "/home/bechir/RewGen4Robotic/models/iter0_responseid1.zip"
model = PPO.load(model_path)

obs = env.reset()

for  _ in range(100000000):
    action = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
env.close()


