import gymnasium as gym
import gymnasium_robotics
from results.tasks import *

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# gym.register_envs(gymnasium_robotics)

# Parallel environments
# vec_env = make_vec_env("Walker2d-v5", n_envs=1)

class TrainingManager :
    def __init__(self, env, config=None):
        self.env = env
        self.config = config or {}
        self.model = PPO("MlpPolicy", self.env, verbose=1)
        obs = env.reset()

    def run(self) :
        self.model.learn(total_timesteps= self.config.get('timesteps',1e6))

        return self.model
