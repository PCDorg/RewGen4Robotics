
import gymnasium as gym
import gymnasium_robotics 
import importlib.util

def load_reward_function(py_reward_path):
    """Dynamically load the reward function from a given .py file."""
    spec = importlib.util.spec_from_file_location("reward_module", py_reward_path)
    reward_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward_module)
    if not hasattr(reward_module, "custom_reward_function"):
        raise AttributeError("custom_reward_function not found in the reward module")
    return reward_module.custom_reward_function

class CustomAntMazeEnv(gym.Wrapper):
    def __init__(self, env, reward_function):
        super().__init__(env)
        self.custom_reward_function = reward_function

    def step(self, action):
        next_obs, default_reward, terminated, truncated, info = self.env.step(action)
        custom_reward = self.custom_reward_function(next_obs, action, terminated, self.env)
        return next_obs, custom_reward, terminated, truncated, info

def make_custom_antmaze(py_reward_path):
    """Creates the AntMaze environment with the custom reward function loaded from py_reward_path."""
    env = gym.make("AntMaze_UMaze-v5", render_mode="human")
    reward_function = load_reward_function(py_reward_path)
    return CustomAntMazeEnv(env, reward_function)
