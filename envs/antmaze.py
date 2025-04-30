import gymnasium as gym
import gymnasium_robotics
import importlib.util
import glfw
import numpy as np
import logging

def load_reward_function(py_reward_path):
    """Dynamically load the reward function from a given .py file."""
    try:
        spec = importlib.util.spec_from_file_location("reward_module", py_reward_path)
        if spec is None:
            raise ImportError(f"Could not load spec for module at {py_reward_path}")
        reward_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reward_module)
        if not hasattr(reward_module, "custom_reward_function"):
            raise AttributeError(f"custom_reward_function not found in the reward module: {py_reward_path}")
        logging.info(f"Successfully loaded custom_reward_function from {py_reward_path}")
        return reward_module.custom_reward_function
    except Exception as e:
        logging.error(f"Error loading reward function from {py_reward_path}: {e}")
        raise

class CustomAntMazeEnv(gym.Wrapper):
    def __init__(self, env, reward_function):
        super().__init__(env)
        self.custom_reward_function = reward_function
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        current_obs = self._last_obs
        if current_obs is None:
            logging.warning("Reset not called before step? Using current observation from env state.")
            raise RuntimeError("Environment reset must be called before the first step.")

        next_obs, default_reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = next_obs

        done = terminated or truncated

        try:
            custom_reward = self.custom_reward_function(obs=current_obs, action=action, done=done, env=self.env)
            info['default_reward'] = default_reward
        except Exception as e:
            logging.error(f"Error executing custom reward function for AntMaze: {e}")
            raise

        return next_obs, custom_reward, terminated, truncated, info

def initialize_glfw():
    try:
        if glfw.init():
            return
        else:
            raise Exception("GLFW could not be initialized")
    except Exception as e:
        if "GLFW has already been initialized" in str(e):
            return
        else:
            logging.error(f"GLFW initialization error: {e}")
            raise

def make_custom_antmaze(reward_function_path, render_mode=None):
    """Creates the AntMaze environment with the custom reward function loaded from reward_function_path."""
    try:
        initialize_glfw()
        env = gym.make("AntMaze_UMaze-v5", render_mode=render_mode)
        reward_function = load_reward_function(reward_function_path)
        logging.info("Wrapping AntMaze environment with custom reward.")
        wrapped_env = CustomAntMazeEnv(env, reward_function)
        wrapped_env.reset()
        return wrapped_env
    except Exception as e:
        logging.error(f"Failed to create custom AntMaze environment: {e}")
        raise
