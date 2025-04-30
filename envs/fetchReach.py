import gymnasium as gym
import gymnasium_robotics
import importlib.util
import logging
import numpy as np

# Assuming GLFW initialization might not be needed for FetchReach, but keeping if required by underlying env
# import glfw

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

class CustomFetchReachEnv(gym.Wrapper):
    """Wrapper for FetchReach to apply a custom reward function."""
    def __init__(self, env, reward_function):
        super().__init__(env)
        self.custom_reward_function = reward_function
        # FetchReach observation is a dict, store it directly
        self._last_obs_dict = None

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        self._last_obs_dict = obs_dict
        return obs_dict, info

    def step(self, action):
        current_obs_dict = self._last_obs_dict
        if current_obs_dict is None:
            raise RuntimeError("Environment reset must be called before the first step.")

        next_obs_dict, default_reward, terminated, truncated, info = self.env.step(action)
        self._last_obs_dict = next_obs_dict

        done = terminated or truncated

        # Note: The custom reward function needs to handle the dictionary observation
        try:
            # Pass the env instance itself (self.env) for access to unwrapped properties like goal
            custom_reward = self.custom_reward_function(obs=current_obs_dict, action=action, done=done, env=self.env)
            info['default_reward'] = default_reward
        except Exception as e:
            # Log the error and re-raise it
            logging.error(f"Error executing custom reward function for FetchReach: {e}")
            # logging.exception("Custom reward function traceback:") # Optional: log full traceback
            raise # Re-raise the exception

        return next_obs_dict, custom_reward, terminated, truncated, info

# Optional: Keep GLFW init if Fetch environments require it indirectly
# def initialize_glfw():
#     try:
#         if glfw.init(): return
#         else: raise Exception("GLFW could not be initialized")
#     except Exception as e:
#         if "GLFW has already been initialized" in str(e): return
#         else: logging.error(f"GLFW initialization error: {e}"); raise

def make_custom_fetch(reward_function_path, render_mode=None):
    """Creates the FetchReach environment with a custom reward function."""
    try:
        # initialize_glfw() # Uncomment if needed
        logging.info(f"Creating FetchReach-v3 environment with render_mode='{render_mode}'")
        env = gym.make("FetchReach-v3", render_mode=render_mode)
        reward_function = load_reward_function(reward_function_path)
        logging.info("Wrapping FetchReach environment with custom reward.")
        wrapped_env = CustomFetchReachEnv(env, reward_function)
        # Reset needed to initialize internal state of wrapper
        wrapped_env.reset()
        return wrapped_env
    except Exception as e:
        logging.error(f"Failed to create custom FetchReach environment: {e}")
        raise
