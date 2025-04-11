import gymnasium as gym
import gymnasium_robotics
from results.tasks import *
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# gym.register_envs(gymnasium_robotics)

# Parallel environments
# vec_env = make_vec_env("Walker2d-v5", n_envs=1)


class CumulativeRewardCallback(BaseCallback):
    """
    Custom callback to track cumulative reward during training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

        self.cumulative_rewards = [] 
        self.cumulative_reward = 0

        self.success_threshold = 200

        self.episode_reward = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]  # reward for the current step
        self.cumulative_reward += reward  
        self.cumulative_rewards.append(self.cumulative_reward)
        self.logger.dump(self.cumulative_reward)

        # updating episodic reward
        self.episode_reward += reward 
        
        
        return True  
    
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.logger.record("cumulative_rewards", self.cumulative_reward)
        self.episode_reward = 0
        self.logger.record("reward_episode",self.episode_reward)
        pass

class TrainingManager :
    def __init__(self, env,root_dir : str,iter,reponse_id, config=None):
        self.env = env
        self.config = config or {}
        self.model = PPO("MlpPolicy", self.env, verbose=1, tensorboard_log=f"{root_dir}/walker2d_tensorboard/")
        obs = env.reset()
        self.iter_info = f"iter{iter}_responseid{reponse_id}"
        self.tb_logs_fullpath = f"{root_dir}/walker2d_tensorboard/" + self.iter_info

        self.model_save_path = f"{root_dir}/models/{self.iter_info}.zip"

        # Ensure the save directory exists
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

    def run(self ) :
        
        self.model.learn(total_timesteps= self.config.get('timesteps',1e6),tb_log_name= self.iter_info,callback=CumulativeRewardCallback())

        self.model.save(path=self.model_save_path)
        return self.model
    def get_logs_path(self):
        return self.tb_logs_fullpath