import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from envs.antmaze import make_custom_antmaze

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_results_folder = "/home/ken2/PCD/results"
reward_function_path = os.path.join(base_results_folder, "reward_function_3.py")  # 4th reward function (0-based index)
model_path = os.path.join(base_results_folder, "trained_model_7.zip")  # 4th model checkpoint
new_model_path = os.path.join(base_results_folder, "trained_model_8.zip")  # New model save path

if not os.path.exists(model_path):
    logging.error(f"Model file not found: {model_path}")
    exit(1)

if not os.path.exists(reward_function_path):
    logging.error(f"Reward function file not found: {reward_function_path}")
    exit(1)

# Load environment with the custom reward function
logging.info("Creating environment with custom reward function...")
env = make_custom_antmaze(reward_function_path)
logging.info("Environment successfully created.")

# Load the previously trained model
logging.info(f"Loading model from {model_path}...")
model = PPO.load(model_path, env=env)  # Resume training with the same environment
logging.info("Model successfully loaded.")

total_timesteps = 5000000  # Adjust as needed
logging.info(f"Starting training for {total_timesteps} timesteps...")

# Custom callback to log training progress and average rewards from rollout
class LoggingCallback(BaseCallback):
    def __init__(self, log_interval=1000, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            logging.info(f"Training step {self.num_timesteps} reached.")
        return True

    def _on_rollout_end(self) -> None:
        # Access rollout buffer rewards if available
        rollout_buffer = self.locals.get("rollout_buffer", None)
        if rollout_buffer is not None and hasattr(rollout_buffer, "rewards"):
            rewards = rollout_buffer.rewards
            avg_rollout_reward = rewards.mean() if rewards.size > 0 else 0
            logging.info(f"Rollout end: Average reward = {avg_rollout_reward:.2f}")
        else:
            logging.info("Rollout end: No rewards found in rollout buffer.")

logger_callback = LoggingCallback(log_interval=1000)

# Resume training with the callback
model.learn(total_timesteps=total_timesteps, callback=logger_callback)

# Save the updated model
model.save(new_model_path)
logging.info(f"Training complete. Model saved at {new_model_path}")
