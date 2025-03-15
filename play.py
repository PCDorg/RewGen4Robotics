from stable_baselines3 import PPO
from envs.antmaze import make_custom_antmaze
import time 

# Path to the saved model
model_path = "/home/ken2/PCD/results/model_iteration_4.zip"  # Change this to your actual model path

# Path to the reward function used during training
reward_function_path = "results/reward_function_4.py"  # Change this to your actual reward function path

# Create the environment
env = make_custom_antmaze(reward_function_path)

# Load the trained model
model = PPO.load(model_path)

obs, info = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)  # Get action from the model
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    env.render()  # Render the environment
    time.sleep(0.05)  # Add a small delay for smoother visualization

env.close()
