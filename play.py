from stable_baselines3 import PPO
from envs.antmaze import make_custom_antmaze



model_path = "/home/ken2/PCD/results/model_iteration_4.zip"  

reward_function_path = "results/reward_function_3.py"  

# Create the environment
env = make_custom_antmaze(reward_function_path)

# Load the trained model
model = PPO.load(model_path)

obs, info = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)  
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    env.render()  

env.close()
