from stable_baselines3 import SAC
from envs.fetchReach import make_custom_fetch


model_path = "/home/ken2/PCD/results/reach/2025-04-11/sac_model_4.zip"  

reward_function_path = "/home/ken2/PCD/results/reach/2025-04-11/reward_function_4.py"  

# Create the environment
env = make_custom_fetch(reward_function_path)

# Load the trained model
model = SAC.load(model_path)

obs, info = env.reset()
done = False
#play for many episodes 
for _ in range(1000):
    action, _ = model.predict(obs)  
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    env.render()
  

env.close()
