from stable_baselines3 import PPO 
from envs.antmaze import make_custom_antmaze 


env = make_custom_antmaze()

model = PPO("MultiInputPolicy" , env , verbose=1)

model.learn(total_timesteps=100000)

model.save("results/ppo_antmaze.zip")
print("Training completed and model saved!")

