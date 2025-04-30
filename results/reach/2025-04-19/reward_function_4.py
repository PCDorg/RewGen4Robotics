import numpy as np
import numpy as np

def custom_reward_function(obs, action, done, env):
    # Extract the base_env to avoid wrappers
    base_env = env.unwrapped
    
    # Extract the actual observation array
    obs = np.asarray(obs)
    
    base_lin_vel = obs[13]  # Assuming base_linear_velocity is at index [13]
    desired_velocity = base_env.goal  # Access the desired velocity from env variables
    
    # Define the weights for components of the reward function
    base_lin_vel_weight = 5.0  # emphasis on achieving desired velocity
    action_weight = 0.05  # slightly penalize higher action values for efficiency

    # Compute each component of the reward
    reward_vel = - base_lin_vel_weight * np.abs(desired_velocity - base_lin_vel)  # reward is inversely proportional to the error in velocity
    reward_action = - action_weight * np.square(action).sum()  # penalty for larger action values

    # Avoid reaching the done state prematurely by only providing a finishing reward in the last timestep
    # Instead of hard-checking the velocity match, reward being in the close vicinity of the desired velocity
    reward_goal = -1.0  # base reward
    if done:
        if np.abs(desired_velocity - base_lin_vel) < 0.05:
            reward_goal = 1.0  # Full reward
        else:
            reward_goal = -1.0  # Full negative reward

    # Compute total reward
    reward = reward_vel + reward_action + reward_goal

    return reward