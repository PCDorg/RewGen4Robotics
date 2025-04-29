import numpy as np
import numpy as np

def custom_reward_function(obs, action, done, env):
    # Access the base environment
    base_env = env.unwrapped
    
    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)
    
    # Reward tuning parameters
    base_lin_vel_weight = 1.0  # Rewards are higher for staying closer to the desired velocity
    action_weight = -0.2  # Higher Negative Rewards for larger action (maximize efficiency)
    
    # Calculate individual component of rewards
    reward_vel = base_lin_vel_weight * np.exp(-10 * np.abs(desired_velocity - base_lin_vel[0]))  # Exponential decay
    reward_action = action_weight * np.square(action).sum()
    
    # Check if the task is done and if the base linear velocity is close to the desired velocity
    if done and np.allclose(base_lin_vel[0], desired_velocity, atol=0.05):
        reward_goal = 1.0 # Full reward
    else:
        reward_goal = -1.0 # Full negative reward

    # Calculate total reward as sum of all component rewards
    reward = reward_vel + reward_action + reward_goal

    return np.asscalar(reward)