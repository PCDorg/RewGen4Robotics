def reward_function(self, x_velocity, observation, action):
    # Forward velocity reward
    forward_velocity_reward = x_velocity * 2.0  # Scale the forward velocity reward
    
    # Control penalty
    control_penalty = np.sum(np.square(action)) * 0.05  # Reduce the control penalty
    
    # Reward for maintaining a certain height
    height_reward = np.exp(-(observation[0] - 1.25) ** 2 / 0.1)  # Reward for maintaining a certain height
    
    # Total reward
    reward = forward_velocity_reward + height_reward - control_penalty
    
    # Reward info
    reward_info = {
        'forward_velocity_reward': forward_velocity_reward,
        'height_reward': height_reward,
        'control_penalty': control_penalty
    }
    
    return reward, reward_info
