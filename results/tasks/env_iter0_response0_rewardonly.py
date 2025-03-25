def reward_function(self, x_velocity, observation, action):
    # Extracting relevant parts of the observation
    z_torso = observation[0]  # z-coordinate of the torso
    x_torso_velocity = observation[8]  # velocity of the x-coordinate of the torso
    
    # Setting temperature variables for transformations
    forward_velocity_temp = 0.5
    torso_height_temp = 0.1
    action_efficiency_temp = 0.01
    
    # Reward components
    forward_velocity_reward = np.tanh(forward_velocity_temp * x_velocity)  # Encourage moving forward
    torso_height_reward = np.exp(-torso_height_temp * np.abs(z_torso - 1.2))  # Encourage maintaining a torso height of roughly 1.2 meters for stability
    action_efficiency_reward = -action_efficiency_temp * np.linalg.norm(action)  # Penalize excessive torque to promote efficient movement
    
    # Calculate total reward
    total_reward = forward_velocity_reward + torso_height_reward + action_efficiency_reward
    
    # Return both total reward and individual components
    reward_info = {
        "forward_velocity_reward": forward_velocity_reward,
        "torso_height_reward": torso_height_reward,
        "action_efficiency_reward": action_efficiency_reward
    }
    
    return total_reward, reward_info
