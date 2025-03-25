def reward_function(self, x_velocity, observation, action):
    # Extract relevant components of the observation
    z_height = observation[0]  # z-coordinate of the torso
    torso_angle = observation[1]  # angle of the torso
    x_velocity_torso = observation[8]  # velocity of the x-coordinate of the torso
    
    # Reward for moving forward
    forward_reward = x_velocity_torso

    # Reward for maintaining a reasonable height
    target_height = 1.2
    height_penalty = -abs(z_height - target_height)
    
    # Reward for maintaining stability (keeping the torso upright)
    stability_penalty = -abs(torso_angle)

    # Minimize excessive torques to promote smooth movement
    torque_penalty_factor = 0.001
    torque_penalty = -torque_penalty_factor * np.sum(np.square(action))

    # Combining the reward components
    total_reward = forward_reward + height_penalty + stability_penalty + torque_penalty

    # Creating a dictionary for individual components
    reward_info = {
        'forward_reward': forward_reward,
        'height_penalty': height_penalty,
        'stability_penalty': stability_penalty,
        'torque_penalty': torque_penalty
    }

    return total_reward, reward_info
