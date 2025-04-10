def reward_function(self, x_velocity, observation, action):
    # Extract the relevant components from observation
    z_coordinate_torso = observation[0]  # Height of the torso
    velocity_x = observation[8]  # Forward velocity

    # Reward for moving forward
    forward_progress_reward = velocity_x

    # Penalty for excessive torque (energy efficiency)
    torque_penalty = 0.001 * np.sum(np.square(action))

    # Reward for maintaining a reasonable torso height
    optimal_height = 1.2
    height_penalty_temperature = 1.0
    height_penalty = np.exp(-height_penalty_temperature * np.abs(z_coordinate_torso - optimal_height))

    # Total reward
    total_reward = forward_progress_reward + height_penalty - torque_penalty

    # Reward components for debugging/information purposes
    reward_info = {
        'forward_progress_reward': forward_progress_reward,
        'height_penalty': height_penalty,
        'torque_penalty': torque_penalty
    }

    return total_reward, reward_info
