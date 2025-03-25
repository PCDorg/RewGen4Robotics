def reward_function(self, x_velocity, observation, action):
    # Extract necessary components from the observation
    z_coordinate, torso_angle = observation[0], observation[1]
    velocities = observation[8:11]  # x_velocity, z_velocity, torso_angular_velocity
    
    # Parameters for transformations
    velocity_temperature = 1.0
    stability_temperature = 1.0
    efficiency_temperature = 1.0

    # Velocity Reward: Encourage forward movement
    velocity_reward = np.exp(x_velocity / velocity_temperature)
    
    # Stability Reward: Encourage maintaining a certain height and limiting torso angle to ensure stability
    stability_reward = np.exp(-(np.abs(z_coordinate - 1.2) + np.abs(torso_angle)) / stability_temperature)
    
    # Efficiency Reward: Encourage minimal action magnitude for energy efficiency
    efficiency_reward = np.exp(-np.linalg.norm(action) / efficiency_temperature)
    
    # Total Reward
    total_reward = velocity_reward + stability_reward + efficiency_reward
    
    # Reward Info
    reward_info = {
        "velocity_reward": velocity_reward,
        "stability_reward": stability_reward,
        "efficiency_reward": efficiency_reward,
        "total_reward": total_reward
    }
    
    return total_reward, reward_info
