def reward_function(self, x_velocity, observation, action):
    import numpy as np
    
    # Decompose the observation array for relevant values
    torso_z_position = observation[0]      # z-coordinate of the torso
    torso_angle = observation[1]           # angle of the torso
    action_penalty_coefficient = 0.1       # Coefficient for energy efficiency penalty
    stability_reward_temperature = 0.5     # Temperature for stability reward transformation
    forward_velocity_temperature = 1.0     # Temperature for forward velocity reward transformation

    # Reward for moving forward: use x_velocity
    forward_velocity_reward = np.clip(x_velocity, 0, None)  # Positive x_velocity is good

    # Penalty for energy inefficient actions: using actions as torques
    energy_efficiency_penalty = action_penalty_coefficient * np.sum(np.square(action))
    
    # Stability reward: aiming to keep torso roughly upright and at a reasonable height
    torso_stability_reward = np.exp(-stability_reward_temperature * (np.abs(torso_z_position - 1) + np.abs(torso_angle)))

    # Total reward calculation
    total_reward = (np.tanh(forward_velocity_temperature * forward_velocity_reward) 
                    + torso_stability_reward 
                    - energy_efficiency_penalty)

    # Create a dictionary of individual reward components
    reward_info = {
        'forward_velocity_reward': forward_velocity_reward,
        'energy_efficiency_penalty': energy_efficiency_penalty,
        'torso_stability_reward': torso_stability_reward
    }

    return total_reward, reward_info
