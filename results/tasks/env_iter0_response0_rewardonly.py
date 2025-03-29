def reward_function(self, x_velocity, observation, action):
    import numpy as np
    
    # Retrieve necessary components from the observation
    z_height = observation[0]
    torso_angle = observation[1]
    x_velocity_torso = observation[8]
    z_velocity_torso = observation[9]
    
    # Parameters for scaling
    height_reward_temp = 0.5
    velocity_reward_temp = 0.1
    torso_angle_penalty_temp = -1.0
    action_effort_penalty_temp = -0.01

    # Reward for maintaining a suitable height for stability
    height_reward = np.exp(-height_reward_temp * np.abs(z_height - 1.25))  # assuming 1.25 is an optimal height

    # Reward for forward velocity
    forward_velocity_reward = np.exp(velocity_reward_temp * x_velocity_torso)
    
    # Penalty for torso angle deviation to encourage upright position
    torso_angle_penalty = np.exp(torso_angle_penalty_temp * np.abs(torso_angle))
    
    # Penalty for action effort to encourage efficiency
    action_effort_penalty = np.exp(action_effort_penalty_temp * np.sum(np.square(action)))
    
    # Total Reward
    total_reward = height_reward + forward_velocity_reward + torso_angle_penalty + action_effort_penalty
    
    # Info dictionary
    reward_info = {
        'height_reward': height_reward,
        'forward_velocity_reward': forward_velocity_reward,
        'torso_angle_penalty': torso_angle_penalty,
        'action_effort_penalty': action_effort_penalty,
    }
    
    return total_reward, reward_info
