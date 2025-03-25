def reward_function(self, x_velocity, observation, action):
    # Unpack the observation
    z_coordinate = observation[0]
    torso_angle = observation[1]
    x_velocity = observation[8]

    # Reward for forward velocity
    velocity_reward_temp = 1.0
    velocity_reward = np.exp(velocity_reward_temp * x_velocity)

    # Penalty for excessive torques (actions)
    torque_penalty_temp = 0.01
    torque_penalty = -torque_penalty_temp * np.linalg.norm(action)

    # Penalty for torso angle deviation from upright position
    torso_angle_penalty_temp = 0.5
    torso_angle_penalty = -np.exp(torso_angle_penalty_temp * np.abs(torso_angle))

    # Total reward
    total_reward = velocity_reward + torque_penalty + torso_angle_penalty

    # Reward components for debugging or analysis
    reward_info = {
        'velocity_reward': velocity_reward,
        'torque_penalty': torque_penalty,
        'torso_angle_penalty': torso_angle_penalty,
    }

    return total_reward, reward_info
