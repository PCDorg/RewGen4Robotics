def reward_function(self, x_velocity, observation, action):
    """
    The walker is a two-dimensional figure with two legs, 
    the goal is to coordinate both sets of feet, legs, and thighs 
    to move in the forward (right) direction by applying torques 
    on the six hinges connecting the six body parts.

    Parameters:
    - x_velocity: the x-coordinate velocity of the Walker2d
    - observation: the observation space of the Walker2d
    - action: the action taken by the agent in the Walker2d environment

    Returns:
    - a tuple containing the total reward and a dictionary of each individual reward component
    """
    forward_reward_weight = 1.0
    velocity_reward_weight = 0.5
    action_reward_weight = 0.1
    forward_temperature = 1.0
    velocity_temperature = 1.0
    action_temperature = 1.0

    # compute forward reward based on x_velocity
    forward_reward = np.tanh(forward_temperature * x_velocity)
    velocity_reward = np.tanh(velocity_temperature * np.abs(observation[8:10]).mean())

    # compute action reward based on action taken
    action_reward = -np.tanh(action_temperature * np.abs(action).mean())

    total_reward = (forward_reward_weight * forward_reward +
                velocity_reward_weight * velocity_reward +
                action_reward_weight * action_reward)

    reward_info = {
        "forward_reward": forward_reward * forward_reward_weight,
        "velocity_reward": total_reward * velocity_reward_weight,
        "action_reward": total_reward * action_reward_weight
    }

    return total_reward, reward_info
