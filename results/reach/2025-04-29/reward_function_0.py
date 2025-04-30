import numpy as np
import numpy as np

def custom_reward_function(obs, action, done, env):
    # Extract the relevant components from the observation dictionary
    linear_velocity = obs.get('linear_velocity', np.zeros(3))
    desired_velocity = env.desired_velocity

    # Calculate the squared difference between the current linear velocity and the desired velocity
    velocity_error = np.sum((linear_velocity - desired_velocity) ** 2)

    # Calculate the reward based on the velocity error
    reward = -velocity_error  # Inverse reward problem, negative reward for large velocity errors

    if done:
        # If the episode is done, reward the agent for reaching the desired velocity
        reward -= velocity_error

    return reward