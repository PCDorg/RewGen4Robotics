import numpy as np
import numpy as np

def custom_reward_function(obs, action, done, env):
    # Extract the relevant components from the observation dictionary
    linear_velocity = obs.get('linear_velocity', np.zeros(3))
    angular_velocity = obs.get('angular_velocity', np.zeros(3))
    desired_linear_velocity = env.desired_velocity
    desired_angular_velocity = envdesired_angular_velocity  # Assuming the desired angular velocity is also provided

    # Calculate the squared difference between the current linear velocity and the desired linear velocity
    linear_velocity_error = np.sum((linear_velocity - desired_linear_velocity) ** 2)

    # Calculate the squared difference between the current angular velocity and the desired angular velocity
    angular_velocity_error = np.sum((angular_velocity - desired_angular_velocity) ** 2)

    # Calculate the reward based on the velocity errors
    reward = -linear_velocity_error - angular_velocity_error  # Inverse reward problem, negative reward for large velocity errors

    if done:
        # If the episode is done, reward the agent for reaching the desired velocity
        reward -= linear_velocity_error - angular_velocity_error

    return reward