import numpy as np
import numpy as np

def custom_reward_function(obs, action, done, env):
    linear_velocity = obs.get('linear_velocity', np.zeros(3))
    desired_linear_velocity = env.desired_velocity
    linear_velocity_error = np.linalg.norm(linear_velocity - desired_linear_velocity)

    reward = -np.exp(-linear_velocity_error) - np.exp(-0.1 * np.linalg.norm(action))  # Inverse reward problem, negative reward for large velocity errors

    if done:
        reward -= 10 * (linear_velocity_error > 0.5)  # Additional penalty for not reaching the desired velocity

    return reward