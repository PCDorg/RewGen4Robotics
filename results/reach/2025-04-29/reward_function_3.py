import numpy as np
import numpy as np

def custom_reward_function(obs, action, done, env):
    linear_velocity = obs['linear_velocity']
    desired_linear_velocity = env.unwrapped.desired_velocity
    linear_velocity_error = np.linalg.norm(linear_velocity - desired_linear_velocity)

    joint_distances = np.array([np.linalg.norm(x - y) for x, y in zip(linear_velocity[:3], desired_linear_velocity[:3])])
    joint_distance_error = np.mean(joint_distances)

    reward = -np.exp(-linear_velocity_error) - 0.1 * np.linalg.norm(action) - 0.5 * joint_distance_error

    if done:
        if np.any(linear_velocity > 0.5) or np.any(linear_velocity < -0.5):
            reward -= 10 * (linear_velocity_error > 0.5)  # Additional penalty for not reaching the desired velocity
        else:
            reward += 10  # Bonus for reaching the desired velocity

    return reward