import numpy as np
import numpy as np

def custom_reward_function(obs, action, done, env):
    # Get the desired velocity
    desired_velocity = env.desired_velocity

    # Calculate the difference between the current and desired velocities
    velocity_diff = obs['linear_velocity'] - desired_velocity

    # Calculate the reward based on the velocity difference
    action_reward = -np.sum(np.abs(velocity_diff))

    # Calculate a penalty for done (episode is finished)
    if done:
        action_reward -= 10.0

    return action_reward