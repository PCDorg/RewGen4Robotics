import numpy as np
import numpy as np

def custom_reward_function(obs, action, done, env):
    obs = obs.get('observation', obs)

    # Desired velocity (linear forward velocity)
    vel_desired = env.desired_velocity 

    # Observed velocity
    vel_obs = obs['linear_velocity']

    # We use absolute velocity difference instead of squaring it
    # This is to keep reward values in a reasonable range
    # Also negative values are removed to keep robot encouraged to take action
    vel_diff = np.abs(vel_obs - vel_desired)

    # Reward for velocity alignment with target
    vel_reward = 1.0 - vel_diff 

    return vel_reward