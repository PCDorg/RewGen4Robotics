import numpy as np
import numpy as np

# We define our custom reward function which takes 
# observation, action, done flag and environment as inputs
def custom_reward_function(obs, action, done, env):

    # If it's a dict, fetch the actual obs array
    obs = obs.get('observation', obs)

    # Velocity reward
    # Observed velocity. We assume obs['linear_velocity'] returns the forward velocity of Go1 robot
    vel_obs = obs['linear_velocity']

    # Desired velocity is expected to be accessed via env.desired_velocity
    vel_desired = env.desired_velocity

    # We define velocity reward as negative difference between desired and actual velocities
    # We square this difference to give high priority to huge difference
    vel_reward = -np.sum(np.square(vel_obs - vel_desired))

    # Stand reward
    # To validate if Go1 robot is standing, we check the base of robot
    # If the z-component of base position is less than 0.8 for example,
    # Go1 robot is not standing and we give a negative reward for this
    # We assume obs['base_pos_z'] returns the z-component of base position
    stand_reward = 0 if obs['base_pos_z'] > 0.8 else -1

    # Control reward
    # We use hinge loss for control reward. If joint torques exceed a threshold,
    # penalty will boost. We assume action is an array of joint torques.
    # x if x < 0 else 0 for each element in actions.
    ctrl_reward = -np.sum(np.square(np.maximum(0, np.abs(action) - 0.2)))

    # We combine all these rewards with different weights
    # It is important to tune these weights based on the complexity and priorities of tasks
    reward = 0.3 * vel_reward + 0.2 * stand_reward + 0.5 * ctrl_reward

    # Stable Baselines3 expects a single float that represents the reward.
    return reward