import numpy as np
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 Robot to move forward at a desired velocity.
    Given the observation, action, done and environment variables, the function calculates a reward
    based on how well the robot moves forward at a desired velocity. 

    Args:
        obs: (OrderedDict) The current observation from the environment.
        action: (numpy.array) The current action taken by the robot.
        done: (bool) Whether the episode is done.
        env: (gym.Env) The environment instance.

    Returns:
        reward: (float) The calculated reward.
    """
    # Access environment's goal
    goal_velocity = env.unwrapped.goal

    # Extract relevant observation variables
    obs = obs.get('observation', obs)
    achieved_velocity = obs['linear_velocity']
    
    # Calculate velocity error
    velocity_error = np.abs(goal_velocity - achieved_velocity)
    
    # Calculate action cost
    action_cost = np.square(action).sum()
    
    if done:
        # At end of the episode, the reward is based on how far the bot is from the desired velocity
        reward = - 100.0 * velocity_error / goal_velocity
    else:
        # Encourage the agent to reach the goal as fastest as possible by giving higher reward for closer to goal velocities and smaller actions
        reward = - velocity_error / goal_velocity - 0.1* action_cost
    
    return reward