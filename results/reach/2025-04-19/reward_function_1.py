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
        # If episode is finished before reaching the goal, give negative reward.
        reward = -100.0
    else:
        # Encourage the agent to reach the goal as fast as possible by giving higher reward.
        reward = -velocity_error - 0.1*action_cost
    
    return reward