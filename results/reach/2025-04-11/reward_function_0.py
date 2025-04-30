import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the robotics manipulation task. In this task, a robot arm
    must reach a specific goal position as quickly and accurately as possible. The reward is
    a negative function of the distance to the goal.

    Parameters
    ----------
    obs : a dictionary or array like
        Current observation of the environment. If it's a dictionary, it must include an "observation" key.

    action : array like
        The last action taken by the agent.

    done : Boolean
        Whether the task is done or not.

    env : gym.Env
        The environment where the agent is interacting.

    Returns
    -------
    reward : a single float value
        The reward for the current state, action, and next state.
    """
    
    # Fetch the unwrapped environment
    base_env = env.unwrapped

    # Extract the observation array from the current steps
    obs = obs.get('observation', obs)

    # Compute the euclidean distance between the current state and the goal
    dist_to_goal = np.linalg.norm(base_env.goal - obs['achieved_goal'])

    # Return the reward as a negative function of the distance to the goal
    reward = -dist_to_goal

    return reward