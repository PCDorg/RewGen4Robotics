import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the robotics manipulation task. In this task, a robot arm
    must reach a specific goal position as quickly and accurately as possible. The reward is
    a function of the distance to the goal and takes into consideration the action taken
    by the agent.

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

    # Parse observation depending on if it is array-like or dictionary
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Environment specific constants
    ctrl_cost_weight = 1e-3     # Increase control weight to incentivize agent to take fewer, more precise moves
    goal = base_env.goal

    # Compute the normalized euclidean distance between the current state and the goal and square it
    dist_to_goal = np.square(np.linalg.norm(goal - obs[:3]))

    # Control cost
    ctrl_cost = ctrl_cost_weight * np.square(action).sum()

    # Reward is now a function of normalized distance reward and a control cost
    reward = 1 - dist_to_goal - ctrl_cost

    return reward