import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the FetchReach robotics manipulation task. The reward is
    based on the euclidean distance to the goal and penalizes extensive action inputs
    while giving more incentive to reaching the goal.

    Parameters
    ----------
    obs : a dictionary or array like
        Current observation of the environment. If it's a dictionary, it needs to include an "observation" key.

    action : array like
        The agent's last action.

    done : Boolean
        Indicates whether the task is done or not.

    env : gym.Env
        The environment where the agent is interacting.

    Returns
    -------
    reward : float
        The reward for the current state, action, and next state.
    """
    # Fetch the unwrapped environment
    base_env = env.unwrapped

    # Extract the actual observation array if obs is a dictionary
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Constants
    ctrl_cost_weight = 0.01  # Decreasing the weight from 0.1 to 0.01 to give priority to reaching the goal.
    goal = base_env.goal

    # Compute the euclidean distance between the current position and the goal
    dist_to_goal = np.linalg.norm(goal - obs[:3])

    # Control cost (penalize extensive action inputs to encourage efficiency)
    ctrl_cost = ctrl_cost_weight * np.square(action).sum()

    # The reward is the inverse of the distance to the goal, but also considering the control cost
    reward = - dist_to_goal - ctrl_cost

    return reward