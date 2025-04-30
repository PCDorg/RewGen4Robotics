import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the FetchReach robotics manipulation task. The reward is
    primarily based on the euclidean distance to the goal. Furthermore, to promote efficiency,
    the agent is penalized proportional to the sum of the squared action inputs.

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

    # Parse observation
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Constants
    ctrl_cost_weight = 0.1   # Increased the weight to discourage sway actions.
    goal = base_env.goal

    # Compute the euclidean distance between the current position and the goal and square it
    dist_to_goal = np.square(np.linalg.norm(goal - obs[:3]))

    # Control cost (penalize extensive action inputs to encourage efficiency)
    ctrl_cost = ctrl_cost_weight * np.square(action).sum()

    # The reward is the inverse of the distance to the goal, but also considering the control cost
    reward = - dist_to_goal - ctrl_cost

    return reward