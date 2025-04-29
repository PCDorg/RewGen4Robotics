import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the FetchReach task.
    
    This function gives more reward when the distance to the goal is smaller and the action taken is bigger.
    This way, the agent is encouraged to reach the goal as quickly and accurately as possible.

    Parameters:
    obs : dict
        The observation dictionary received from the environment.
    action : list of float
        The list of actions taken by the robot arm.
    done : bool
        A boolean that indicates whether the episode is done.
    env :
        The environment instance where the agent performs actions.

    Returns:
    float
        The calculated reward based on the current observation and action.
    """

    # Get the actual observation array from the observation dictionary
    obs = obs.get('observation', obs)

    # Get the position of the end effector and the goal position
    effector_position = obs[:3]
    goal_position = env.unwrapped.goal

    # Calculate the Euclidean distance between the end effector and the goal position
    distance = np.linalg.norm(effector_position - goal_position)

    # Calculate the absolute value of the action taken
    speed = np.abs(action).sum()

    # Define the weight for distance and speed
    distance_weight = -1.0
    speed_weight = 1.0

    # Calculate the custom reward
    reward = (distance_weight * distance) + (speed_weight * speed)

    return reward