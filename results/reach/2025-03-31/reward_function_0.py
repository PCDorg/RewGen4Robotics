import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Reward function for the robotic arm reaching task

    :param obs: The observation from the environment
    :param action: The action taken by the agent
    :param done: Flag indicating if the episode has ended
    :param env: The environment instance
    
    :return: reward: A single float value representing the reward
    """

    # Check if the observation is a dictionary, and if so, extract the actual observation
    obs = obs.get('observation', obs)

    # Get the goal position from the base environment
    goal = env.unwrapped.goal

    # Calculate the Euclidean distance between the robotic arm gripper position and the goal
    distance_to_goal = np.linalg.norm(obs['achieved_goal'] - goal)

    # Define the reward as the negative distance to the goal. As the robot gets closer to the goal,
    # the distance decreases hence reward increases.
    reward = -distance_to_goal

    # If the robotic arm reaches the goal, grant an additional reward
    # This is to encourage the arm to reach the goal faster
    if done and distance_to_goal < 0.05:  # typically a threshold of success
        reward += 100.0  # substantial positive reward

    return reward