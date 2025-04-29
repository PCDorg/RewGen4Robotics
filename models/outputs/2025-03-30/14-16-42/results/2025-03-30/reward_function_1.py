import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for FetchReach task.
  
    The function will compute the Euclidean distance between the current end-effector's position 
    and the goal's position. The reward for each timestep is the negative of this distance,
    encouraging the robot to minimize the distance to the goal.
  
    In addition, if the robot successfully reaches its goal (distance < 0.05), 
    a bonus reward is added. And each action taken is penalised slightly,
    incentivizing the robot to reach the goal more quickly and with fewer actions.
  
    Parameters:
    obs: Observation from the environment
    action: Action taken by the agent
    done: Flag to represent whether an episode is finished or not
    env: Instance of the environment
    """
    # Access the base environment for the goal position
    env = env.unwrapped

    # Extract the observation array from the dictionary if needed 
    obs_dict = obs if isinstance(obs, dict) else env.get_state_observation()

    # Calculate the Euclidean distance between the current position and the goal
    goal_distance = np.linalg.norm(env.goal - obs_dict['achieved_goal'])

    # Initialize the reward as the negative distance to the goal 
    reward = -goal_distance

    # Add bonus for successfully reaching the goal
    if goal_distance < 0.05:
        reward += 0.1  # Bonus reward

    # Penalize every step to encourage faster solutions.
    reward -= 0.01 * np.linalg.norm(action)

    return reward