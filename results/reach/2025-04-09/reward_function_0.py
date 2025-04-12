import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for a FetchReach task.
    
    This function computes a reward based on the distance between
    the end effector of the robotic arm and the target location. 
    The closer the end effector to the target, the higher the reward.
    
    Args:
        obs (dict): observation dictionary containing 'observation', 'achieved_goal', and 'desired_goal'
        action (numpy.ndarray): action taken by the agent
        done (bool): whether the episode has terminated
        env (gym.Env): gym Environment object
    
    Returns:
        float: reward value
    """
    # Extract the goal from the environment
    goal = env.unwrapped.goal
    
    # Extract the relevant observation array
    obs = obs.get('observation', obs)

    # FetchReach observations include 
    #   - the position of the gripper
    #   - the position of the target
    # Here we consider only the 3D position of the gripper and the target, 
    # and ignore other parts such as velocities which aren't necessary for the reward calculation.
    gripper_pos = obs[:3]
    target_pos = goal if goal is not None else obs[3:6]
  
    # Compute the distance between the gripper and the target
    distance = np.linalg.norm(gripper_pos - target_pos)

    # Negative distance so that the closer to the target, the more positive the reward
    # Use a small range offset (e.g., -np.inf) to enforce reaching the correct position
    reward = -distance if not done else -np.inf

    return reward