import numpy as np

def custom_reward_function(obs, action, done, env):
    # Ensure 'obs' is a numpy array, regardless of whether
    # the observations from the environment are dictionaries or arrays
    if isinstance(obs, dict):
        obs = obs.get('observation', obs)
    obs = np.asarray(obs)
    
    # Extract position and velocity from the observation
    pos = obs[:2]
    vel = obs[2:4]
    
    # Access goal position
    goal = env.unwrapped.goal
    
    # Calculate distance to the goal and the ant's speed
    dist_to_goal = np.linalg.norm(pos - goal)
    speed = np.linalg.norm(vel)
    
    # Penalize large joint movements
    joint_penalty = 0.1 * np.sum(np.square(action))
    
    # Main reward is the negative distance to the goal (so closer = higher reward),
    # but we also encourage speed
    reward = -dist_to_goal + 0.1 * speed - joint_penalty
    
    # Large bonus if the ant reaches the goal (within some threshold)
    if dist_to_goal < 0.5:
        reward += 100

    # If episode is done but the ant didn't reach the goal, apply a penalty
    if done and dist_to_goal > 0.5:
        reward -= 50
        
    return reward