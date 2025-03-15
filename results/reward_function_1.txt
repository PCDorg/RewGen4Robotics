import numpy as np

def custom_reward_function(obs, action, done, env):
    # Get the observation
    obs = obs.get('observation', obs)
    
    # Extract the ant's position and velocity
    ant_x, ant_y, v_x, v_y = obs[:4]
    ant_position = np.array([ant_x, ant_y])
    velocity = np.array([v_x, v_y])
    
    # Access the goal
    goal = np.array(env.unwrapped.goal)
    
    # Compute the distance to the goal and the speed towards the goal
    dist_to_goal = np.linalg.norm(goal - ant_position)
    speed_towards_goal = np.dot(velocity, (goal - ant_position) / dist_to_goal) if dist_to_goal > 0 else 0
    
    # Compute the inefficiency cost
    inefficiency_cost = np.sum(np.abs(action)) / len(action)
    
    # Compute the reward
    reward = -dist_to_goal + speed_towards_goal - inefficiency_cost
    
    # Add a bonus if we have reached the goal
    if np.linalg.norm(goal - ant_position) < 0.5:
        reward += 1000
    # Add a penalty if the episode has ended before reaching the goal
    if done and dist_to_goal > 0.5:
        reward -= 1000
    
    return reward