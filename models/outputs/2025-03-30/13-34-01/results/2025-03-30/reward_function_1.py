import numpy as np

def custom_reward_function(obs, action, done, env):
    # Get the actual observation from dictionary (if applicable)
    obs = obs.get('observation', obs)
    
    # Assume the first and second elements of the observation are x and y position, respectively
    x_pos, y_pos = obs[0], obs[1]
    
    # The next two elements are the velocities
    x_vel, y_vel = obs[2], obs[3]
    
    # Get the goal position
    goal_x, goal_y = env.unwrapped.goal
    
    # Compute distance from current position to the goal
    dist_to_goal = np.sqrt((goal_x - x_pos) ** 2 + (goal_y - y_pos) ** 2)
    
    # Compute the magnitude of the velocity vector
    velocity_magnitude = np.sqrt(x_vel ** 2 + y_vel ** 2)
    
    # Penalize inefficiency in joint movements based on the torques applied
    inefficiency_penalty = np.sum(np.abs(action))
    
    # Reward is based on distance to goal and velocity in the direction of the goal
    distance_reward = -dist_to_goal
    velocity_reward = np.dot([x_vel, y_vel], [goal_x - x_pos, goal_y - y_pos]) / (dist_to_goal + 1e-9)
  
    # Calculate reward
    reward = distance_reward + velocity_reward - 0.1 * inefficiency_penalty
    
    # If done, but not reached the goal, give a big penalty
    if done and dist_to_goal > 1:
        reward -= 50
        
    # If reached the goal, give a big reward
    if dist_to_goal < 1:
        reward += 100
    
    return reward