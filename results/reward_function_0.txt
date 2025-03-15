import numpy as np

def custom_reward_function(obs, action, done, env):
    # Check if obs is a dictionary and extract the observation array
    if type(obs) == dict:
        obs = obs.get('observation', obs)
    
    # Extract the positional and velocity information
    ant_x, ant_y = obs[:2]
    ant_vx, ant_vy = obs[2:4]
    # Extract the goal position from the environment
    goal_x, goal_y = env.unwrapped.goal
   
    # Calculate distance to the goal
    dist_to_goal = np.sqrt(np.square(ant_x - goal_x) + np.square(ant_y - goal_y))
    
    # Penalize the robot for being far from the goal
    distance_penalty = -dist_to_goal
    
    # Encourage the robot to move by rewarding it for velocity towards the goal
    velocity_reward = np.dot([ant_vx, ant_vy], [goal_x - ant_x, goal_y - ant_y])
    # Encourage the robot to be efficient by penalizing high action values
    efficiency_penalty = -np.square(action).sum()
    
    # Define constant penalty and bonus values
    done_penalty = -100.0
    goal_bonus = 100.0

    # If the episode ends up without reaching the goal, penalize it
    if done and dist_to_goal > 1.0:
        return distance_penalty + done_penalty

    # If the goal is reached, provide a large bonus
    if dist_to_goal < 1.0:
        return goal_bonus

    # Add up all the components of the reward
    reward = distance_penalty + velocity_reward + efficiency_penalty
    
    return reward