import numpy as np

def custom_reward_function(obs, action, done, env):
    # Get the actual observation array if the input obs is a dictionary
    obs = obs.get('observation', obs)
    
    # Extract ant's pos and vel information from the observation
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]

    # Get the goal position
    goal_x, goal_y = env.unwrapped.goal

    # Calculate the distance to the goal
    dist_to_goal = np.sqrt((x_pos - goal_x)**2 + (y_pos - goal_y)**2)

    # Define a reward for moving toward the goal
    reward_toward_goal = -dist_to_goal

    # Also, reward the agent for moving in the goal direction
    goal_dir = np.arctan2(goal_y - y_pos, goal_x - x_pos)
    current_vel_dir = np.arctan2(y_vel, x_vel)
    reward_goal_dir = -np.abs(goal_dir - current_vel_dir)

    # Penalize large actions (encourage efficient movements)
    action_cost = -np.sum(np.abs(action))

    # Combine the rewards
    reward = reward_toward_goal + reward_goal_dir + action_cost

    # If the episode is done and goal not reached, apply a large penalty
    if done and dist_to_goal > 1.0:  # Assuming a threshold of 1.0 for being "at the goal"
        reward -= 1000

    # If the ant is near the goal, give a large bonus
    if dist_to_goal < 1.0:
        reward += 1000

    return reward