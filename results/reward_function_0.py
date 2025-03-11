import numpy as np

def custom_reward_function(obs, action, done, env):
    # Extracting the actual observation array.
    obs = obs.get('observation', obs)
    
    # Access the goal position
    goal = env.unwrapped.goal

    # Assume the first two elements correspond to the ant's x and y positions
    ant_position = np.array(obs[:2])
    
    # The next two correspond to the velocities.
    ant_velocity = np.array(obs[2:4])

    # Compute the L2 norm (distance) to the goal.
    goal_distance = np.linalg.norm(ant_position - goal)
    
    # Reward should be higher for smaller distance to the goal.
    reward_goal = -goal_distance
    
    # Punish excessive joint torques
    reward_torque = -np.square(action).sum()

    # The ant should move towards the goal.
    reward_direction = np.dot(ant_velocity, (goal - ant_position))
    
    # Combine the three parts
    reward = reward_goal + 0.1*reward_direction + 0.01*reward_torque

    # If the episode ends before reaching the goal, apply a penalty
    if done and goal_distance > 0.05:  # 0.05 can be adjusted depending on goal size
        reward -= 10.0

    # If the ant reaches the goal, get a large bonus
    if goal_distance < 0.05:  # 0.05 can be adjusted depending on goal size
        reward += 50.0

    return reward