import numpy as np

def custom_reward_function(obs, action, done, env):
    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    # Get the goal position
    goal = env.unwrapped.goal

    # Get the ant's position and velocity
    position = obs[:2]
    velocity = obs[2:4]

    # Define the target proximity threshold to give bonus reward
    goal_threshold = 0.05

    # Calculate distance to the goal
    distance_to_goal = np.linalg.norm(position - goal)

    # Reward is inversely proportional to the distance to goal.
    # Subtract an amount proportional to the sum of absolute values of actions to discourage excessive movements.
    # For every step the agent is not in the goal location, it gets a small negative reward.
    epsilon = 1e-8
    reward = -distance_to_goal - 0.1 * np.abs(action).sum() - 0.01

    # If we are within the threshold, we add a large bonus and finish the episode
    if distance_to_goal < goal_threshold:
        reward += 10.0

    # If episode done but goal not reached, give a large negative reward.
    if done and distance_to_goal >= goal_threshold:
        reward -= 20.0

    return reward