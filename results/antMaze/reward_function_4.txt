import numpy as np

def custom_reward_function(obs, action, done, env):
    # Check if observation is a dictionary and extract the actual observation
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Assume that the first two elements correspond to the ant's x and y positions,
    # and the next two correspond to the velocities.
    x, y, vx, vy = obs[:4]

    # Access the goal position  
    goal_position = env.unwrapped.goal

    # Compute the distance to the goal
    delta_x, delta_y = goal_position[0] - x, goal_position[1] - y
    distance_to_goal = np.sqrt(delta_x**2 + delta_y**2)

    # Encourage the ant to move toward the target:
    # Compute the velocity in the direction of the goal
    velocity_towards_goal = ((vx * delta_x) + (vy * delta_y)) / (distance_to_goal + 1e-9)

    # Penalize excessive or inefficient joint movements:
    # Compute the sum of absolute torques - as a proxy for energy expenditure
    energy_expenditure = np.sum(np.abs(action))

    # Provide a large bonus if the ant reaches the goal:
    # Define the threshold for being 'close' to the goal
    closeness_threshold = 0.05
    goal_reached_bonus = 100.0 if distance_to_goal < closeness_threshold else 0.0

    # Penalize if it finishes before reaching the goal
    if done and distance_to_goal >= closeness_threshold:
        goal_reached_bonus -= 100.0

    # Construct the final reward
    # The coefficients are hyperparameters and can be tuned for better performance.
    reward = 2.0 * velocity_towards_goal - 0.1 * energy_expenditure + goal_reached_bonus

    return float(reward)