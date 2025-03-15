import numpy as np

def custom_reward_function(obs, action, done, env):
    # Get the underlying environment (unwrap if necessary)
    unwrapped_env = env.unwrapped

    # Extract the actual observation array
    obs = obs.get('observation', obs)

    # Assume the first two elements of the observation array are the x and y positions
    ant_position = np.array(obs[:2])

    # Assume the next two elements of the observation are the x and y velocities
    ant_velocity = np.array(obs[2:4])

    # Access the goal position
    goal_position = unwrapped_env.goal

    # Calculate the distance to the goal
    dist_to_goal = np.linalg.norm(ant_position - goal_position)

    # Calculate the speed of the ant
    speed = np.linalg.norm(ant_velocity)

    # Get the absolute value of the action to assess the magnitude of torques applied
    torque = np.abs(action)

    # Set up constants
    dist_weight = -1.0
    speed_weight = 1.0
    torque_weight = -0.1
    goal_bonus = 1000.0
    timeout_penalty = -1000.0

    # Calculate the reward as a weighted sum of distance, speed, and torque.
    reward = dist_weight * dist_to_goal + speed_weight * speed + torque_weight * np.sum(torque)

    # If the goal is achieved, give a bonus and complete the episode
    if np.isclose(dist_to_goal, 0.0, atol=0.1):
        reward += goal_bonus
        done = True

    # If the episode ends without reaching the goal, apply a penalty
    if done and not np.isclose(dist_to_goal, 0.0, atol=0.1):
        reward += timeout_penalty

    return reward