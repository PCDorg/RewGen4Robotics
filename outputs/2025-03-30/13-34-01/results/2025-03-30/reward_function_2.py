import numpy as np

def custom_reward_function(obs, action, done, env):
    # Access base/unwrapped environment
    unwrapped_env = env.unwrapped

    # Access the goal position
    goal_position = unwrapped_env.goal

    # Extract the observation data
    obs_data = np.array(obs.get('observation', obs))

    # Extract the positional and velocity information
    ant_position = obs_data[:2]
    ant_velocity = obs_data[2:4]

    # Distance to goal and velocity towards goal
    dist_to_goal = np.linalg.norm(goal_position - ant_position) # Euclidean distance
    velocity_towards_goal = np.dot(goal_position - ant_position, ant_velocity)

    # Calculate reward as a mix of achieving goal, staying alive, and efficiency
    reward_goal = -dist_to_goal  # Reward is more when closer to goal
    reward_efficiency = -np.square(action).sum()  # Penalize inefficient use of joints
    reward_living = 1.0  # baseline reward for staying alive

    # Large bonus for reaching goal, large penalty for ending too soon
    if done:
        if dist_to_goal < 0.1:  # Goal threshold, can be adjusted
            reward_goal += 100.0  # Large bonus!
        else:
            reward_living -= 100.0  # Large penalty for ending the episode without the goal reached

    # Weighted combination of goals
    reward = reward_goal + 0.1 * reward_efficiency + reward_living

    return reward