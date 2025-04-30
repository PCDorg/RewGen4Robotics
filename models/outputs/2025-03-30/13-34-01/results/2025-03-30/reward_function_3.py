import numpy as np

def custom_reward_function(obs, action, done, env):
    # If observation is a dictionary, get the actual observation array 
    obs = obs.get('observation', obs)

    # The first two elements correspond to the ant's x and y positions.
    ant_position = np.array(obs[:2])
    # The next two correspond to the x and y velocities.
    ant_velocity = np.array(obs[2:4])

    # The absolute target position
    target_position = np.array(env.unwrapped.goal)

    # Compute the distance from the ant to the goal
    distance_to_goal = np.linalg.norm(target_position - ant_position)

    # Compute penalty for moves that are away from the target
    away_penalty = - 0.1 * distance_to_goal
    
    # Compute a bonus for moves that result in progress toward the destination
    movement_bonus = 0.5 * np.dot(ant_velocity, target_position - ant_position)

    # Compute the penalty for excessive joint movements
    joint_movement_penalty = -0.01 * np.sum(np.square(action[:8]))

    # Check if the episode has ended
    if done:
        # If it has reached the goal position provide a large bonus
        if distance_to_goal < 0.1:
            completion_bonus = 1000
        else:
            # Otherwise give a penalty
            completion_penalty = -500
    else:
        completion_bonus = 0
        completion_penalty = 0

    # The final reward
    reward = away_penalty + movement_bonus + joint_movement_penalty + completion_bonus + completion_penalty

    return reward