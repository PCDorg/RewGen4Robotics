import numpy as np
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Reward function for AntMazeCustom-v0 environment.
    
    Args:
    obs (dict): Current observation from the environment.
    action (array-like): Action taken by the agent.
    done (bool): Whether the episode is terminated.
    env: Environment object.
    
    Returns:
    float: Reward value for the current state and action.
    """
    
    # Initialize the reward value
    reward = 0
    
    # Get the current position of the ant
    position = obs['body_pos']
    
    # Calculate the Euclidean distance from the current position to the target position
    target_position = env.unwrapped.target_pos
    distance = np.linalg.norm(target_position - position)
    
    # If the episode is not terminated, reward the agent for moving towards the target
    if not done:
        # Reward the agent for reducing the distance to the target
        reward -= distance
        
    # If the episode is terminated
    if done:
        # Reward the agent for reaching the target
        if distance < 0.1:  # Consider the target reached if within 0.1 units from the target
            reward += 10
        elif distance < 0.3:  # Consider the target reached if within 0.3 units from the target
            reward += 5
        else:
            # If the agent did not reach the target, penalize it for not trying hard enough
            reward -= 10
    
    # Ensure the reward is between -1 and 1 to improve learning stability
    reward = np.clip(reward, -1, 1)
    
    # Return the calculated reward
    return reward