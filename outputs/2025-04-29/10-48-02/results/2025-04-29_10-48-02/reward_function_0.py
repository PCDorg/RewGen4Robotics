import numpy as np
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for controlling a Unitree Go1 quadruped robot to walk forward at a desired velocity.

    obs: dictionary of observations
    action: joint torques applied
    done: episode done (True/False)
    env: environment object

    Returns:
    reward: float value for the current state-action pair
    """
    # Extract relevant observations
    current_linear_velocity = obs.get('linear_velocity', np.zeros(3))
    desired_linear_velocity = env.desired_velocity

    # Calculate the reward
    # Reward the agent for moving forward at the desired velocity
    velocity_reward = np.sum(np.square(desired_linear_velocity - current_linear_velocity))

    # Add a penalty for non-zero joint torques (faster learning, but might make the robot less stable)
    torque_penalty = 0.1 * np.sum(np.square(action))

    # Set the reward to 0 when the episode is done (either episode end or success)
    if done:
        reward = 0
    else:
        reward = -velocity_reward - torque_penalty

    return reward