import numpy as np
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    This function computes the reward for the Go1MujocoCustom-v0 task.
    The task is to make the Go1 robot walk forward at a desired velocity.

    obs: Dict[str, np.ndarray]
        Dictionary of observations, including 'position', 'velocity',
        'desired_velocity', etc.
    action: np.ndarray
        The current action (torque values) taken by the robot.
    done: bool
        Whether or not the episode has ended.
    env: OpenAI Gym Environment
        The current environment instance.

    Returns
    -------
    float
        The computed reward for the current step.
    """

    # Extract the key observations
    actual_velocity = obs.get('velocity', None)
    desired_velocity = obs.get('desired_velocity', env.unwrapped.desired_velocity)

    # Initialize reward as zero
    reward = 0.0
  
    # Reward the robot for achieving a velocity close to the desired_velocity
    # Use negative of the L1 distance between actual and desired velocities 
    # Note: Multiply by -1 because the agent needs to minimize this value
    # Change to L1 distance, to emphasize on getting closer to the desired vector rather than its accumulated distance
    velocity_reward = -np.abs(actual_velocity - desired_velocity).sum()

    # Provide a reward for forward motion while penalising any angular deviation
    # Here, the x-axis might represent the forward direction
    # Modify appropriately based on the observation space definition
    forward_reward = actual_velocity[0] - np.abs(actual_velocity[1]) * 0.1  # tone down the penalty factor to balance out progress and the reward from velocity achievement

    # Penalize high torque actions to encourage energy efficiency
    # Penalize more heavily to ensure a balance between velocity achievement and energy efficiency
    # Use a square root instead of a square to lessen the impact of higher torque actions
    # This would still allow the agent to perform higher torque actions when necessary, but would slightly discourage it.
    action_cost = np.sqrt(np.abs(action)).sum() * 0.05 

    # Weighted sum of the above components
    # Increase the impact of velocity_reward as it's the primary task goal
    reward += 0.6 * velocity_reward + 0.3 * forward_reward - 0.1 * action_cost
  
    return reward