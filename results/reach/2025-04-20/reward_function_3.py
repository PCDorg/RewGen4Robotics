import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    This function computes the reward for the Go1MujocoCustom-v0 task.
    The task is to make the Go1 robot walk forward at a desired velocity.

    obs: Dict[str, np.ndarray]
        Dictionary of observations, including:
        - linear_velocity: [vx, vy, vz]
        - angular_velocity: [wx, wy, wz]
        - projected_gravity: [gx, gy, gz]
        - desired_velocity: [vx, vy, wz]
        - dofs_position: joint positions
        - dofs_velocity: joint velocities
        - last_action: previous action
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

    # Get current velocity and desired velocity
    current_velocity = obs['linear_velocity']  # [vx, vy, vz]
    desired_velocity = env.desired_velocity    # [vx, vy, wz]

    # Calculate velocity tracking reward (focus on forward velocity)
    forward_velocity_error = abs(current_velocity[0] - desired_velocity[0])
    velocity_reward = np.exp(-forward_velocity_error)

    # Add penalty for large actions to encourage smooth control
    action_penalty = 0.01 * np.sum(np.square(action))

    # Add penalty for non-zero angular velocity to encourage straight walking
    angular_velocity_penalty = 0.1 * np.sum(np.square(obs['angular_velocity']))

    # Add penalty for non-flat base (using projected gravity)
    base_orientation_penalty = 0.1 * np.sum(np.square(obs['projected_gravity'][:2]))

    # Combine rewards and penalties
    total_reward = velocity_reward - action_penalty - angular_velocity_penalty - base_orientation_penalty

    # Add bonus for successful completion
    if done and forward_velocity_error < 0.1:
        total_reward += 5.0

    return total_reward