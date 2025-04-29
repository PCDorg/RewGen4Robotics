import numpy as np
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot.
    - obs : Observation
    - action : The action taken
    - done : Is the episode done?
    - env : The environment object.

    Returns a reward that is a function of the difference between the current velocity and the desired velocity.
    Note that this assumes 'linear_velocity' and 'angular_velocity' keys in the observation dictionary. Adjust as necessary.
    """

    # Extract the actual observation array from the observation dictionary
    obs = obs.get('observation', obs)
    
    # Access the desired velocity from the environment
    desired_velocity = env.unwrapped.desired_velocity
    
    # Get the current linear and angular velocity
    linear_velocity = obs['linear_velocity']
    angular_velocity = obs['angular_velocity']

    # Compute the reward as negative difference between the current and desired velocities
    linear_velocity_difference = np.linalg.norm(linear_velocity - desired_velocity)
    angular_velocity_difference = np.linalg.norm(angular_velocity - desired_velocity)

    # Encourage the minimization of the action
    action_penalty = np.sum(np.square(action))

    # Compute the total reward
    # The weights can be adjusted as per requirements
    reward = -1.0 * linear_velocity_difference - 1.0 * angular_velocity_difference - 0.1 * action_penalty

    return reward