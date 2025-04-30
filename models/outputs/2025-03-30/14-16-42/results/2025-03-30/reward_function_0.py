import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Ralph the Robotic arm.

    The function will compute the Euclidean distance between the current end-effector's position 
    and the desired goal's position and then it will return the negative of that distance as a reward.
    Subgoals' reward will depend on their appropriate achieving. 
    A potential can be added, it will reflect the bounded reward as the policy's performance gets better, 
    so it will incentive the robotic arm to improve its reaching manner.

    Parameters:

    obs: dict
        A dictionary that contains observation array which state and desired goal, achieved goal position are included.
    action: ndarray
        An action that should be taken based on the `obs` which is provided as the 4D torque command to the robot.
    done: bool
        A boolean flag that indicates whether or not the episode is terminated.
    env: str
        The specific environment that the function is used for. It's helpful for debugging.

    Output:

    reward: float
        A single float standing for the reward in the environment at current point.

    """
    # unwrap the environment to have access to the base environment
    env = env.unwrapped

    # if the observation is a dictionary, extract the actual observation array
    obs = obs.get('observation', obs)

    # calculate the goal distance 
    goal_distance = np.linalg.norm(env.goal - obs['achieved_goal'])

    # specify the reward function for the task
    # we'll give a negative reward proportional to the distance from the goal position
    # this will incentivize the robotic arm to minimize the distance to the goal
    reward = -goal_distance

    # add a bonus reward for successfully reaching the goal position
    # this should encourage the arm to not just get close to the goal, but to accurately reach it
    if goal_distance < 0.05:
        reward += 0.1

    # penalize the robotic arm for using high magnitude actions
    # this will incentivize the arm to reach the goal efficiently, using the smallest possible actions
    action_magnitude = np.linalg.norm(action)
    reward -= 0.01 * action_magnitude

    return reward