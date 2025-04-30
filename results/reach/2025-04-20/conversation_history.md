## Iteration 1/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task.
    
    Params:
    ---------
    obs : dict
        Observation dictionary, containing:
             joint_positions : array
               The positions of each joint.
             joint_velocities : array
               The velocities of each joint.
             base_velocities : array
               The velocities of the base.
             gravity_vector : array
               The projected gravity vector.
             desired_velocity : float
               The desired velocity.
             previous_actions : array
               The torques applied in the previous time step.
    action : array
        The action, or torque, taken.
    done : bool
        Whether the episode is complete or not.
    env : Go1MujocoCustom-v0 environment instance
        The environment instance. We can use this to access environment-specific 
        parameters and methods.

    Returns:
    -------
    A scalar representing the task-specific reward.
    """
    # Desired velocity
    desired_velocity = obs.get('desired_velocity')

    # Current linear velocity of the robot
    current_velocity = obs.get('base_velocities')[0]

    # Difference between current and desired velocities
    velocity_diff = np.abs(desired_velocity - current_velocity)

    # Simple reward - inverse of the velocity difference
    # Note: the smaller the difference the higher the reward
    reward = -velocity_diff

    return reward
```
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task.
    
    Params:
    ---------
    obs : dict
        Observation dictionary, containing:
             joint_positions : array
               The positions of each joint.
             joint_velocities : array
               The velocities of each joint.
             base_velocities : array
               The velocities of the base.
             gravity_vector : array
               The projected gravity vector.
             desired_velocity : float
               The desired velocity.
             previous_actions : array
               The torques applied in the previous time step.
    action : array
        The action, or torque, taken.
    done : bool
        Whether the episode is complete or not.
    env : Go1MujocoCustom-v0 environment instance
        The environment instance. We can use this to access environment-specific 
        parameters and methods.

    Returns:
    -------
    A scalar representing the task-specific reward.
    """
    # Desired velocity
    desired_velocity = obs.get('desired_velocity')

    # Current linear velocity of the robot
    current_velocity = obs.get('base_velocities')[0]

    # Difference between current and desired velocities
    velocity_diff = np.abs(desired_velocity - current_velocity)

    # Simple reward - inverse of the velocity difference
    # Note: the smaller the difference the higher the reward
    reward = -velocity_diff

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_0`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task.
    
    Params:
    ---------
    obs : dict
        Observation dictionary, containing:
             joint_positions : array
               The positions of each joint.
             joint_velocities : array
               The velocities of each joint.
             base_velocities : array
               The velocities of the base.
             gravity_vector : array
               The projected gravity vector.
             desired_velocity : float
               The desired velocity.
             previous_actions : array
               The torques applied in the previous time step.
    action : array
        The action, or torque, taken.
    done : bool
        Whether the episode is complete or not.
    env : Go1MujocoCustom-v0 environment instance
        The environment instance. We can use this to access environment-specific 
        parameters and methods.

    Returns:
    -------
    A scalar representing the task-specific reward.
    """
    # Desired velocity
    desired_velocity = obs.get('desired_velocity')

    # Current linear velocity of the robot
    current_velocity = obs.get('base_velocities')[0]

    # Difference between current and desired velocities
    velocity_diff = np.abs(desired_velocity - current_velocity)

    # Simple reward - inverse of the velocity difference
    # Note: the smaller the difference the higher the reward
    reward = -velocity_diff

    return reward
```
```

---

## Iteration 2/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task.
    
    Params:
    ---------
    obs : dict
        Observation dictionary, containing:
             joint_positions : array
               The positions of each joint.
             joint_velocities : array
               The velocities of each joint.
             base_velocities : array
               The velocities of the base.
             gravity_vector : array
               The projected gravity vector.
             desired_velocity : float
               The desired velocity.
             previous_actions : array
               The torques applied in the previous time step.
    action : array
        The action, or torque, taken.
    done : bool
        Whether the episode is complete or not.
    env : Go1MujocoCustom-v0 environment instance
        The environment instance. We can use this to access environment-specific 
        parameters and methods.

    Returns:
    -------
    A scalar representing the task-specific reward.
    """
    # Desired velocity
    desired_velocity = obs.get('desired_velocity')

    # Current linear velocity of the robot
    current_velocity = obs.get('base_velocities')[0]

    # Difference between current and desired velocities
    velocity_diff = np.abs(desired_velocity - current_velocity)

    # Simple reward - inverse of the velocity difference
    # Note: the smaller the difference the higher the reward
    reward = -velocity_diff

    return reward
```

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Based on the feedback and the failure of the evaluation stage, two potential issues could be at stake: The current velocity might be significantly different than the desired velocity, leading to an extremely low reward that doesn't encourage learning. In addition, the reward function is missing an important aspect - the energy efficiency, which is a function of joint torques(action).

A revised reward function should consider not only the tracking of the desired velocity but also the smoothness and energy efficiency of the actions, which can be captured by summing squared torques. Here is a revised version of the reward function:

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task. The function consists of two main components: 
    tracking the desired velocity and minimizing the action taken.
    
    obs : dict
        Observation dictionary, containing:
             joint_positions : array
               The positions of each joint.
             joint_velocities : array
               The velocities of each joint.
             base_velocities : array
               The velocities of the base.
             gravity_vector : array
               The projected gravity vector.
             desired_velocity : float
               The desired velocity.
             previous_actions : array
               The torques applied in the previous time step.
    action : array
        The action, or torque, taken.
    done : bool
        Whether the episode is complete or not.
    env : Go1MujocoCustom-v0 environment instance
        The environment instance. We can use this to access environment-specific 
        parameters and methods.
    
    Returns: reward : float
        The combined reward calculated from the velocity tracking reward and 
        energy efficiency reward.
    """
    
    # Velocity tracking
    desired_velocity = obs.get('desired_velocity')
    current_velocity = obs.get('base_velocities')[0]
    velocity_diff = np.abs(desired_velocity - current_velocity)
    vel_reward = -velocity_diff

    # Energy efficiency
    action_cost = 0.1 * np.square(action).sum()
    
    # It might be helpful to scale the individual components of the reward to ensure that one does not dominate the other
    reward = vel_reward - action_cost

    return reward
```
This revised function should encourage the Go1 robot to move with the desired velocity while being efficient in using actuator torques. By balancing between these two factors, the robot should perform more effectively and potentially solve the task.
```

**Generated Reward Code (saved to reward_function_1.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task. The function consists of two main components: 
    tracking the desired velocity and minimizing the action taken.
    
    obs : dict
        Observation dictionary, containing:
             joint_positions : array
               The positions of each joint.
             joint_velocities : array
               The velocities of each joint.
             base_velocities : array
               The velocities of the base.
             gravity_vector : array
               The projected gravity vector.
             desired_velocity : float
               The desired velocity.
             previous_actions : array
               The torques applied in the previous time step.
    action : array
        The action, or torque, taken.
    done : bool
        Whether the episode is complete or not.
    env : Go1MujocoCustom-v0 environment instance
        The environment instance. We can use this to access environment-specific 
        parameters and methods.
    
    Returns: reward : float
        The combined reward calculated from the velocity tracking reward and 
        energy efficiency reward.
    """
    
    # Velocity tracking
    desired_velocity = obs.get('desired_velocity')
    current_velocity = obs.get('base_velocities')[0]
    velocity_diff = np.abs(desired_velocity - current_velocity)
    vel_reward = -velocity_diff

    # Energy efficiency
    action_cost = 0.1 * np.square(action).sum()
    
    # It might be helpful to scale the individual components of the reward to ensure that one does not dominate the other
    reward = vel_reward - action_cost

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_1`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task. The function consists of two main components: 
    tracking the desired velocity and minimizing the action taken.
    
    obs : dict
        Observation dictionary, containing:
             joint_positions : array
               The positions of each joint.
             joint_velocities : array
               The velocities of each joint.
             base_velocities : array
               The velocities of the base.
             gravity_vector : array
               The projected gravity vector.
             desired_velocity : float
               The desired velocity.
             previous_actions : array
               The torques applied in the previous time step.
    action : array
        The action, or torque, taken.
    done : bool
        Whether the episode is complete or not.
    env : Go1MujocoCustom-v0 environment instance
        The environment instance. We can use this to access environment-specific 
        parameters and methods.
    
    Returns: reward : float
        The combined reward calculated from the velocity tracking reward and 
        energy efficiency reward.
    """
    
    # Velocity tracking
    desired_velocity = obs.get('desired_velocity')
    current_velocity = obs.get('base_velocities')[0]
    velocity_diff = np.abs(desired_velocity - current_velocity)
    vel_reward = -velocity_diff

    # Energy efficiency
    action_cost = 0.1 * np.square(action).sum()
    
    # It might be helpful to scale the individual components of the reward to ensure that one does not dominate the other
    reward = vel_reward - action_cost

    return reward
```
```

---

## Iteration 3/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task. The function consists of two main components: 
    tracking the desired velocity and minimizing the action taken.
    
    obs : dict
        Observation dictionary, containing:
             joint_positions : array
               The positions of each joint.
             joint_velocities : array
               The velocities of each joint.
             base_velocities : array
               The velocities of the base.
             gravity_vector : array
               The projected gravity vector.
             desired_velocity : float
               The desired velocity.
             previous_actions : array
               The torques applied in the previous time step.
    action : array
        The action, or torque, taken.
    done : bool
        Whether the episode is complete or not.
    env : Go1MujocoCustom-v0 environment instance
        The environment instance. We can use this to access environment-specific 
        parameters and methods.
    
    Returns: reward : float
        The combined reward calculated from the velocity tracking reward and 
        energy efficiency reward.
    """
    
    # Velocity tracking
    desired_velocity = obs.get('desired_velocity')
    current_velocity = obs.get('base_velocities')[0]
    velocity_diff = np.abs(desired_velocity - current_velocity)
    vel_reward = -velocity_diff

    # Energy efficiency
    action_cost = 0.1 * np.square(action).sum()
    
    # It might be helpful to scale the individual components of the reward to ensure that one does not dominate the other
    reward = vel_reward - action_cost

    return reward
```

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Let's write down the new reward function based on the given task, environment configuration, observation space, and feedback. One potential issue with the previous reward function is the scale of the velocity and energy efficiency rewards. To better balance these components, we can normalize the velocity reward between 0 and 1 and similarly scale down the energy efficiency reward. Furthermore, let's add an additional reward component for staying upright, based on the gravity vector. 

Here is the revised reward function code:

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task. The function consists of three main components: 
    (1) sharp tracking of the desired velocity, 
    (2) minimizing the energy usage (or action taken), and
    (3) a new reward for maintaining an upright posture.
    
    obs : dict
        Observation dictionary.
    action : array
        The action, or torque, applied.
    done : bool
        Once the episode is complete this turns True.
    env : Go1MujocoCustom-v0 environment instance
        The environment instance. 
    
    Returns
    -------
    reward : float
        The combined reward.
    """

    # Velocity tracking
    desired_velocity = obs.get('desired_velocity')
    current_velocity = obs.get('base_velocities')[0]
    velocity_diff = np.abs(desired_velocity - current_velocity)
    # normalize the velocity reward to a value between 0 and 1
    vel_reward = np.exp(-velocity_diff)

    # Energy efficiency: high action cost should lead to less reward
    action_cost = 0.01 * np.square(action).sum()  # scale down to limit impact

    # Upright reward: the robot should stay as upright as possible.
    # We consider the alignment of the projected gravity direction (assuming it points 'down') and the robot's current local vertical direction.
    vert_reward = obs.get('gravity_vector')[2]  # z component of gravity direction

    # Combine reward components
    reward = vel_reward - action_cost + vert_reward

    return reward
```

This reward function incentivizes the robot to precisely track the desired velocity and minimize energy usage while maintaining an upright posture, which are all critical for the task of walking forward at a desired velocity. By better balancing the scale of the reward components, this function should help the reinforcement learning algorithm more effectively learn the task.
```

**Generated Reward Code (saved to reward_function_2.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task. The function consists of three main components: 
    (1) sharp tracking of the desired velocity, 
    (2) minimizing the energy usage (or action taken), and
    (3) a new reward for maintaining an upright posture.
    
    obs : dict
        Observation dictionary.
    action : array
        The action, or torque, applied.
    done : bool
        Once the episode is complete this turns True.
    env : Go1MujocoCustom-v0 environment instance
        The environment instance. 
    
    Returns
    -------
    reward : float
        The combined reward.
    """

    # Velocity tracking
    desired_velocity = obs.get('desired_velocity')
    current_velocity = obs.get('base_velocities')[0]
    velocity_diff = np.abs(desired_velocity - current_velocity)
    # normalize the velocity reward to a value between 0 and 1
    vel_reward = np.exp(-velocity_diff)

    # Energy efficiency: high action cost should lead to less reward
    action_cost = 0.01 * np.square(action).sum()  # scale down to limit impact

    # Upright reward: the robot should stay as upright as possible.
    # We consider the alignment of the projected gravity direction (assuming it points 'down') and the robot's current local vertical direction.
    vert_reward = obs.get('gravity_vector')[2]  # z component of gravity direction

    # Combine reward components
    reward = vel_reward - action_cost + vert_reward

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_2`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task. The function consists of three main components: 
    (1) sharp tracking of the desired velocity, 
    (2) minimizing the energy usage (or action taken), and
    (3) a new reward for maintaining an upright posture.
    
    obs : dict
        Observation dictionary.
    action : array
        The action, or torque, applied.
    done : bool
        Once the episode is complete this turns True.
    env : Go1MujocoCustom-v0 environment instance
        The environment instance. 
    
    Returns
    -------
    reward : float
        The combined reward.
    """

    # Velocity tracking
    desired_velocity = obs.get('desired_velocity')
    current_velocity = obs.get('base_velocities')[0]
    velocity_diff = np.abs(desired_velocity - current_velocity)
    # normalize the velocity reward to a value between 0 and 1
    vel_reward = np.exp(-velocity_diff)

    # Energy efficiency: high action cost should lead to less reward
    action_cost = 0.01 * np.square(action).sum()  # scale down to limit impact

    # Upright reward: the robot should stay as upright as possible.
    # We consider the alignment of the projected gravity direction (assuming it points 'down') and the robot's current local vertical direction.
    vert_reward = obs.get('gravity_vector')[2]  # z component of gravity direction

    # Combine reward components
    reward = vel_reward - action_cost + vert_reward

    return reward
```
```

---

## Iteration 1/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
The custom reward function for this assignment should balance forward progress, stability, and control. In the context of the Go1 robot, forward progress is associated with advancing to the desired velocity, stability is typically correlated with the robot's stance, and control might be associated with keeping actions (i.e., joint torques) as small as possible. See the Python code below:

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Define custom reward function for the Go1 robot task.

    The reward function emphasizes three aspects:
    1. Velocity tracking: the robot should match the desired velocity.
    2. Stability: the robot should maintain a stable stance (measured as vertical orientation).
    3. Control: the robot should take actions as small as possible for energy efficiency.

    Parameters
    ----------
    obs : dict
        The agent's current observation of the environment.
    action : array
        An array of actions taken by the agent.
    done : bool
        Whether the episode is done.
    env : MujocoEnv
        The environment object.

    Returns
    -------
    float
        The computed reward.
    """

    # Extract relevant variables from the observation
    base_orientation = obs['gravity_projection']
    current_velocity = obs['linear_velocity']
    desired_velocity = env.unwrapped.desired_velocity_map

    # Compute velocity tracking component of the reward
    velocity_deviation = np.linalg.norm(current_velocity - desired_velocity)
    velocity_reward = -velocity_deviation

    # Compute stability component of the reward
    # Use dot product to measure alignment of the base with gravity
    stability_reward = np.dot(base_orientation, np.array([0, 0, 1]))

    # Compute control component of the reward
    control_penalty = 0.1 * np.square(action).sum()

    # Combine components into a single reward
    reward = velocity_reward + stability_reward - control_penalty

    return reward
```
Please note that the weights and components used in this reward function are just a starting point and might need to be adjusted based on how well they lead to desirable behavior and learning progress. For example, if the robot falls over frequently, you might want to increase the weight of the stability component. If the robot moves significantly slower than the desired velocity, consider increasing the weight of the velocity component.
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Define custom reward function for the Go1 robot task.

    The reward function emphasizes three aspects:
    1. Velocity tracking: the robot should match the desired velocity.
    2. Stability: the robot should maintain a stable stance (measured as vertical orientation).
    3. Control: the robot should take actions as small as possible for energy efficiency.

    Parameters
    ----------
    obs : dict
        The agent's current observation of the environment.
    action : array
        An array of actions taken by the agent.
    done : bool
        Whether the episode is done.
    env : MujocoEnv
        The environment object.

    Returns
    -------
    float
        The computed reward.
    """

    # Extract relevant variables from the observation
    base_orientation = obs['gravity_projection']
    current_velocity = obs['linear_velocity']
    desired_velocity = env.unwrapped.desired_velocity_map

    # Compute velocity tracking component of the reward
    velocity_deviation = np.linalg.norm(current_velocity - desired_velocity)
    velocity_reward = -velocity_deviation

    # Compute stability component of the reward
    # Use dot product to measure alignment of the base with gravity
    stability_reward = np.dot(base_orientation, np.array([0, 0, 1]))

    # Compute control component of the reward
    control_penalty = 0.1 * np.square(action).sum()

    # Combine components into a single reward
    reward = velocity_reward + stability_reward - control_penalty

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_0`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Define custom reward function for the Go1 robot task.

    The reward function emphasizes three aspects:
    1. Velocity tracking: the robot should match the desired velocity.
    2. Stability: the robot should maintain a stable stance (measured as vertical orientation).
    3. Control: the robot should take actions as small as possible for energy efficiency.

    Parameters
    ----------
    obs : dict
        The agent's current observation of the environment.
    action : array
        An array of actions taken by the agent.
    done : bool
        Whether the episode is done.
    env : MujocoEnv
        The environment object.

    Returns
    -------
    float
        The computed reward.
    """

    # Extract relevant variables from the observation
    base_orientation = obs['gravity_projection']
    current_velocity = obs['linear_velocity']
    desired_velocity = env.unwrapped.desired_velocity_map

    # Compute velocity tracking component of the reward
    velocity_deviation = np.linalg.norm(current_velocity - desired_velocity)
    velocity_reward = -velocity_deviation

    # Compute stability component of the reward
    # Use dot product to measure alignment of the base with gravity
    stability_reward = np.dot(base_orientation, np.array([0, 0, 1]))

    # Compute control component of the reward
    control_penalty = 0.1 * np.square(action).sum()

    # Combine components into a single reward
    reward = velocity_reward + stability_reward - control_penalty

    return reward
```
```

---

## Iteration 2/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Define custom reward function for the Go1 robot task.

    The reward function emphasizes three aspects:
    1. Velocity tracking: the robot should match the desired velocity.
    2. Stability: the robot should maintain a stable stance (measured as vertical orientation).
    3. Control: the robot should take actions as small as possible for energy efficiency.

    Parameters
    ----------
    obs : dict
        The agent's current observation of the environment.
    action : array
        An array of actions taken by the agent.
    done : bool
        Whether the episode is done.
    env : MujocoEnv
        The environment object.

    Returns
    -------
    float
        The computed reward.
    """

    # Extract relevant variables from the observation
    base_orientation = obs['gravity_projection']
    current_velocity = obs['linear_velocity']
    desired_velocity = env.unwrapped.desired_velocity_map

    # Compute velocity tracking component of the reward
    velocity_deviation = np.linalg.norm(current_velocity - desired_velocity)
    velocity_reward = -velocity_deviation

    # Compute stability component of the reward
    # Use dot product to measure alignment of the base with gravity
    stability_reward = np.dot(base_orientation, np.array([0, 0, 1]))

    # Compute control component of the reward
    control_penalty = 0.1 * np.square(action).sum()

    # Combine components into a single reward
    reward = velocity_reward + stability_reward - control_penalty

    return reward
```

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Define a custom reward function for the Go1 robot task.

    This reward function prioritizes the following attributes:
    1. Velocity Tracking: The robot should match the desired velocity.
    2. Stability: The robot should maintain a stable stance (upright orientation).
    3. Control: The robot should use as less energy as possible for achieving its tasks.
    4. Goal Distance: The robot should get closer to the target each step.
    
    Parameters
    ----------
    obs : dict
        The agent's current observation of the environment.
    action : array
        An array of actions taken by the agent.
    done : bool
        Whether the episode is done.
    env : Go1RobotEnv
        The environment object.

    Returns
    -------
    float
        The computed reward.
    """

    # Extract relevant variables from the observation
    base_orientation = obs['gravity_projection']
    current_velocity = obs['linear_velocity']
    desired_velocity = env.unwrapped.desired_velocity
    previous_velocity = obs['previous_velocity']

    # Calculate the difference in the previous velocity and the current velocity
    diff_velocity = abs(previous_velocity - current_velocity)
    
    # Calculate the velocity tracking part of the reward - Make it inversely proportional to difference in velocities
    velocity_reward = 1 / (1 + np.linalg.norm(current_velocity - desired_velocity))
    
    # Calculate the stability part of the reward - Use dot product to measure alignment of the base with gravity
    stability_reward = np.dot(base_orientation, np.array([0, 0, 1]))
    
    # Calculate the control part of the reward - Encourage smaller actions
    control_penalty = 0.1 * np.square(action).sum()
    
    # Calculate the goal distance part of the reward - reward getting closer to the target velocity
    distance_reward = -np.linalg.norm(desired_velocity - current_velocity)

    # Add all components of reward and return
    return velocity_reward + 2 * stability_reward - control_penalty + distance_reward
```

This reward function should help the robot to be more robust in solving the task by:
1. Encouraging matching the desired velocity.
2. Rewarding maintaining stability.
3. Penalizing high-energy actions.
4. Rewarding getting closer to the desired velocity.

We introduce a new reward signal "distance_reward" which rewards the robot for getting closer to target velocity on each step. This encourages the robot to progress towards the desired velocity over time and not just stay put.

The stability reward scaling factor is increased to 2 to emphasize the importance of maintaining stability.

In addition, the computation of velocity_reward is adjusted. Previously, the reward was directly proportional to the velocity deviation, making it possible for the robot to receive a higher reward for larger deviation. Instead, we make the velocity_reward inversely proportional to the velocity deviation, ensuring the robot gets smaller rewards for larger deviations and bigger rewards for smaller deviations, therefore encouraging better tracking of the desired velocity.

Finally, we introduce diff_velocity or difference in velocity from previous to current step, which can also be used to modulate reward if the robot's velocity changes drastically. However, it's best to first test without this component and then use it for fine-tuning.

Always validate the new reward function via training and evaluation loops! Remember that reward design is an iterative process and some trial-and-error is expected.
```

**Generated Reward Code (saved to reward_function_1.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Define a custom reward function for the Go1 robot task.

    This reward function prioritizes the following attributes:
    1. Velocity Tracking: The robot should match the desired velocity.
    2. Stability: The robot should maintain a stable stance (upright orientation).
    3. Control: The robot should use as less energy as possible for achieving its tasks.
    4. Goal Distance: The robot should get closer to the target each step.
    
    Parameters
    ----------
    obs : dict
        The agent's current observation of the environment.
    action : array
        An array of actions taken by the agent.
    done : bool
        Whether the episode is done.
    env : Go1RobotEnv
        The environment object.

    Returns
    -------
    float
        The computed reward.
    """

    # Extract relevant variables from the observation
    base_orientation = obs['gravity_projection']
    current_velocity = obs['linear_velocity']
    desired_velocity = env.unwrapped.desired_velocity
    previous_velocity = obs['previous_velocity']

    # Calculate the difference in the previous velocity and the current velocity
    diff_velocity = abs(previous_velocity - current_velocity)
    
    # Calculate the velocity tracking part of the reward - Make it inversely proportional to difference in velocities
    velocity_reward = 1 / (1 + np.linalg.norm(current_velocity - desired_velocity))
    
    # Calculate the stability part of the reward - Use dot product to measure alignment of the base with gravity
    stability_reward = np.dot(base_orientation, np.array([0, 0, 1]))
    
    # Calculate the control part of the reward - Encourage smaller actions
    control_penalty = 0.1 * np.square(action).sum()
    
    # Calculate the goal distance part of the reward - reward getting closer to the target velocity
    distance_reward = -np.linalg.norm(desired_velocity - current_velocity)

    # Add all components of reward and return
    return velocity_reward + 2 * stability_reward - control_penalty + distance_reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_1`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Define a custom reward function for the Go1 robot task.

    This reward function prioritizes the following attributes:
    1. Velocity Tracking: The robot should match the desired velocity.
    2. Stability: The robot should maintain a stable stance (upright orientation).
    3. Control: The robot should use as less energy as possible for achieving its tasks.
    4. Goal Distance: The robot should get closer to the target each step.
    
    Parameters
    ----------
    obs : dict
        The agent's current observation of the environment.
    action : array
        An array of actions taken by the agent.
    done : bool
        Whether the episode is done.
    env : Go1RobotEnv
        The environment object.

    Returns
    -------
    float
        The computed reward.
    """

    # Extract relevant variables from the observation
    base_orientation = obs['gravity_projection']
    current_velocity = obs['linear_velocity']
    desired_velocity = env.unwrapped.desired_velocity
    previous_velocity = obs['previous_velocity']

    # Calculate the difference in the previous velocity and the current velocity
    diff_velocity = abs(previous_velocity - current_velocity)
    
    # Calculate the velocity tracking part of the reward - Make it inversely proportional to difference in velocities
    velocity_reward = 1 / (1 + np.linalg.norm(current_velocity - desired_velocity))
    
    # Calculate the stability part of the reward - Use dot product to measure alignment of the base with gravity
    stability_reward = np.dot(base_orientation, np.array([0, 0, 1]))
    
    # Calculate the control part of the reward - Encourage smaller actions
    control_penalty = 0.1 * np.square(action).sum()
    
    # Calculate the goal distance part of the reward - reward getting closer to the target velocity
    distance_reward = -np.linalg.norm(desired_velocity - current_velocity)

    # Add all components of reward and return
    return velocity_reward + 2 * stability_reward - control_penalty + distance_reward
```
```

---

## Iteration 1/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment. The dictionary from Go1MujocoCustom-v0 includes keys for:
              - joint positions
              - joint velocities
              - base linear velocities
              - base angular velocities
              - projected gravity vector
              - desired velocity
              - previous actions
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific information needed for the reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    
    # Define weights for different components of the reward function
    weight_velocity_tracking = 1.0  # weight for tracking the desired linear velocity and angular velocity
    weight_energy_efficiency = 0.1  # weight for encouraging energy efficiency (i.e., lower joint torques)

    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))
    # Access desired velocities
    desired_velocity = np.array(env.desired_velocity)
    # Calculate how much the current velocity differs from desired velocity. 
    # Low cost (high reward) is given when the robot is moving at the desired linear velocity.
    velocity_tracking_cost = np.linalg.norm(base_linear_velocity - desired_velocity)

    # Encourage energy efficiency by prefering use of lower joint torques
    energy_efficiency_cost = np.square(action).mean()

    # Reward is the negative of cost, we want to minimize the cost hence the negative sign.
    reward = - weight_velocity_tracking * velocity_tracking_cost - weight_energy_efficiency * energy_efficiency_cost

    return reward
```
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment. The dictionary from Go1MujocoCustom-v0 includes keys for:
              - joint positions
              - joint velocities
              - base linear velocities
              - base angular velocities
              - projected gravity vector
              - desired velocity
              - previous actions
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific information needed for the reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    
    # Define weights for different components of the reward function
    weight_velocity_tracking = 1.0  # weight for tracking the desired linear velocity and angular velocity
    weight_energy_efficiency = 0.1  # weight for encouraging energy efficiency (i.e., lower joint torques)

    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))
    # Access desired velocities
    desired_velocity = np.array(env.desired_velocity)
    # Calculate how much the current velocity differs from desired velocity. 
    # Low cost (high reward) is given when the robot is moving at the desired linear velocity.
    velocity_tracking_cost = np.linalg.norm(base_linear_velocity - desired_velocity)

    # Encourage energy efficiency by prefering use of lower joint torques
    energy_efficiency_cost = np.square(action).mean()

    # Reward is the negative of cost, we want to minimize the cost hence the negative sign.
    reward = - weight_velocity_tracking * velocity_tracking_cost - weight_energy_efficiency * energy_efficiency_cost

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_0`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment. The dictionary from Go1MujocoCustom-v0 includes keys for:
              - joint positions
              - joint velocities
              - base linear velocities
              - base angular velocities
              - projected gravity vector
              - desired velocity
              - previous actions
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific information needed for the reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    
    # Define weights for different components of the reward function
    weight_velocity_tracking = 1.0  # weight for tracking the desired linear velocity and angular velocity
    weight_energy_efficiency = 0.1  # weight for encouraging energy efficiency (i.e., lower joint torques)

    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))
    # Access desired velocities
    desired_velocity = np.array(env.desired_velocity)
    # Calculate how much the current velocity differs from desired velocity. 
    # Low cost (high reward) is given when the robot is moving at the desired linear velocity.
    velocity_tracking_cost = np.linalg.norm(base_linear_velocity - desired_velocity)

    # Encourage energy efficiency by prefering use of lower joint torques
    energy_efficiency_cost = np.square(action).mean()

    # Reward is the negative of cost, we want to minimize the cost hence the negative sign.
    reward = - weight_velocity_tracking * velocity_tracking_cost - weight_energy_efficiency * energy_efficiency_cost

    return reward
```
```

---

## Iteration 2/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment. The dictionary from Go1MujocoCustom-v0 includes keys for:
              - joint positions
              - joint velocities
              - base linear velocities
              - base angular velocities
              - projected gravity vector
              - desired velocity
              - previous actions
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific information needed for the reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    
    # Define weights for different components of the reward function
    weight_velocity_tracking = 1.0  # weight for tracking the desired linear velocity and angular velocity
    weight_energy_efficiency = 0.1  # weight for encouraging energy efficiency (i.e., lower joint torques)

    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))
    # Access desired velocities
    desired_velocity = np.array(env.desired_velocity)
    # Calculate how much the current velocity differs from desired velocity. 
    # Low cost (high reward) is given when the robot is moving at the desired linear velocity.
    velocity_tracking_cost = np.linalg.norm(base_linear_velocity - desired_velocity)

    # Encourage energy efficiency by prefering use of lower joint torques
    energy_efficiency_cost = np.square(action).mean()

    # Reward is the negative of cost, we want to minimize the cost hence the negative sign.
    reward = - weight_velocity_tracking * velocity_tracking_cost - weight_energy_efficiency * energy_efficiency_cost

    return reward
```

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Here is a revised reward function that modifies the balance between velocity tracking and energy efficiency:

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment, includes:
              - joint positions,
              - joint velocities,
              - base linear velocities,
              - base angular velocities,
              - projected gravity vector,
              - desired velocity,
              - previous actions.
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific info needed for reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    # Define weights for different components of the reward function
    weight_velocity_tracking = 1.0   # Weight for tracking the desired velocity
    weight_energy_efficiency = 0.05  # REDUCED Weight to encourage energy efficiency (i.e., lower joint torques)

    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))
    # Access desired velocities
    desired_velocity = np.array(obs.get('desired_velocity'))
    # Calculate how much the current velocity differs from desired velocity
    # Low cost (high reward) for moving at the desired velocity
    velocity_tracking_cost = np.abs(base_linear_velocity - desired_velocity).sum()

    # Encourage energy efficiency by prefering lower joint torques
    energy_efficiency_cost = np.square(action).sum()

    # Reward is the negative of cost. Minimizing cost = maximizing negative cost
    reward = - weight_velocity_tracking * velocity_tracking_cost - weight_energy_efficiency * energy_efficiency_cost

    # Penalty for falling
    if done:
        reward -= 100.0

    return reward
```
In this revised function, the balance between the velocity tracking cost and the energy efficiency cost has been modified with a reduced weight on energy efficiency. The robot should be able to better track the desired velocity without being overly penalized for using the necessary joint torques.

Also, a penalty has been added for case when mission ends prematurely (the robot falls). This will discourage the robot from actions that lead to falling.

The reward is still defined as the negative cost, so the RL algorithm is still performing cost minimization.
```

**Generated Reward Code (saved to reward_function_1.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment, includes:
              - joint positions,
              - joint velocities,
              - base linear velocities,
              - base angular velocities,
              - projected gravity vector,
              - desired velocity,
              - previous actions.
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific info needed for reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    # Define weights for different components of the reward function
    weight_velocity_tracking = 1.0   # Weight for tracking the desired velocity
    weight_energy_efficiency = 0.05  # REDUCED Weight to encourage energy efficiency (i.e., lower joint torques)

    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))
    # Access desired velocities
    desired_velocity = np.array(obs.get('desired_velocity'))
    # Calculate how much the current velocity differs from desired velocity
    # Low cost (high reward) for moving at the desired velocity
    velocity_tracking_cost = np.abs(base_linear_velocity - desired_velocity).sum()

    # Encourage energy efficiency by prefering lower joint torques
    energy_efficiency_cost = np.square(action).sum()

    # Reward is the negative of cost. Minimizing cost = maximizing negative cost
    reward = - weight_velocity_tracking * velocity_tracking_cost - weight_energy_efficiency * energy_efficiency_cost

    # Penalty for falling
    if done:
        reward -= 100.0

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_1`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment, includes:
              - joint positions,
              - joint velocities,
              - base linear velocities,
              - base angular velocities,
              - projected gravity vector,
              - desired velocity,
              - previous actions.
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific info needed for reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    # Define weights for different components of the reward function
    weight_velocity_tracking = 1.0   # Weight for tracking the desired velocity
    weight_energy_efficiency = 0.05  # REDUCED Weight to encourage energy efficiency (i.e., lower joint torques)

    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))
    # Access desired velocities
    desired_velocity = np.array(obs.get('desired_velocity'))
    # Calculate how much the current velocity differs from desired velocity
    # Low cost (high reward) for moving at the desired velocity
    velocity_tracking_cost = np.abs(base_linear_velocity - desired_velocity).sum()

    # Encourage energy efficiency by prefering lower joint torques
    energy_efficiency_cost = np.square(action).sum()

    # Reward is the negative of cost. Minimizing cost = maximizing negative cost
    reward = - weight_velocity_tracking * velocity_tracking_cost - weight_energy_efficiency * energy_efficiency_cost

    # Penalty for falling
    if done:
        reward -= 100.0

    return reward
```
```

---

## Iteration 3/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment, includes:
              - joint positions,
              - joint velocities,
              - base linear velocities,
              - base angular velocities,
              - projected gravity vector,
              - desired velocity,
              - previous actions.
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific info needed for reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    # Define weights for different components of the reward function
    weight_velocity_tracking = 1.0   # Weight for tracking the desired velocity
    weight_energy_efficiency = 0.05  # REDUCED Weight to encourage energy efficiency (i.e., lower joint torques)

    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))
    # Access desired velocities
    desired_velocity = np.array(obs.get('desired_velocity'))
    # Calculate how much the current velocity differs from desired velocity
    # Low cost (high reward) for moving at the desired velocity
    velocity_tracking_cost = np.abs(base_linear_velocity - desired_velocity).sum()

    # Encourage energy efficiency by prefering lower joint torques
    energy_efficiency_cost = np.square(action).sum()

    # Reward is the negative of cost. Minimizing cost = maximizing negative cost
    reward = - weight_velocity_tracking * velocity_tracking_cost - weight_energy_efficiency * energy_efficiency_cost

    # Penalty for falling
    if done:
        reward -= 100.0

    return reward
```

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment, includes:
              - joint positions,
              - joint velocities,
              - base linear velocities,
              - base angular velocities,
              - projected gravity vector,
              - desired velocity,
              - previous actions.
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific info needed for reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    
    # Define weights for different components of the reward function 
    weight_velocity_tracking = 1.0  # Weight for tracking the desired velocity
    weight_energy_efficiency = 0.1  # INCREASED Weight to encourage energy efficiency (i.e., lower joint torques)
    weight_joints_use = 0.1  # NEW Weight for encouraging balanced use of joints
    
    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))
    # Access desired velocities
    desired_velocity = np.array(obs.get('desired_velocity'))
    # Calculate how much the current velocity differs from desired velocity
    velocity_tracking_cost = np.abs(base_linear_velocity - desired_velocity).sum()

    # Encourage energy efficiency by preferring lower and balanced joint torques
    energy_efficiency_cost = np.square(action).sum()

    # Encourage balanced use of robot's joints
    joints_use_cost = np.std(action)  # We are using standard deviation as a measure of balance
    
    # Reward is the negative of cost.
    reward = - weight_velocity_tracking * velocity_tracking_cost - weight_energy_efficiency * energy_efficiency_cost - weight_joints_use * joints_use_cost

    # Penalty for falling
    if done:
        reward -= 100.0

    return reward
```
This new reward function includes a balance component which encourages the robot to utilise all its joints equally, rather than relying on a subset of joints for locomotion. It also increases the weight of the energy efficiency cost to reward lower joint torques, which promotes more efficient locomotion.
```

**Generated Reward Code (saved to reward_function_2.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment, includes:
              - joint positions,
              - joint velocities,
              - base linear velocities,
              - base angular velocities,
              - projected gravity vector,
              - desired velocity,
              - previous actions.
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific info needed for reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    
    # Define weights for different components of the reward function 
    weight_velocity_tracking = 1.0  # Weight for tracking the desired velocity
    weight_energy_efficiency = 0.1  # INCREASED Weight to encourage energy efficiency (i.e., lower joint torques)
    weight_joints_use = 0.1  # NEW Weight for encouraging balanced use of joints
    
    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))
    # Access desired velocities
    desired_velocity = np.array(obs.get('desired_velocity'))
    # Calculate how much the current velocity differs from desired velocity
    velocity_tracking_cost = np.abs(base_linear_velocity - desired_velocity).sum()

    # Encourage energy efficiency by preferring lower and balanced joint torques
    energy_efficiency_cost = np.square(action).sum()

    # Encourage balanced use of robot's joints
    joints_use_cost = np.std(action)  # We are using standard deviation as a measure of balance
    
    # Reward is the negative of cost.
    reward = - weight_velocity_tracking * velocity_tracking_cost - weight_energy_efficiency * energy_efficiency_cost - weight_joints_use * joints_use_cost

    # Penalty for falling
    if done:
        reward -= 100.0

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_2`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment, includes:
              - joint positions,
              - joint velocities,
              - base linear velocities,
              - base angular velocities,
              - projected gravity vector,
              - desired velocity,
              - previous actions.
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific info needed for reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    
    # Define weights for different components of the reward function 
    weight_velocity_tracking = 1.0  # Weight for tracking the desired velocity
    weight_energy_efficiency = 0.1  # INCREASED Weight to encourage energy efficiency (i.e., lower joint torques)
    weight_joints_use = 0.1  # NEW Weight for encouraging balanced use of joints
    
    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))
    # Access desired velocities
    desired_velocity = np.array(obs.get('desired_velocity'))
    # Calculate how much the current velocity differs from desired velocity
    velocity_tracking_cost = np.abs(base_linear_velocity - desired_velocity).sum()

    # Encourage energy efficiency by preferring lower and balanced joint torques
    energy_efficiency_cost = np.square(action).sum()

    # Encourage balanced use of robot's joints
    joints_use_cost = np.std(action)  # We are using standard deviation as a measure of balance
    
    # Reward is the negative of cost.
    reward = - weight_velocity_tracking * velocity_tracking_cost - weight_energy_efficiency * energy_efficiency_cost - weight_joints_use * joints_use_cost

    # Penalty for falling
    if done:
        reward -= 100.0

    return reward
```
```

---

## Iteration 4/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment, includes:
              - joint positions,
              - joint velocities,
              - base linear velocities,
              - base angular velocities,
              - projected gravity vector,
              - desired velocity,
              - previous actions.
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific info needed for reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    
    # Define weights for different components of the reward function 
    weight_velocity_tracking = 1.0  # Weight for tracking the desired velocity
    weight_energy_efficiency = 0.1  # INCREASED Weight to encourage energy efficiency (i.e., lower joint torques)
    weight_joints_use = 0.1  # NEW Weight for encouraging balanced use of joints
    
    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))
    # Access desired velocities
    desired_velocity = np.array(obs.get('desired_velocity'))
    # Calculate how much the current velocity differs from desired velocity
    velocity_tracking_cost = np.abs(base_linear_velocity - desired_velocity).sum()

    # Encourage energy efficiency by preferring lower and balanced joint torques
    energy_efficiency_cost = np.square(action).sum()

    # Encourage balanced use of robot's joints
    joints_use_cost = np.std(action)  # We are using standard deviation as a measure of balance
    
    # Reward is the negative of cost.
    reward = - weight_velocity_tracking * velocity_tracking_cost - weight_energy_efficiency * energy_efficiency_cost - weight_joints_use * joints_use_cost

    # Penalty for falling
    if done:
        reward -= 100.0

    return reward
```

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment, includes:
              - joint positions,
              - joint velocities,
              - base linear velocities,
              - base angular velocities,
              - projected gravity vector,
              - desired velocity,
              - previous actions.
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific info needed for reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    
    # Define weights for different components of the reward function 
    weight_velocity_tracking = 1.0  # Weight for tracking the desired velocity
    weight_energy_efficiency = 0.1  # Weight to encourage energy efficiency (i.e., lower joint torques)
    weight_joints_use = 0.01  # REDUCED Weight for encouraging balanced use of joints
    
    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))[0]
    # Access base angular velocities
    base_angular_velocity = np.array(obs.get('base_angular_velocities'))[2]
    # Access desired velocities
    desired_velocity = obs.get('desired_velocity')[0]
    desired_angular_velocity = obs.get('desired_velocity')[1]
    
    # Calculate how much the current velocity differs from desired velocity
    velocity_tracking_cost = np.abs(base_linear_velocity - desired_velocity)

    # Calculate how much the current angular velocity differs from desired angular velocity
    angular_velocity_tracking_cost = np.abs(base_angular_velocity - desired_angular_velocity)

    # Encourage energy efficiency by preferring lower and balanced joint torques
    energy_efficiency_cost = np.square(action).sum()

    # Encourage balanced use of robot's joints
    joints_use_cost = np.std(action)  # We are using standard deviation as a measure of balance
    
    # Reward is the negative of cost.
    tracking_cost = weight_velocity_tracking * velocity_tracking_cost + weight_velocity_tracking * angular_velocity_tracking_cost
    efficiency_cost = weight_energy_efficiency * energy_efficiency_cost + weight_joints_use * joints_use_cost
    reward = - tracking_cost - efficiency_cost

    # Penalty for falling
    if done:
        reward -= 100.0

    return reward
```
This is an improved reward function for the problem. The improvements include consideration for correct angular velocity and a reduced weight for encouraging equal use of all joints. This should lead to a better balance of learning the task and energy efficiency. A performance-dependent penalty for falling has also been added to enhance the stability of the robot.
```

**Generated Reward Code (saved to reward_function_3.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment, includes:
              - joint positions,
              - joint velocities,
              - base linear velocities,
              - base angular velocities,
              - projected gravity vector,
              - desired velocity,
              - previous actions.
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific info needed for reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    
    # Define weights for different components of the reward function 
    weight_velocity_tracking = 1.0  # Weight for tracking the desired velocity
    weight_energy_efficiency = 0.1  # Weight to encourage energy efficiency (i.e., lower joint torques)
    weight_joints_use = 0.01  # REDUCED Weight for encouraging balanced use of joints
    
    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))[0]
    # Access base angular velocities
    base_angular_velocity = np.array(obs.get('base_angular_velocities'))[2]
    # Access desired velocities
    desired_velocity = obs.get('desired_velocity')[0]
    desired_angular_velocity = obs.get('desired_velocity')[1]
    
    # Calculate how much the current velocity differs from desired velocity
    velocity_tracking_cost = np.abs(base_linear_velocity - desired_velocity)

    # Calculate how much the current angular velocity differs from desired angular velocity
    angular_velocity_tracking_cost = np.abs(base_angular_velocity - desired_angular_velocity)

    # Encourage energy efficiency by preferring lower and balanced joint torques
    energy_efficiency_cost = np.square(action).sum()

    # Encourage balanced use of robot's joints
    joints_use_cost = np.std(action)  # We are using standard deviation as a measure of balance
    
    # Reward is the negative of cost.
    tracking_cost = weight_velocity_tracking * velocity_tracking_cost + weight_velocity_tracking * angular_velocity_tracking_cost
    efficiency_cost = weight_energy_efficiency * energy_efficiency_cost + weight_joints_use * joints_use_cost
    reward = - tracking_cost - efficiency_cost

    # Penalty for falling
    if done:
        reward -= 100.0

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_3`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 task.

    Args:
        obs : dict
            The observation from the environment, includes:
              - joint positions,
              - joint velocities,
              - base linear velocities,
              - base angular velocities,
              - projected gravity vector,
              - desired velocity,
              - previous actions.
        action : array_like
            The torques to be applied at each joint of the robot.
        done : bool
            Whether or not the episode has ended.
        env : Env
            The current environment. Use it to access any environment-specific info needed for reward function.

    Returns:
        float : The reward signal for the current state-action pair.
    """
    
    # Define weights for different components of the reward function 
    weight_velocity_tracking = 1.0  # Weight for tracking the desired velocity
    weight_energy_efficiency = 0.1  # Weight to encourage energy efficiency (i.e., lower joint torques)
    weight_joints_use = 0.01  # REDUCED Weight for encouraging balanced use of joints
    
    # Access base linear velocities
    base_linear_velocity = np.array(obs.get('base_linear_velocities'))[0]
    # Access base angular velocities
    base_angular_velocity = np.array(obs.get('base_angular_velocities'))[2]
    # Access desired velocities
    desired_velocity = obs.get('desired_velocity')[0]
    desired_angular_velocity = obs.get('desired_velocity')[1]
    
    # Calculate how much the current velocity differs from desired velocity
    velocity_tracking_cost = np.abs(base_linear_velocity - desired_velocity)

    # Calculate how much the current angular velocity differs from desired angular velocity
    angular_velocity_tracking_cost = np.abs(base_angular_velocity - desired_angular_velocity)

    # Encourage energy efficiency by preferring lower and balanced joint torques
    energy_efficiency_cost = np.square(action).sum()

    # Encourage balanced use of robot's joints
    joints_use_cost = np.std(action)  # We are using standard deviation as a measure of balance
    
    # Reward is the negative of cost.
    tracking_cost = weight_velocity_tracking * velocity_tracking_cost + weight_velocity_tracking * angular_velocity_tracking_cost
    efficiency_cost = weight_energy_efficiency * energy_efficiency_cost + weight_joints_use * joints_use_cost
    reward = - tracking_cost - efficiency_cost

    # Penalty for falling
    if done:
        reward -= 100.0

    return reward
```
```

---

## Iteration 1/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the 'Go1MujocoCustom-v0' environment.

    Args:
        obs (dict): The observation from the environment. It's a dictionary with keys: 'joint_positions',
                    'joint_velocities', 'base_velocity', 'gravity_vector', 'desired_velocity', and 'previous_action'.
        action (np.ndarray): The action taken by the agent. It's an array of joint torques.
        done (bool): Whether the episode has finished.
        env: The environment instance.

    Returns:
        float: The computed reward for the current state and action.
    """

    # Access the forward velocity (the x-component of the base linear velocity)
    forward_velocity = obs['base_velocity'][0]

    # Access the desired forward velocity (the x-component of the desired velocity)
    desired_velocity = env.desired_velocity[0]

    # Calculate the velocity error (the absolute difference between current and desired velocities)
    vel_error = np.abs(forward_velocity - desired_velocity)

    # We will subtract a factor of the velocity error from a base reward of 1,
    # so the robot is encouraged to minimize this error
    # The factor of 0.1 ensures our reward stays in a reasonable scale
    reward = 1 - 0.1 * vel_error

    # Penalize the action if it's too large, to encourage more stable, smooth movement
    # Use the Euclidean norm (L2 norm) of the action vector
    action_magnitude = np.linalg.norm(action)
    reward -= 0.01 * action_magnitude

    # Adjustments for done state
    if done:
        if vel_error < 0.05:
            # Big reward for reaching the goal
            reward += 1.0
        else:
            # Big penalty for failing to reach the goal
            reward -= 1.0

    return reward
```
Note: This is just a basic reward function and may need further tuning for your specific task. The numbers used for weighing different components of the reward (e.g., `0.1` for the velocity error, `0.01` for penalizing large actions, `1.0` for the bonus/penalty when the episode ends) are somewhat arbitrary and may need to be adjusted depending on the specifics of the environment and the desired behavior of the robot.
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the 'Go1MujocoCustom-v0' environment.

    Args:
        obs (dict): The observation from the environment. It's a dictionary with keys: 'joint_positions',
                    'joint_velocities', 'base_velocity', 'gravity_vector', 'desired_velocity', and 'previous_action'.
        action (np.ndarray): The action taken by the agent. It's an array of joint torques.
        done (bool): Whether the episode has finished.
        env: The environment instance.

    Returns:
        float: The computed reward for the current state and action.
    """

    # Access the forward velocity (the x-component of the base linear velocity)
    forward_velocity = obs['base_velocity'][0]

    # Access the desired forward velocity (the x-component of the desired velocity)
    desired_velocity = env.desired_velocity[0]

    # Calculate the velocity error (the absolute difference between current and desired velocities)
    vel_error = np.abs(forward_velocity - desired_velocity)

    # We will subtract a factor of the velocity error from a base reward of 1,
    # so the robot is encouraged to minimize this error
    # The factor of 0.1 ensures our reward stays in a reasonable scale
    reward = 1 - 0.1 * vel_error

    # Penalize the action if it's too large, to encourage more stable, smooth movement
    # Use the Euclidean norm (L2 norm) of the action vector
    action_magnitude = np.linalg.norm(action)
    reward -= 0.01 * action_magnitude

    # Adjustments for done state
    if done:
        if vel_error < 0.05:
            # Big reward for reaching the goal
            reward += 1.0
        else:
            # Big penalty for failing to reach the goal
            reward -= 1.0

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_0`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the 'Go1MujocoCustom-v0' environment.

    Args:
        obs (dict): The observation from the environment. It's a dictionary with keys: 'joint_positions',
                    'joint_velocities', 'base_velocity', 'gravity_vector', 'desired_velocity', and 'previous_action'.
        action (np.ndarray): The action taken by the agent. It's an array of joint torques.
        done (bool): Whether the episode has finished.
        env: The environment instance.

    Returns:
        float: The computed reward for the current state and action.
    """

    # Access the forward velocity (the x-component of the base linear velocity)
    forward_velocity = obs['base_velocity'][0]

    # Access the desired forward velocity (the x-component of the desired velocity)
    desired_velocity = env.desired_velocity[0]

    # Calculate the velocity error (the absolute difference between current and desired velocities)
    vel_error = np.abs(forward_velocity - desired_velocity)

    # We will subtract a factor of the velocity error from a base reward of 1,
    # so the robot is encouraged to minimize this error
    # The factor of 0.1 ensures our reward stays in a reasonable scale
    reward = 1 - 0.1 * vel_error

    # Penalize the action if it's too large, to encourage more stable, smooth movement
    # Use the Euclidean norm (L2 norm) of the action vector
    action_magnitude = np.linalg.norm(action)
    reward -= 0.01 * action_magnitude

    # Adjustments for done state
    if done:
        if vel_error < 0.05:
            # Big reward for reaching the goal
            reward += 1.0
        else:
            # Big penalty for failing to reach the goal
            reward -= 1.0

    return reward
```
```

---

## Iteration 2/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the 'Go1MujocoCustom-v0' environment.

    Args:
        obs (dict): The observation from the environment. It's a dictionary with keys: 'joint_positions',
                    'joint_velocities', 'base_velocity', 'gravity_vector', 'desired_velocity', and 'previous_action'.
        action (np.ndarray): The action taken by the agent. It's an array of joint torques.
        done (bool): Whether the episode has finished.
        env: The environment instance.

    Returns:
        float: The computed reward for the current state and action.
    """

    # Access the forward velocity (the x-component of the base linear velocity)
    forward_velocity = obs['base_velocity'][0]

    # Access the desired forward velocity (the x-component of the desired velocity)
    desired_velocity = env.desired_velocity[0]

    # Calculate the velocity error (the absolute difference between current and desired velocities)
    vel_error = np.abs(forward_velocity - desired_velocity)

    # We will subtract a factor of the velocity error from a base reward of 1,
    # so the robot is encouraged to minimize this error
    # The factor of 0.1 ensures our reward stays in a reasonable scale
    reward = 1 - 0.1 * vel_error

    # Penalize the action if it's too large, to encourage more stable, smooth movement
    # Use the Euclidean norm (L2 norm) of the action vector
    action_magnitude = np.linalg.norm(action)
    reward -= 0.01 * action_magnitude

    # Adjustments for done state
    if done:
        if vel_error < 0.05:
            # Big reward for reaching the goal
            reward += 1.0
        else:
            # Big penalty for failing to reach the goal
            reward -= 1.0

    return reward
```

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Based on the feedback, the policy seems to fail during evaluation, which means we might need to re-write the reward function. Besides, there is a consideration of action penalty and also a separate reward or penalty for completing the episode.

In analyzing the previous reward function it is observed that it is very sensitive to velocity errors. This might sometimes lead to negative rewards which is not ideal. In the new reward function, we'll take these into account and define a way to assign rewards which are positive and can allow our RL to better optimize.

In the new function, velocity errors are still penalized, but we'll use a continuous and smooth function (in this case, an exponential decay), which ensures a higher reward when the velocity error is low and the reward gradually decreases as error increases - keeping reward positive all the time. Also, we will remove the absolute measure of the action magnitude penalty and use a scaled down variant of it. Done state reward adjustment also will focus on larger reward for success and smaller penalty for failure.

Here's the new reward function:

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the 'Go1MujocoCustom-v0' environment.

    Args:
        obs (dict): The observation from the environment. It's a dictionary with keys: 'joint_positions',
                    'joint_velocities', 'base_velocity', 'gravity_vector', 'desired_velocity', and 'previous_action'.
        action (np.ndarray): The action taken by the agent. It's an array of joint torques.
        done (bool): Whether the episode has finished.
        env: The environment instance.

    Returns:
        float: The computed reward for the current state and action.
    """

    # Access the forward velocity (the x-component of the base linear velocity)
    forward_velocity = obs['base_velocity'][0]

    # Access the desired forward velocity (the x-component of the desired velocity)
    desired_velocity = env.desired_velocity[0]

    # Calculate the velocity error (the absolute difference between current and desired velocities)
    vel_error = np.abs(forward_velocity - desired_velocity)

    # Reward based on the velocity error -- exponential decay
    reward = np.exp(-vel_error)

    # Penalize according to action size, but scaled down
    action_penalty = 0.001 * np.linalg.norm(action)
    reward -= action_penalty

    # Adjustments for done state
    if done:
        if vel_error < 0.05:
            # The reward for successful completion is large
            reward += 5.0
        else:
            # The penalty for not meeting the target velocity should not be too large
            # Subtracting a small constant from the reward
            reward -= 1.0

    return reward
```
This revised reward function should give a more balanced and robust reward assignment which could help in learning effectively.
```

**Generated Reward Code (saved to reward_function_1.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the 'Go1MujocoCustom-v0' environment.

    Args:
        obs (dict): The observation from the environment. It's a dictionary with keys: 'joint_positions',
                    'joint_velocities', 'base_velocity', 'gravity_vector', 'desired_velocity', and 'previous_action'.
        action (np.ndarray): The action taken by the agent. It's an array of joint torques.
        done (bool): Whether the episode has finished.
        env: The environment instance.

    Returns:
        float: The computed reward for the current state and action.
    """

    # Access the forward velocity (the x-component of the base linear velocity)
    forward_velocity = obs['base_velocity'][0]

    # Access the desired forward velocity (the x-component of the desired velocity)
    desired_velocity = env.desired_velocity[0]

    # Calculate the velocity error (the absolute difference between current and desired velocities)
    vel_error = np.abs(forward_velocity - desired_velocity)

    # Reward based on the velocity error -- exponential decay
    reward = np.exp(-vel_error)

    # Penalize according to action size, but scaled down
    action_penalty = 0.001 * np.linalg.norm(action)
    reward -= action_penalty

    # Adjustments for done state
    if done:
        if vel_error < 0.05:
            # The reward for successful completion is large
            reward += 5.0
        else:
            # The penalty for not meeting the target velocity should not be too large
            # Subtracting a small constant from the reward
            reward -= 1.0

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_1`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the 'Go1MujocoCustom-v0' environment.

    Args:
        obs (dict): The observation from the environment. It's a dictionary with keys: 'joint_positions',
                    'joint_velocities', 'base_velocity', 'gravity_vector', 'desired_velocity', and 'previous_action'.
        action (np.ndarray): The action taken by the agent. It's an array of joint torques.
        done (bool): Whether the episode has finished.
        env: The environment instance.

    Returns:
        float: The computed reward for the current state and action.
    """

    # Access the forward velocity (the x-component of the base linear velocity)
    forward_velocity = obs['base_velocity'][0]

    # Access the desired forward velocity (the x-component of the desired velocity)
    desired_velocity = env.desired_velocity[0]

    # Calculate the velocity error (the absolute difference between current and desired velocities)
    vel_error = np.abs(forward_velocity - desired_velocity)

    # Reward based on the velocity error -- exponential decay
    reward = np.exp(-vel_error)

    # Penalize according to action size, but scaled down
    action_penalty = 0.001 * np.linalg.norm(action)
    reward -= action_penalty

    # Adjustments for done state
    if done:
        if vel_error < 0.05:
            # The reward for successful completion is large
            reward += 5.0
        else:
            # The penalty for not meeting the target velocity should not be too large
            # Subtracting a small constant from the reward
            reward -= 1.0

    return reward
```
```

---

## Iteration 1/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
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

    # Extract observations relevant to the reward function
    actual_velocity = obs.get('velocity', None)
    desired_velocity = obs.get('desired_velocity', env.unwrapped.desired_velocity)

    # Compute absolute difference between actual and desired velocities
    velocity_diff = np.abs(actual_velocity - desired_velocity)

    # Compute reward as negative sum of the velocity difference
    # (we want to minimize this value)
    reward = -np.sum(velocity_diff)

    # Consider the energy efficiency - penalize high torque actions
    action_cost = np.sum(np.square(action))*0.1
    reward -= action_cost

    return reward
```
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
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

    # Extract observations relevant to the reward function
    actual_velocity = obs.get('velocity', None)
    desired_velocity = obs.get('desired_velocity', env.unwrapped.desired_velocity)

    # Compute absolute difference between actual and desired velocities
    velocity_diff = np.abs(actual_velocity - desired_velocity)

    # Compute reward as negative sum of the velocity difference
    # (we want to minimize this value)
    reward = -np.sum(velocity_diff)

    # Consider the energy efficiency - penalize high torque actions
    action_cost = np.sum(np.square(action))*0.1
    reward -= action_cost

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_0`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
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

    # Extract observations relevant to the reward function
    actual_velocity = obs.get('velocity', None)
    desired_velocity = obs.get('desired_velocity', env.unwrapped.desired_velocity)

    # Compute absolute difference between actual and desired velocities
    velocity_diff = np.abs(actual_velocity - desired_velocity)

    # Compute reward as negative sum of the velocity difference
    # (we want to minimize this value)
    reward = -np.sum(velocity_diff)

    # Consider the energy efficiency - penalize high torque actions
    action_cost = np.sum(np.square(action))*0.1
    reward -= action_cost

    return reward
```
```

---

## Iteration 2/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
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

    # Extract observations relevant to the reward function
    actual_velocity = obs.get('velocity', None)
    desired_velocity = obs.get('desired_velocity', env.unwrapped.desired_velocity)

    # Compute absolute difference between actual and desired velocities
    velocity_diff = np.abs(actual_velocity - desired_velocity)

    # Compute reward as negative sum of the velocity difference
    # (we want to minimize this value)
    reward = -np.sum(velocity_diff)

    # Consider the energy efficiency - penalize high torque actions
    action_cost = np.sum(np.square(action))*0.1
    reward -= action_cost

    return reward
```

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
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
    # Use negative of the L2 distance between actual and desired velocities 
    # Note: Multiply by -1 because the agent needs to minimize this value
    velocity_reward = -np.linalg.norm(actual_velocity - desired_velocity)

    # Provide a reward for forward motion while penalising any angular deviation
    # Here, the x-axis might represent the forward direction
    # Modify appropriately based on the observation space definition
    forward_reward = actual_velocity[0] - np.abs(actual_velocity[1]) * 0.5

    # Penalize high torque actions to encourage energy efficiency
    # Penalize more heavily to ensure a balance between velocity achievement and energy efficiency
    action_cost = np.sum(np.square(action)) * 0.25

    # Weighted sum of the above components
    reward += 0.5 * velocity_reward + 0.4 * forward_reward - 0.1 * action_cost
  
    return reward
```

The new reward function now better captures all important aspects of the task - hitting the target velocity, moving in the forward direction, and using efficient actions. 

Most importantly, the penalties and rewards are now set in relative terms to each other (weighted averaging), this should help optimize the reward function and improve learning. 

The weights can be further tuned during hyperparameter optimization.
```

**Generated Reward Code (saved to reward_function_1.py):**
```python
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
    # Use negative of the L2 distance between actual and desired velocities 
    # Note: Multiply by -1 because the agent needs to minimize this value
    velocity_reward = -np.linalg.norm(actual_velocity - desired_velocity)

    # Provide a reward for forward motion while penalising any angular deviation
    # Here, the x-axis might represent the forward direction
    # Modify appropriately based on the observation space definition
    forward_reward = actual_velocity[0] - np.abs(actual_velocity[1]) * 0.5

    # Penalize high torque actions to encourage energy efficiency
    # Penalize more heavily to ensure a balance between velocity achievement and energy efficiency
    action_cost = np.sum(np.square(action)) * 0.25

    # Weighted sum of the above components
    reward += 0.5 * velocity_reward + 0.4 * forward_reward - 0.1 * action_cost
  
    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_1`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
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
    # Use negative of the L2 distance between actual and desired velocities 
    # Note: Multiply by -1 because the agent needs to minimize this value
    velocity_reward = -np.linalg.norm(actual_velocity - desired_velocity)

    # Provide a reward for forward motion while penalising any angular deviation
    # Here, the x-axis might represent the forward direction
    # Modify appropriately based on the observation space definition
    forward_reward = actual_velocity[0] - np.abs(actual_velocity[1]) * 0.5

    # Penalize high torque actions to encourage energy efficiency
    # Penalize more heavily to ensure a balance between velocity achievement and energy efficiency
    action_cost = np.sum(np.square(action)) * 0.25

    # Weighted sum of the above components
    reward += 0.5 * velocity_reward + 0.4 * forward_reward - 0.1 * action_cost
  
    return reward
```
```

---

## Iteration 3/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
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
    # Use negative of the L2 distance between actual and desired velocities 
    # Note: Multiply by -1 because the agent needs to minimize this value
    velocity_reward = -np.linalg.norm(actual_velocity - desired_velocity)

    # Provide a reward for forward motion while penalising any angular deviation
    # Here, the x-axis might represent the forward direction
    # Modify appropriately based on the observation space definition
    forward_reward = actual_velocity[0] - np.abs(actual_velocity[1]) * 0.5

    # Penalize high torque actions to encourage energy efficiency
    # Penalize more heavily to ensure a balance between velocity achievement and energy efficiency
    action_cost = np.sum(np.square(action)) * 0.25

    # Weighted sum of the above components
    reward += 0.5 * velocity_reward + 0.4 * forward_reward - 0.1 * action_cost
  
    return reward
```

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
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
```
In the code above, moderate changes are made to the previous reward function. The primary changes were in the distance computation (from L2 to L1), the way the action cost was computed, and the balancing or weightage of the rewards. Conceptually, these changes emphasize on getting closer to the target velocity, (but with a lesser discouraged factor for higher torque) and maintaining a forward course with a more lenient penalty. This should provide a more balanced problem space for the RL algorithm to optimize.
```

**Generated Reward Code (saved to reward_function_2.py):**
```python
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
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_2`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
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
```
```

---

## Iteration 4/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
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
```

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Based on the existing reward function, a significant problem is the absence of a termination statement. If the desired speed is reached, the environment should terminate successfully to help the agent understand its accomplishment. The reward system for forward motion appears to place a significant emphasis on the x-direction, ignoring possible implementation in other directions.

Let's refactor the reward function accordingly.

```python
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
    # Use negative of the L2 distance between actual and desired velocities instead 
    # of L1 to increase the reward as the velocity gets closer to the desired velocity.
    # Note: Multiply by -1 because the agent needs to minimize this value
    velocity_reward = -np.linalg.norm(actual_velocity - desired_velocity)

    # Provide a reward for moving towards the goal while penalising any deviation from the goal
    # We calculate a vector towards the goal and penalize any actual_velocity not in that direction
    # This places as much emphasis on other directions and not just the forward direction 
    forward_reward = np.dot(actual_velocity, desired_velocity) 

    # Penalize high torque actions to encourage energy efficiency
    # Use L2 norm instead of square root to encourage smaller actions due to the squared term.
    action_cost = np.linalg.norm(action) * 0.05 

    # Weighted sum of the above components
    # Increase the impact of velocity_reward as it's the primary task goal
    reward += 0.6 * velocity_reward + 0.3 * forward_reward - 0.1 * action_cost
  
    # Add a termination statement. If the desired speed is reached (within a small tolerance), 
    # then the environment should terminate successfully. This is to aid the agent's understanding of 
    # its objective and help it learn faster.
    if np.isclose(actual_velocity, desired_velocity, atol=0.1):
        reward += 10.0  # big reward for reaching the goal
        done = True  # end the episode

    return reward
```
This reward function encourages movement towards the desired velocity, punishes large actions, and marks the end of an episode once the desired velocity is reached. It is more aligned with the task requirements and should help the learning agent generalize better. The forward reward computation also now involves all direction velocities, making it more robust.
```

**Generated Reward Code (saved to reward_function_3.py):**
```python
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
    # Use negative of the L2 distance between actual and desired velocities instead 
    # of L1 to increase the reward as the velocity gets closer to the desired velocity.
    # Note: Multiply by -1 because the agent needs to minimize this value
    velocity_reward = -np.linalg.norm(actual_velocity - desired_velocity)

    # Provide a reward for moving towards the goal while penalising any deviation from the goal
    # We calculate a vector towards the goal and penalize any actual_velocity not in that direction
    # This places as much emphasis on other directions and not just the forward direction 
    forward_reward = np.dot(actual_velocity, desired_velocity) 

    # Penalize high torque actions to encourage energy efficiency
    # Use L2 norm instead of square root to encourage smaller actions due to the squared term.
    action_cost = np.linalg.norm(action) * 0.05 

    # Weighted sum of the above components
    # Increase the impact of velocity_reward as it's the primary task goal
    reward += 0.6 * velocity_reward + 0.3 * forward_reward - 0.1 * action_cost
  
    # Add a termination statement. If the desired speed is reached (within a small tolerance), 
    # then the environment should terminate successfully. This is to aid the agent's understanding of 
    # its objective and help it learn faster.
    if np.isclose(actual_velocity, desired_velocity, atol=0.1):
        reward += 10.0  # big reward for reaching the goal
        done = True  # end the episode

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_3`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
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
    # Use negative of the L2 distance between actual and desired velocities instead 
    # of L1 to increase the reward as the velocity gets closer to the desired velocity.
    # Note: Multiply by -1 because the agent needs to minimize this value
    velocity_reward = -np.linalg.norm(actual_velocity - desired_velocity)

    # Provide a reward for moving towards the goal while penalising any deviation from the goal
    # We calculate a vector towards the goal and penalize any actual_velocity not in that direction
    # This places as much emphasis on other directions and not just the forward direction 
    forward_reward = np.dot(actual_velocity, desired_velocity) 

    # Penalize high torque actions to encourage energy efficiency
    # Use L2 norm instead of square root to encourage smaller actions due to the squared term.
    action_cost = np.linalg.norm(action) * 0.05 

    # Weighted sum of the above components
    # Increase the impact of velocity_reward as it's the primary task goal
    reward += 0.6 * velocity_reward + 0.3 * forward_reward - 0.1 * action_cost
  
    # Add a termination statement. If the desired speed is reached (within a small tolerance), 
    # then the environment should terminate successfully. This is to aid the agent's understanding of 
    # its objective and help it learn faster.
    if np.isclose(actual_velocity, desired_velocity, atol=0.1):
        reward += 10.0  # big reward for reaching the goal
        done = True  # end the episode

    return reward
```
```

---

## Iteration 1/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Here, we'll consider two fundamental aspects of the reward function for Go1 robot while trying to move at a desired velocity: a tracking term and a regularization term. The tracking term tries to minimize the difference between the robot's actual velocity and the desired velocity, while the regularization term penalizes large actions to smooth and stabilize the robot's movement.

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1MujocoCustom-v0 task. The function computes
    a reward based on the agent's performance in tracking a desired linear velocity
    and the magnitude of the applied actions (control torques).

    Args:
        obs (dict): observation dictionary from the Go1 robot environment.
        action (np.ndarray): applied action (joint torques) in the current step.
        done (bool): episode termination flag.
        env (gym.Env): Go1 robot environment. Use env.unwrapped.desired_velocity to access target velocity.

    Returns:
        reward (float): the computed reward for the current step.
    """
    # Extract data from the observation dictionary
    actual_velocity = obs['base_velocity']  # linear_velocity can be part of base_velocity
    desired_velocity = env.unwrapped.desired_velocity  # extract desired velocity from env

    # Conversion from dictionary format to NumPy format for easier computation
    actual_velocity = np.array(actual_velocity)
    desired_velocity = np.array(desired_velocity)

    # Step 1: Calculate the tracking reward
    # Compute the difference between desired (goal) and actual velocities
    diff = desired_velocity - actual_velocity
    # Reward is proportional to the negative sum of squared differences. The closer the actual 
    # velocity is to the desired velocity, the smaller the squared difference, hence a larger reward. 
    tracking_reward = -np.sum(diff**2)

    # Step 2: Calculate regularization reward
    # The regularization_reward is given by the negative sum of squared actions. The rationale is 
    # to encourage the agent to achieve the task with small and smooth actions.
    regularization_reward = -np.sum(action**2)

    # Combine reward terms with appropriate weights
    # You may adjust the weights based on the specific requirements of your experiment.
    reward = 0.8*tracking_reward + 0.2*regularization_reward

    return reward
```

The above function encourages the robot to track the desired forward velocity (tracking_reward) while keeping its actions (joint torques) as small and smooth as possible (regularization_reward). The coefficients (0.8 and 0.2) can be adjusted based on the specific requirements and performance of the experiment. The relation of these coefficients determines the balance between tracking the desired velocity and regularizing the actions.
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1MujocoCustom-v0 task. The function computes
    a reward based on the agent's performance in tracking a desired linear velocity
    and the magnitude of the applied actions (control torques).

    Args:
        obs (dict): observation dictionary from the Go1 robot environment.
        action (np.ndarray): applied action (joint torques) in the current step.
        done (bool): episode termination flag.
        env (gym.Env): Go1 robot environment. Use env.unwrapped.desired_velocity to access target velocity.

    Returns:
        reward (float): the computed reward for the current step.
    """
    # Extract data from the observation dictionary
    actual_velocity = obs['base_velocity']  # linear_velocity can be part of base_velocity
    desired_velocity = env.unwrapped.desired_velocity  # extract desired velocity from env

    # Conversion from dictionary format to NumPy format for easier computation
    actual_velocity = np.array(actual_velocity)
    desired_velocity = np.array(desired_velocity)

    # Step 1: Calculate the tracking reward
    # Compute the difference between desired (goal) and actual velocities
    diff = desired_velocity - actual_velocity
    # Reward is proportional to the negative sum of squared differences. The closer the actual 
    # velocity is to the desired velocity, the smaller the squared difference, hence a larger reward. 
    tracking_reward = -np.sum(diff**2)

    # Step 2: Calculate regularization reward
    # The regularization_reward is given by the negative sum of squared actions. The rationale is 
    # to encourage the agent to achieve the task with small and smooth actions.
    regularization_reward = -np.sum(action**2)

    # Combine reward terms with appropriate weights
    # You may adjust the weights based on the specific requirements of your experiment.
    reward = 0.8*tracking_reward + 0.2*regularization_reward

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_0`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1MujocoCustom-v0 task. The function computes
    a reward based on the agent's performance in tracking a desired linear velocity
    and the magnitude of the applied actions (control torques).

    Args:
        obs (dict): observation dictionary from the Go1 robot environment.
        action (np.ndarray): applied action (joint torques) in the current step.
        done (bool): episode termination flag.
        env (gym.Env): Go1 robot environment. Use env.unwrapped.desired_velocity to access target velocity.

    Returns:
        reward (float): the computed reward for the current step.
    """
    # Extract data from the observation dictionary
    actual_velocity = obs['base_velocity']  # linear_velocity can be part of base_velocity
    desired_velocity = env.unwrapped.desired_velocity  # extract desired velocity from env

    # Conversion from dictionary format to NumPy format for easier computation
    actual_velocity = np.array(actual_velocity)
    desired_velocity = np.array(desired_velocity)

    # Step 1: Calculate the tracking reward
    # Compute the difference between desired (goal) and actual velocities
    diff = desired_velocity - actual_velocity
    # Reward is proportional to the negative sum of squared differences. The closer the actual 
    # velocity is to the desired velocity, the smaller the squared difference, hence a larger reward. 
    tracking_reward = -np.sum(diff**2)

    # Step 2: Calculate regularization reward
    # The regularization_reward is given by the negative sum of squared actions. The rationale is 
    # to encourage the agent to achieve the task with small and smooth actions.
    regularization_reward = -np.sum(action**2)

    # Combine reward terms with appropriate weights
    # You may adjust the weights based on the specific requirements of your experiment.
    reward = 0.8*tracking_reward + 0.2*regularization_reward

    return reward
```
```

---

## Iteration 2/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1MujocoCustom-v0 task. The function computes
    a reward based on the agent's performance in tracking a desired linear velocity
    and the magnitude of the applied actions (control torques).

    Args:
        obs (dict): observation dictionary from the Go1 robot environment.
        action (np.ndarray): applied action (joint torques) in the current step.
        done (bool): episode termination flag.
        env (gym.Env): Go1 robot environment. Use env.unwrapped.desired_velocity to access target velocity.

    Returns:
        reward (float): the computed reward for the current step.
    """
    # Extract data from the observation dictionary
    actual_velocity = obs['base_velocity']  # linear_velocity can be part of base_velocity
    desired_velocity = env.unwrapped.desired_velocity  # extract desired velocity from env

    # Conversion from dictionary format to NumPy format for easier computation
    actual_velocity = np.array(actual_velocity)
    desired_velocity = np.array(desired_velocity)

    # Step 1: Calculate the tracking reward
    # Compute the difference between desired (goal) and actual velocities
    diff = desired_velocity - actual_velocity
    # Reward is proportional to the negative sum of squared differences. The closer the actual 
    # velocity is to the desired velocity, the smaller the squared difference, hence a larger reward. 
    tracking_reward = -np.sum(diff**2)

    # Step 2: Calculate regularization reward
    # The regularization_reward is given by the negative sum of squared actions. The rationale is 
    # to encourage the agent to achieve the task with small and smooth actions.
    regularization_reward = -np.sum(action**2)

    # Combine reward terms with appropriate weights
    # You may adjust the weights based on the specific requirements of your experiment.
    reward = 0.8*tracking_reward + 0.2*regularization_reward

    return reward
```

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1MujocoCustom-v0 task. The function computes
    a reward based on the agent's performance in tracking a desired linear and angular velocity
    and the magnitude of the applied actions (control torques).

    Args:
        obs (dict): observation dictionary from the Go1 robot environment.
        action (np.ndarray): applied action (joint torques) in the current step.
        done (bool): episode termination flag.
        env (gym.Env): Go1 robot environment. Use env.unwrapped.desired_velocity to access target velocities.

    Returns:
        reward (float): the computed reward for the current step.
    """
    # Extract data from the observation dictionary
    actual_linear_velocity = obs['base_velocity'][0:3]  # Assumes that linear velocity component is the first three elements of base_velocity
    desired_linear_velocity = env.unwrapped.desired_velocity[0:3]  # Assumes that desired linear velocity component is the first three elements of desired_velocity
    actual_angular_velocity = obs['base_velocity'][3:6]  # Assumes that angular velocity component is the last three elements of base_velocity
    desired_angular_velocity = env.unwrapped.desired_velocity[3:6]  # Assumes that desired angular velocity component is the last three elements of desired_velocity

    # Conversion from dictionary format to NumPy format for easier computation
    actual_linear_velocity = np.array(actual_linear_velocity)
    desired_linear_velocity = np.array(desired_linear_velocity)
    actual_angular_velocity = np.array(actual_angular_velocity)
    desired_angular_velocity = np.array(desired_angular_velocity)

    # Step 1: Calculate the tracking reward
    # Compute the difference between desired (goal) and actual velocities 
    linear_diff = desired_linear_velocity - actual_linear_velocity
    angular_diff = desired_angular_velocity - actual_angular_velocity
    # Reward is proportional to the negative sum of squared differences. The closer the actual 
    # velocity is to the desired velocity, the smaller the squared difference, hence a larger reward. 
    tracking_reward = -np.sum(linear_diff**2) - np.sum(angular_diff**2)

    # Step 2: Calculate regularization reward
    # The regularization_reward is given by the negative sum of squared actions. The rationale is 
    # to encourage the agent to achieve the task with small and smooth actions.
    regularization_reward = -np.sum(action**2)

    # Combine reward terms with appropriate weights
    # You may adjust the weights based on the specific requirements of your experiment.
    reward = 0.8*tracking_reward + 0.2*regularization_reward

    # Step 3: Additional reward for moving forward
    # If the actual linear velocity in the forward direction (assumed to be the first element) 
    # is closer to the desired linear velocity, it gets an additional reward.
    forward_reward = 1 - np.abs(desired_linear_velocity[0] - actual_linear_velocity[0])

    # Combine the forward reward with the previous computed reward
    reward = 0.5 * reward + 0.5 * forward_reward

    return reward
```
The function above considers not only the tracking of the desired linear velocity but also the angular velocity. Additionally, we've added a new reward (forward_reward) for moving forward which encourages the agent to correctly align in the direction of the desired velocity. Note that the weights used for combining the reward components are arbitrary and may need to be tuned for optimal performance. The assumptions regarding the components of the velocity vectors (e.g., the first three elements corresponding to linear velocity) should be verified based on the configuration of the environment.
```

**Generated Reward Code (saved to reward_function_1.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1MujocoCustom-v0 task. The function computes
    a reward based on the agent's performance in tracking a desired linear and angular velocity
    and the magnitude of the applied actions (control torques).

    Args:
        obs (dict): observation dictionary from the Go1 robot environment.
        action (np.ndarray): applied action (joint torques) in the current step.
        done (bool): episode termination flag.
        env (gym.Env): Go1 robot environment. Use env.unwrapped.desired_velocity to access target velocities.

    Returns:
        reward (float): the computed reward for the current step.
    """
    # Extract data from the observation dictionary
    actual_linear_velocity = obs['base_velocity'][0:3]  # Assumes that linear velocity component is the first three elements of base_velocity
    desired_linear_velocity = env.unwrapped.desired_velocity[0:3]  # Assumes that desired linear velocity component is the first three elements of desired_velocity
    actual_angular_velocity = obs['base_velocity'][3:6]  # Assumes that angular velocity component is the last three elements of base_velocity
    desired_angular_velocity = env.unwrapped.desired_velocity[3:6]  # Assumes that desired angular velocity component is the last three elements of desired_velocity

    # Conversion from dictionary format to NumPy format for easier computation
    actual_linear_velocity = np.array(actual_linear_velocity)
    desired_linear_velocity = np.array(desired_linear_velocity)
    actual_angular_velocity = np.array(actual_angular_velocity)
    desired_angular_velocity = np.array(desired_angular_velocity)

    # Step 1: Calculate the tracking reward
    # Compute the difference between desired (goal) and actual velocities 
    linear_diff = desired_linear_velocity - actual_linear_velocity
    angular_diff = desired_angular_velocity - actual_angular_velocity
    # Reward is proportional to the negative sum of squared differences. The closer the actual 
    # velocity is to the desired velocity, the smaller the squared difference, hence a larger reward. 
    tracking_reward = -np.sum(linear_diff**2) - np.sum(angular_diff**2)

    # Step 2: Calculate regularization reward
    # The regularization_reward is given by the negative sum of squared actions. The rationale is 
    # to encourage the agent to achieve the task with small and smooth actions.
    regularization_reward = -np.sum(action**2)

    # Combine reward terms with appropriate weights
    # You may adjust the weights based on the specific requirements of your experiment.
    reward = 0.8*tracking_reward + 0.2*regularization_reward

    # Step 3: Additional reward for moving forward
    # If the actual linear velocity in the forward direction (assumed to be the first element) 
    # is closer to the desired linear velocity, it gets an additional reward.
    forward_reward = 1 - np.abs(desired_linear_velocity[0] - actual_linear_velocity[0])

    # Combine the forward reward with the previous computed reward
    reward = 0.5 * reward + 0.5 * forward_reward

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_1`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1MujocoCustom-v0 task. The function computes
    a reward based on the agent's performance in tracking a desired linear and angular velocity
    and the magnitude of the applied actions (control torques).

    Args:
        obs (dict): observation dictionary from the Go1 robot environment.
        action (np.ndarray): applied action (joint torques) in the current step.
        done (bool): episode termination flag.
        env (gym.Env): Go1 robot environment. Use env.unwrapped.desired_velocity to access target velocities.

    Returns:
        reward (float): the computed reward for the current step.
    """
    # Extract data from the observation dictionary
    actual_linear_velocity = obs['base_velocity'][0:3]  # Assumes that linear velocity component is the first three elements of base_velocity
    desired_linear_velocity = env.unwrapped.desired_velocity[0:3]  # Assumes that desired linear velocity component is the first three elements of desired_velocity
    actual_angular_velocity = obs['base_velocity'][3:6]  # Assumes that angular velocity component is the last three elements of base_velocity
    desired_angular_velocity = env.unwrapped.desired_velocity[3:6]  # Assumes that desired angular velocity component is the last three elements of desired_velocity

    # Conversion from dictionary format to NumPy format for easier computation
    actual_linear_velocity = np.array(actual_linear_velocity)
    desired_linear_velocity = np.array(desired_linear_velocity)
    actual_angular_velocity = np.array(actual_angular_velocity)
    desired_angular_velocity = np.array(desired_angular_velocity)

    # Step 1: Calculate the tracking reward
    # Compute the difference between desired (goal) and actual velocities 
    linear_diff = desired_linear_velocity - actual_linear_velocity
    angular_diff = desired_angular_velocity - actual_angular_velocity
    # Reward is proportional to the negative sum of squared differences. The closer the actual 
    # velocity is to the desired velocity, the smaller the squared difference, hence a larger reward. 
    tracking_reward = -np.sum(linear_diff**2) - np.sum(angular_diff**2)

    # Step 2: Calculate regularization reward
    # The regularization_reward is given by the negative sum of squared actions. The rationale is 
    # to encourage the agent to achieve the task with small and smooth actions.
    regularization_reward = -np.sum(action**2)

    # Combine reward terms with appropriate weights
    # You may adjust the weights based on the specific requirements of your experiment.
    reward = 0.8*tracking_reward + 0.2*regularization_reward

    # Step 3: Additional reward for moving forward
    # If the actual linear velocity in the forward direction (assumed to be the first element) 
    # is closer to the desired linear velocity, it gets an additional reward.
    forward_reward = 1 - np.abs(desired_linear_velocity[0] - actual_linear_velocity[0])

    # Combine the forward reward with the previous computed reward
    reward = 0.5 * reward + 0.5 * forward_reward

    return reward
```
```

---

## Iteration 1/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Make the Go1 robot walk forward at a desired velocity.
Environment Name: Go1MujocoCustom-v0
Full Task Description:
The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.

Detailed Environment Configuration:
- env_name: Go1MujocoCustom-v0
- task: Make the Go1 robot walk forward at a desired velocity.
- task_description: The agent controls a Unitree Go1 quadruped robot simulated in MuJoCo. The goal is to track a desired linear (forward) and angular velocity. Observations include joint positions/velocities, base linear/angular velocities, projected gravity vector, desired velocity, and previous actions. Actions are joint torques.
- timestamps: 1000000
- ctrl_type: torque

Observation Space Details:
Observation space structure not available.

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`

Environment-Specific Notes:
- **Go1 Env:** The target is a velocity vector. Access it using `env.desired_velocity`. Do NOT use `env.goal`.
- **FetchReach Env:** The target is a position. Access the desired goal position using `env.unwrapped.goal`.
- **AntMaze Env:** Refer to environment documentation or observation details for goal/target information.

IMPORTANT: `obs` is a dictionary for Go1 and FetchReach. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.

Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

# We define our custom reward function which takes 
# observation, action, done flag and environment as inputs
def custom_reward_function(obs, action, done, env):

    # If it's a dict, fetch the actual obs array
    obs = obs.get('observation', obs)

    # Velocity reward
    # Observed velocity. We assume obs['linear_velocity'] returns the forward velocity of Go1 robot
    vel_obs = obs['linear_velocity']

    # Desired velocity is expected to be accessed via env.desired_velocity
    vel_desired = env.desired_velocity

    # We define velocity reward as negative difference between desired and actual velocities
    # We square this difference to give high priority to huge difference
    vel_reward = -np.sum(np.square(vel_obs - vel_desired))

    # Stand reward
    # To validate if Go1 robot is standing, we check the base of robot
    # If the z-component of base position is less than 0.8 for example,
    # Go1 robot is not standing and we give a negative reward for this
    # We assume obs['base_pos_z'] returns the z-component of base position
    stand_reward = 0 if obs['base_pos_z'] > 0.8 else -1

    # Control reward
    # We use hinge loss for control reward. If joint torques exceed a threshold,
    # penalty will boost. We assume action is an array of joint torques.
    # x if x < 0 else 0 for each element in actions.
    ctrl_reward = -np.sum(np.square(np.maximum(0, np.abs(action) - 0.2)))

    # We combine all these rewards with different weights
    # It is important to tune these weights based on the complexity and priorities of tasks
    reward = 0.3 * vel_reward + 0.2 * stand_reward + 0.5 * ctrl_reward

    # Stable Baselines3 expects a single float that represents the reward.
    return reward
```

The function assigns rewards based on the difference between the robot's desired velocity and the actual velocity, whether the robot is standing or not, and the control effort exerted by the robot (measured by joint torques). The total reward is computed as a weighted sum of these components. Higher weight is assigned to velocity tracking while maintaining some reward to encourage the robot to stay upright and exert less effort. The weights are chosen based on priorities and they should be carefully tuned according to the specific scenario. Since obs is a dictionary type, we use string keys to access the components of the observations.
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

# We define our custom reward function which takes 
# observation, action, done flag and environment as inputs
def custom_reward_function(obs, action, done, env):

    # If it's a dict, fetch the actual obs array
    obs = obs.get('observation', obs)

    # Velocity reward
    # Observed velocity. We assume obs['linear_velocity'] returns the forward velocity of Go1 robot
    vel_obs = obs['linear_velocity']

    # Desired velocity is expected to be accessed via env.desired_velocity
    vel_desired = env.desired_velocity

    # We define velocity reward as negative difference between desired and actual velocities
    # We square this difference to give high priority to huge difference
    vel_reward = -np.sum(np.square(vel_obs - vel_desired))

    # Stand reward
    # To validate if Go1 robot is standing, we check the base of robot
    # If the z-component of base position is less than 0.8 for example,
    # Go1 robot is not standing and we give a negative reward for this
    # We assume obs['base_pos_z'] returns the z-component of base position
    stand_reward = 0 if obs['base_pos_z'] > 0.8 else -1

    # Control reward
    # We use hinge loss for control reward. If joint torques exceed a threshold,
    # penalty will boost. We assume action is an array of joint torques.
    # x if x < 0 else 0 for each element in actions.
    ctrl_reward = -np.sum(np.square(np.maximum(0, np.abs(action) - 0.2)))

    # We combine all these rewards with different weights
    # It is important to tune these weights based on the complexity and priorities of tasks
    reward = 0.3 * vel_reward + 0.2 * stand_reward + 0.5 * ctrl_reward

    # Stable Baselines3 expects a single float that represents the reward.
    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_0`
- Saved Model: Failed or Skipped
- Average Evaluation Reward: `N/A`
- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `N/A`
- Error Encountered: Yes (See Status/Feedback)

**Feedback Content Generated for Next Iteration:**
```

Please carefully analyze the following policy feedback and reward function components. Based on this analysis, provide a new, improved reward function that can better solve the task.

Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward function. (Note: Success rate info might not be explicitly available, use Average Evaluation Reward as a proxy).
(2) If the values for a certain reward component are near identical throughout (Note: Component-wise breakdown is not provided, infer from code structure and overall results), then this means RL is not able to optimize this component as it is written. You may consider:
    (a) Changing its scale or the value of its temperature parameter,
    (b) Re-writing the reward component, or
    (c) Discarding the reward component.
(3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range.

Please analyze the existing reward function code in the suggested manner above first, considering the provided results (Average Eval Reward, Final Training Reward, Error messages if any). Then write the new reward function code.

The reward function must have the signature:
    def custom_reward_function(obs, action, done, env):
Use only NumPy and standard Python, and access the target position using env.unwrapped.goal.

--- [End of Analysis Tips] ---

Below is the information from the previous iteration:

**Status:** Partial Success - Evaluation Failed
**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `N/A`
**Evaluation Result:** Failed to get an average reward (likely no episodes completed).

Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals).

**Previous Reward Function Code:**
```python
import numpy as np

# We define our custom reward function which takes 
# observation, action, done flag and environment as inputs
def custom_reward_function(obs, action, done, env):

    # If it's a dict, fetch the actual obs array
    obs = obs.get('observation', obs)

    # Velocity reward
    # Observed velocity. We assume obs['linear_velocity'] returns the forward velocity of Go1 robot
    vel_obs = obs['linear_velocity']

    # Desired velocity is expected to be accessed via env.desired_velocity
    vel_desired = env.desired_velocity

    # We define velocity reward as negative difference between desired and actual velocities
    # We square this difference to give high priority to huge difference
    vel_reward = -np.sum(np.square(vel_obs - vel_desired))

    # Stand reward
    # To validate if Go1 robot is standing, we check the base of robot
    # If the z-component of base position is less than 0.8 for example,
    # Go1 robot is not standing and we give a negative reward for this
    # We assume obs['base_pos_z'] returns the z-component of base position
    stand_reward = 0 if obs['base_pos_z'] > 0.8 else -1

    # Control reward
    # We use hinge loss for control reward. If joint torques exceed a threshold,
    # penalty will boost. We assume action is an array of joint torques.
    # x if x < 0 else 0 for each element in actions.
    ctrl_reward = -np.sum(np.square(np.maximum(0, np.abs(action) - 0.2)))

    # We combine all these rewards with different weights
    # It is important to tune these weights based on the complexity and priorities of tasks
    reward = 0.3 * vel_reward + 0.2 * stand_reward + 0.5 * ctrl_reward

    # Stable Baselines3 expects a single float that represents the reward.
    return reward
```
```

---

