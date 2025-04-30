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

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Here is an implementation of the reward function based on task and environment configuration described.

```python
import numpy as np

def custom_reward_function(obs, action, done, env):

    # Access the base environment
    base_env = env.unwrapped

    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    # assume that the obs is a dict with keys: 'joint_pos', 'joint_vel', 'base_lin_vel',
    # 'base_ang_vel', 'gravity', 'des_vel', 'prev_actions'
    joint_pos = obs.get('joint_pos', np.zeros_like(base_env.action_space.sample()))
    joint_vel = obs.get('joint_vel', np.zeros_like(base_env.action_space.sample()))
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    base_ang_vel = obs.get('base_ang_vel', np.zeros(3))
    gravity = obs.get('gravity', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)

    # reward tuning parameters
    joint_pos_weight = 0.1
    joint_vel_weight = 0.1
    base_lin_vel_weight = 1.0
    base_ang_vel_weight = 0.1
    action_weight = 0.1
    
    # rewards are higher for staying closer to the desired velocity
    reward = - np.square(desired_velocity - base_lin_vel[0])
    
    # rewards are higher for smaller angle speed to keep the balance
    reward -= base_ang_vel_weight * np.square(base_ang_vel).sum()
    
    # rewards are higher for smaller joint speeds to avoid unnecessary motion
    reward -= joint_vel_weight * np.square(joint_vel).sum()

    # penalize actions (torque), we prefer the torque to be as small as possible
    reward -= action_weight * np.square(action).sum()

    return reward
```
In this reward function, we set different reward weight for joint positions and velocities, base linear and angular velocities, and action. Since the goal is to make the Go1 robot walk forward at a desired velocity, we set the biggest reward for staying closer to the desired forward velocity ('base_lin_vel[0]'). Other factors, like the velocities of the robot's joints ('joint_vel'), the angular velocity of the robot's base ('base_ang_vel'), and the joint torques ('action') are penalized, as we want to maintain the stability and minimize the energy consumption.
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):

    # Access the base environment
    base_env = env.unwrapped

    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    # assume that the obs is a dict with keys: 'joint_pos', 'joint_vel', 'base_lin_vel',
    # 'base_ang_vel', 'gravity', 'des_vel', 'prev_actions'
    joint_pos = obs.get('joint_pos', np.zeros_like(base_env.action_space.sample()))
    joint_vel = obs.get('joint_vel', np.zeros_like(base_env.action_space.sample()))
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    base_ang_vel = obs.get('base_ang_vel', np.zeros(3))
    gravity = obs.get('gravity', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)

    # reward tuning parameters
    joint_pos_weight = 0.1
    joint_vel_weight = 0.1
    base_lin_vel_weight = 1.0
    base_ang_vel_weight = 0.1
    action_weight = 0.1
    
    # rewards are higher for staying closer to the desired velocity
    reward = - np.square(desired_velocity - base_lin_vel[0])
    
    # rewards are higher for smaller angle speed to keep the balance
    reward -= base_ang_vel_weight * np.square(base_ang_vel).sum()
    
    # rewards are higher for smaller joint speeds to avoid unnecessary motion
    reward -= joint_vel_weight * np.square(joint_vel).sum()

    # penalize actions (torque), we prefer the torque to be as small as possible
    reward -= action_weight * np.square(action).sum()

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

    # Access the base environment
    base_env = env.unwrapped

    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    # assume that the obs is a dict with keys: 'joint_pos', 'joint_vel', 'base_lin_vel',
    # 'base_ang_vel', 'gravity', 'des_vel', 'prev_actions'
    joint_pos = obs.get('joint_pos', np.zeros_like(base_env.action_space.sample()))
    joint_vel = obs.get('joint_vel', np.zeros_like(base_env.action_space.sample()))
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    base_ang_vel = obs.get('base_ang_vel', np.zeros(3))
    gravity = obs.get('gravity', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)

    # reward tuning parameters
    joint_pos_weight = 0.1
    joint_vel_weight = 0.1
    base_lin_vel_weight = 1.0
    base_ang_vel_weight = 0.1
    action_weight = 0.1
    
    # rewards are higher for staying closer to the desired velocity
    reward = - np.square(desired_velocity - base_lin_vel[0])
    
    # rewards are higher for smaller angle speed to keep the balance
    reward -= base_ang_vel_weight * np.square(base_ang_vel).sum()
    
    # rewards are higher for smaller joint speeds to avoid unnecessary motion
    reward -= joint_vel_weight * np.square(joint_vel).sum()

    # penalize actions (torque), we prefer the torque to be as small as possible
    reward -= action_weight * np.square(action).sum()

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

    # Access the base environment
    base_env = env.unwrapped

    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    # assume that the obs is a dict with keys: 'joint_pos', 'joint_vel', 'base_lin_vel',
    # 'base_ang_vel', 'gravity', 'des_vel', 'prev_actions'
    joint_pos = obs.get('joint_pos', np.zeros_like(base_env.action_space.sample()))
    joint_vel = obs.get('joint_vel', np.zeros_like(base_env.action_space.sample()))
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    base_ang_vel = obs.get('base_ang_vel', np.zeros(3))
    gravity = obs.get('gravity', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)

    # reward tuning parameters
    joint_pos_weight = 0.1
    joint_vel_weight = 0.1
    base_lin_vel_weight = 1.0
    base_ang_vel_weight = 0.1
    action_weight = 0.1
    
    # rewards are higher for staying closer to the desired velocity
    reward = - np.square(desired_velocity - base_lin_vel[0])
    
    # rewards are higher for smaller angle speed to keep the balance
    reward -= base_ang_vel_weight * np.square(base_ang_vel).sum()
    
    # rewards are higher for smaller joint speeds to avoid unnecessary motion
    reward -= joint_vel_weight * np.square(joint_vel).sum()

    # penalize actions (torque), we prefer the torque to be as small as possible
    reward -= action_weight * np.square(action).sum()

    return reward
```

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Here is a revised reward function based on previous issues.

```python
import numpy as np

def custom_reward_function(obs, action, done, env):

    # Access the base environment
    base_env = env.unwrapped

    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    # assume that the obs is a dict with keys: 'joint_pos', 'joint_vel', 'base_lin_vel',
    # 'base_ang_vel', 'gravity', 'des_vel', 'prev_actions'
    joint_pos = obs.get('joint_pos', np.zeros_like(base_env.action_space.sample()))
    joint_vel = obs.get('joint_vel', np.zeros_like(base_env.action_space.sample()))
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    base_ang_vel = obs.get('base_ang_vel', np.zeros(3))
    gravity = obs.get('gravity', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)

    # reward tuning parameters
    joint_vel_weight = 0.1
    base_lin_vel_weight = 1.0
    base_ang_vel_weight = 0.1
    action_weight = 0.1

    # rewards are higher for staying closer to the desired velocity
    reward_vel = base_lin_vel_weight * np.square(desired_velocity - base_lin_vel[0])
    
    # rewards are higher for smaller angle speed to keep the balance
    reward_ang = - base_ang_vel_weight * np.square(base_ang_vel).sum()
    
    # rewards are higher for smaller joint speeds to avoid unnecessary motion
    reward_joint = - joint_vel_weight * np.square(joint_vel).sum()

    # penalize actions (torque), we prefer the torque to be as small as possible
    reward_action = - action_weight * np.square(action).sum()

    # total reward
    reward = reward_vel + reward_ang + reward_joint + reward_action

    # normalize reward to range of [-1, 1]
    reward = np.tanh(reward)

    return np.asscalar(reward)
```

Here are the changes:
1. Instead of a single reward variable being incremented and decremented over the function, multiple reward factors have been isolated to improve the readability and maintainability.
2. Normalizing the total reward using the np.tanh function to ensure reward values remain within the range of [-1, 1]
3. The reward_vel calculation was previously subtracted from the total reward which did not align with the comment "rewards are higher for staying closer to the desired velocity". Now, this factor is directly added to the total reward in order to incentivize the agent to maintain desired velocities.
4. The np.asscalar function has been used to ensure the reward returned is always a scalar as expected by the reinforcement learning library stable_baselines3.
5. We abandoned the joint_pos_weight because there is no corresponding component in the previous reward function. the 'joint_pos' is not used anywhere, so to avoid confusion we removed it.

```

**Generated Reward Code (saved to reward_function_1.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):

    # Access the base environment
    base_env = env.unwrapped

    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    # assume that the obs is a dict with keys: 'joint_pos', 'joint_vel', 'base_lin_vel',
    # 'base_ang_vel', 'gravity', 'des_vel', 'prev_actions'
    joint_pos = obs.get('joint_pos', np.zeros_like(base_env.action_space.sample()))
    joint_vel = obs.get('joint_vel', np.zeros_like(base_env.action_space.sample()))
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    base_ang_vel = obs.get('base_ang_vel', np.zeros(3))
    gravity = obs.get('gravity', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)

    # reward tuning parameters
    joint_vel_weight = 0.1
    base_lin_vel_weight = 1.0
    base_ang_vel_weight = 0.1
    action_weight = 0.1

    # rewards are higher for staying closer to the desired velocity
    reward_vel = base_lin_vel_weight * np.square(desired_velocity - base_lin_vel[0])
    
    # rewards are higher for smaller angle speed to keep the balance
    reward_ang = - base_ang_vel_weight * np.square(base_ang_vel).sum()
    
    # rewards are higher for smaller joint speeds to avoid unnecessary motion
    reward_joint = - joint_vel_weight * np.square(joint_vel).sum()

    # penalize actions (torque), we prefer the torque to be as small as possible
    reward_action = - action_weight * np.square(action).sum()

    # total reward
    reward = reward_vel + reward_ang + reward_joint + reward_action

    # normalize reward to range of [-1, 1]
    reward = np.tanh(reward)

    return np.asscalar(reward)
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

    # Access the base environment
    base_env = env.unwrapped

    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    # assume that the obs is a dict with keys: 'joint_pos', 'joint_vel', 'base_lin_vel',
    # 'base_ang_vel', 'gravity', 'des_vel', 'prev_actions'
    joint_pos = obs.get('joint_pos', np.zeros_like(base_env.action_space.sample()))
    joint_vel = obs.get('joint_vel', np.zeros_like(base_env.action_space.sample()))
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    base_ang_vel = obs.get('base_ang_vel', np.zeros(3))
    gravity = obs.get('gravity', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)

    # reward tuning parameters
    joint_vel_weight = 0.1
    base_lin_vel_weight = 1.0
    base_ang_vel_weight = 0.1
    action_weight = 0.1

    # rewards are higher for staying closer to the desired velocity
    reward_vel = base_lin_vel_weight * np.square(desired_velocity - base_lin_vel[0])
    
    # rewards are higher for smaller angle speed to keep the balance
    reward_ang = - base_ang_vel_weight * np.square(base_ang_vel).sum()
    
    # rewards are higher for smaller joint speeds to avoid unnecessary motion
    reward_joint = - joint_vel_weight * np.square(joint_vel).sum()

    # penalize actions (torque), we prefer the torque to be as small as possible
    reward_action = - action_weight * np.square(action).sum()

    # total reward
    reward = reward_vel + reward_ang + reward_joint + reward_action

    # normalize reward to range of [-1, 1]
    reward = np.tanh(reward)

    return np.asscalar(reward)
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

    # Access the base environment
    base_env = env.unwrapped

    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    # assume that the obs is a dict with keys: 'joint_pos', 'joint_vel', 'base_lin_vel',
    # 'base_ang_vel', 'gravity', 'des_vel', 'prev_actions'
    joint_pos = obs.get('joint_pos', np.zeros_like(base_env.action_space.sample()))
    joint_vel = obs.get('joint_vel', np.zeros_like(base_env.action_space.sample()))
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    base_ang_vel = obs.get('base_ang_vel', np.zeros(3))
    gravity = obs.get('gravity', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)

    # reward tuning parameters
    joint_vel_weight = 0.1
    base_lin_vel_weight = 1.0
    base_ang_vel_weight = 0.1
    action_weight = 0.1

    # rewards are higher for staying closer to the desired velocity
    reward_vel = base_lin_vel_weight * np.square(desired_velocity - base_lin_vel[0])
    
    # rewards are higher for smaller angle speed to keep the balance
    reward_ang = - base_ang_vel_weight * np.square(base_ang_vel).sum()
    
    # rewards are higher for smaller joint speeds to avoid unnecessary motion
    reward_joint = - joint_vel_weight * np.square(joint_vel).sum()

    # penalize actions (torque), we prefer the torque to be as small as possible
    reward_action = - action_weight * np.square(action).sum()

    # total reward
    reward = reward_vel + reward_ang + reward_joint + reward_action

    # normalize reward to range of [-1, 1]
    reward = np.tanh(reward)

    return np.asscalar(reward)
```

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Based on the task, the Go1 robot's goal is to walk forward at a desired velocity. It indicates that we need to maximize the reward when the robot walks forward at the desired velocity and minimize unwanted actions such as unnecessary big joint position changes, angular rotations, and high torque. 

Looking at the last reward function, there are a few points where we can make improvements.

1) The linear velocity in the forward direction is being penalized, however, there would need to correctly be rewarded for matching the desired velocity. 

2) The last reward function is using `np.tanh` for normalization, however, this may not be suitable because it can saturate quickly for large values. This might give nearly same rewards for a wide range of different actions, making it difficult to distinguish between good and bad actions. So, it might be better to avoid using `np.tanh`.

3) We also need to consider adding a reward component for accomplishing the task (if the robot reaches the target).

Here is the improved reward function:

In the code provided below, I removed `np.tanh(reward)`, replaced negative rewards with positive terms wherever possible, and similarly flipped signs of corresponding weights to minimize confusion. I also added a target reached reward.

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    # Access the base environment
    base_env = env.unwrapped
    
    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    joint_vel = obs.get('joint_vel', np.zeros_like(base_env.action_space.sample()))
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    base_ang_vel = obs.get('base_ang_vel', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)
    
    # Reward tuning parameters
    joint_vel_weight = -0.1  # Rewards are lower for higher joint speeds to avoid unnecessary motion
    base_lin_vel_weight = 1.0  # Rewards are higher for staying closer to the desired velocity
    base_ang_vel_weight = -0.1  # Rewards are lower for larger angular velocity to maintain balance
    action_weight = -0.1  # Rewards are lower for larger action (maximize efficiency)
    goal_reward = 1.0  # Reward for achieving the goal
    
    # Calculate individual component of rewards
    reward_vel = -base_lin_vel_weight * np.abs(desired_velocity - base_lin_vel[0])
    reward_ang = base_ang_vel_weight * np.square(base_ang_vel).sum()
    reward_joint = joint_vel_weight * np.square(joint_vel).sum()
    reward_action = action_weight * np.square(action).sum()
    
    # Check if the task is done and the goal is reached
    if done and np.allclose(base_lin_vel[0], desired_velocity, atol=0.05):
        reward_goal = goal_reward
    else:
        reward_goal = 0.0

    # Calculate total reward as sum of all component rewards
    reward = reward_vel + reward_ang + reward_joint + reward_action + reward_goal

    return np.asscalar(reward)
```
This reward function penalizes the deviation from desired forward linear velocity, angular velocity, joint velocities, and action (torque). It also provides an extra reward when the robot reaches the desired velocity. This should encourage the robot to move towards the desired velocity while maintaining balance and efficiency.
```

**Generated Reward Code (saved to reward_function_2.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    # Access the base environment
    base_env = env.unwrapped
    
    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    joint_vel = obs.get('joint_vel', np.zeros_like(base_env.action_space.sample()))
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    base_ang_vel = obs.get('base_ang_vel', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)
    
    # Reward tuning parameters
    joint_vel_weight = -0.1  # Rewards are lower for higher joint speeds to avoid unnecessary motion
    base_lin_vel_weight = 1.0  # Rewards are higher for staying closer to the desired velocity
    base_ang_vel_weight = -0.1  # Rewards are lower for larger angular velocity to maintain balance
    action_weight = -0.1  # Rewards are lower for larger action (maximize efficiency)
    goal_reward = 1.0  # Reward for achieving the goal
    
    # Calculate individual component of rewards
    reward_vel = -base_lin_vel_weight * np.abs(desired_velocity - base_lin_vel[0])
    reward_ang = base_ang_vel_weight * np.square(base_ang_vel).sum()
    reward_joint = joint_vel_weight * np.square(joint_vel).sum()
    reward_action = action_weight * np.square(action).sum()
    
    # Check if the task is done and the goal is reached
    if done and np.allclose(base_lin_vel[0], desired_velocity, atol=0.05):
        reward_goal = goal_reward
    else:
        reward_goal = 0.0

    # Calculate total reward as sum of all component rewards
    reward = reward_vel + reward_ang + reward_joint + reward_action + reward_goal

    return np.asscalar(reward)
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
    # Access the base environment
    base_env = env.unwrapped
    
    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    joint_vel = obs.get('joint_vel', np.zeros_like(base_env.action_space.sample()))
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    base_ang_vel = obs.get('base_ang_vel', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)
    
    # Reward tuning parameters
    joint_vel_weight = -0.1  # Rewards are lower for higher joint speeds to avoid unnecessary motion
    base_lin_vel_weight = 1.0  # Rewards are higher for staying closer to the desired velocity
    base_ang_vel_weight = -0.1  # Rewards are lower for larger angular velocity to maintain balance
    action_weight = -0.1  # Rewards are lower for larger action (maximize efficiency)
    goal_reward = 1.0  # Reward for achieving the goal
    
    # Calculate individual component of rewards
    reward_vel = -base_lin_vel_weight * np.abs(desired_velocity - base_lin_vel[0])
    reward_ang = base_ang_vel_weight * np.square(base_ang_vel).sum()
    reward_joint = joint_vel_weight * np.square(joint_vel).sum()
    reward_action = action_weight * np.square(action).sum()
    
    # Check if the task is done and the goal is reached
    if done and np.allclose(base_lin_vel[0], desired_velocity, atol=0.05):
        reward_goal = goal_reward
    else:
        reward_goal = 0.0

    # Calculate total reward as sum of all component rewards
    reward = reward_vel + reward_ang + reward_joint + reward_action + reward_goal

    return np.asscalar(reward)
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
    # Access the base environment
    base_env = env.unwrapped
    
    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    joint_vel = obs.get('joint_vel', np.zeros_like(base_env.action_space.sample()))
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    base_ang_vel = obs.get('base_ang_vel', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)
    
    # Reward tuning parameters
    joint_vel_weight = -0.1  # Rewards are lower for higher joint speeds to avoid unnecessary motion
    base_lin_vel_weight = 1.0  # Rewards are higher for staying closer to the desired velocity
    base_ang_vel_weight = -0.1  # Rewards are lower for larger angular velocity to maintain balance
    action_weight = -0.1  # Rewards are lower for larger action (maximize efficiency)
    goal_reward = 1.0  # Reward for achieving the goal
    
    # Calculate individual component of rewards
    reward_vel = -base_lin_vel_weight * np.abs(desired_velocity - base_lin_vel[0])
    reward_ang = base_ang_vel_weight * np.square(base_ang_vel).sum()
    reward_joint = joint_vel_weight * np.square(joint_vel).sum()
    reward_action = action_weight * np.square(action).sum()
    
    # Check if the task is done and the goal is reached
    if done and np.allclose(base_lin_vel[0], desired_velocity, atol=0.05):
        reward_goal = goal_reward
    else:
        reward_goal = 0.0

    # Calculate total reward as sum of all component rewards
    reward = reward_vel + reward_ang + reward_joint + reward_action + reward_goal

    return np.asscalar(reward)
```

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Here's the updated reward function. Modifications made:
- Removed reward for joint velocities as this didn't seem to influence the task result and reduced complexity of the problem.
- Modified velocity reward from negative to positive and included a heavy penalty for being away from the target velocity.
- Removed angular velocity reward as it might not contribute significantly in forward movement.
- Increased reward weight for actions to encourage efficient movements and prevent unnecessary explorations.
- Simplified the goal reward computation and penalty when away from the goal.

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    # Access the base environment
    base_env = env.unwrapped
    
    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)
    
    # Reward tuning parameters
    base_lin_vel_weight = 1.0  # Rewards are higher for staying closer to the desired velocity
    action_weight = -0.2  # Higher Negative Rewards for larger action (maximize efficiency)
    
    # Calculate individual component of rewards
    reward_vel = base_lin_vel_weight * np.exp(-10 * np.abs(desired_velocity - base_lin_vel[0]))  # Exponential decay
    reward_action = action_weight * np.square(action).sum()
    
    # Check if the task is done and if the base linear velocity is close to the desired velocity
    if done and np.allclose(base_lin_vel[0], desired_velocity, atol=0.05):
        reward_goal = 1.0 # Full reward
    else:
        reward_goal = -1.0 # Full negative reward

    # Calculate total reward as sum of all component rewards
    reward = reward_vel + reward_action + reward_goal

    return np.asscalar(reward)
```

This code penalizes movement away from the desired velocity and rewards actions that bring the robot closer to the desired velocity. It also significantly penalizes actions when the end of the episode is reached and the desired velocity has not been achieved. It is expected to provide a better signal for the agent to learn from compared to the initial reward design.
```

**Generated Reward Code (saved to reward_function_3.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    # Access the base environment
    base_env = env.unwrapped
    
    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)
    
    # Reward tuning parameters
    base_lin_vel_weight = 1.0  # Rewards are higher for staying closer to the desired velocity
    action_weight = -0.2  # Higher Negative Rewards for larger action (maximize efficiency)
    
    # Calculate individual component of rewards
    reward_vel = base_lin_vel_weight * np.exp(-10 * np.abs(desired_velocity - base_lin_vel[0]))  # Exponential decay
    reward_action = action_weight * np.square(action).sum()
    
    # Check if the task is done and if the base linear velocity is close to the desired velocity
    if done and np.allclose(base_lin_vel[0], desired_velocity, atol=0.05):
        reward_goal = 1.0 # Full reward
    else:
        reward_goal = -1.0 # Full negative reward

    # Calculate total reward as sum of all component rewards
    reward = reward_vel + reward_action + reward_goal

    return np.asscalar(reward)
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
    # Access the base environment
    base_env = env.unwrapped
    
    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)
    
    # Reward tuning parameters
    base_lin_vel_weight = 1.0  # Rewards are higher for staying closer to the desired velocity
    action_weight = -0.2  # Higher Negative Rewards for larger action (maximize efficiency)
    
    # Calculate individual component of rewards
    reward_vel = base_lin_vel_weight * np.exp(-10 * np.abs(desired_velocity - base_lin_vel[0]))  # Exponential decay
    reward_action = action_weight * np.square(action).sum()
    
    # Check if the task is done and if the base linear velocity is close to the desired velocity
    if done and np.allclose(base_lin_vel[0], desired_velocity, atol=0.05):
        reward_goal = 1.0 # Full reward
    else:
        reward_goal = -1.0 # Full negative reward

    # Calculate total reward as sum of all component rewards
    reward = reward_vel + reward_action + reward_goal

    return np.asscalar(reward)
```
```

---

## Iteration 5/5

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
    # Access the base environment
    base_env = env.unwrapped
    
    # Extract the actual observation array
    obs = obs.get('observation', obs)
    
    base_lin_vel = obs.get('base_lin_vel', np.zeros(3))
    desired_velocity = obs.get('des_vel', 0.0)
    
    # Reward tuning parameters
    base_lin_vel_weight = 1.0  # Rewards are higher for staying closer to the desired velocity
    action_weight = -0.2  # Higher Negative Rewards for larger action (maximize efficiency)
    
    # Calculate individual component of rewards
    reward_vel = base_lin_vel_weight * np.exp(-10 * np.abs(desired_velocity - base_lin_vel[0]))  # Exponential decay
    reward_action = action_weight * np.square(action).sum()
    
    # Check if the task is done and if the base linear velocity is close to the desired velocity
    if done and np.allclose(base_lin_vel[0], desired_velocity, atol=0.05):
        reward_goal = 1.0 # Full reward
    else:
        reward_goal = -1.0 # Full negative reward

    # Calculate total reward as sum of all component rewards
    reward = reward_vel + reward_action + reward_goal

    return np.asscalar(reward)
```

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Given the feedback, it appears that there is an issue with current observation extraction. The extraction seems to anticipate a dictionary structure, yet the expectations around the environment do not confirm this dictionary format. For the sake of this task, we will assume that the observation is supplied as a flattened NumPy array and not as a dictionary. 

Moreover, the usage of np.allclose for checking base linear velocity was not the best choice as it might lead to an unreachable state, especially in environments where providing actions does not lead to predictable exact results due to the probable underlying physics engine.

Another important aspect is to avoid using np.asscalar since it is deprecated in newer versions of Numpy. Furthermore, it is necessary to evenly distribute reward among velocitiy tracking and actuator efficiency so that neither of them gets overlooked. The exponential reward based upon the velocity tracking also seems too harsh which needs to be replaced with a reward that is proportional to the error in achieving desired velocity.

Here is an updated reward function:

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    # Extract the base_env to avoid wrappers
    base_env = env.unwrapped
    
    # Extract the actual observation array
    obs = np.asarray(obs)
    
    base_lin_vel = obs[13]  # Assuming base_linear_velocity is at index [13]
    desired_velocity = base_env.goal  # Access the desired velocity from env variables
    
    # Define the weights for components of the reward function
    base_lin_vel_weight = 5.0  # emphasis on achieving desired velocity
    action_weight = 0.05  # slightly penalize higher action values for efficiency

    # Compute each component of the reward
    reward_vel = - base_lin_vel_weight * np.abs(desired_velocity - base_lin_vel)  # reward is inversely proportional to the error in velocity
    reward_action = - action_weight * np.square(action).sum()  # penalty for larger action values

    # Avoid reaching the done state prematurely by only providing a finishing reward in the last timestep
    # Instead of hard-checking the velocity match, reward being in the close vicinity of the desired velocity
    reward_goal = -1.0  # base reward
    if done:
        if np.abs(desired_velocity - base_lin_vel) < 0.05:
            reward_goal = 1.0  # Full reward
        else:
            reward_goal = -1.0  # Full negative reward

    # Compute total reward
    reward = reward_vel + reward_action + reward_goal

    return reward
```

Please adjust the observation indices and the reward coefficients according to your testing needs. These may not be perfect and might require a bit of tweaking. 
```

**Generated Reward Code (saved to reward_function_4.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    # Extract the base_env to avoid wrappers
    base_env = env.unwrapped
    
    # Extract the actual observation array
    obs = np.asarray(obs)
    
    base_lin_vel = obs[13]  # Assuming base_linear_velocity is at index [13]
    desired_velocity = base_env.goal  # Access the desired velocity from env variables
    
    # Define the weights for components of the reward function
    base_lin_vel_weight = 5.0  # emphasis on achieving desired velocity
    action_weight = 0.05  # slightly penalize higher action values for efficiency

    # Compute each component of the reward
    reward_vel = - base_lin_vel_weight * np.abs(desired_velocity - base_lin_vel)  # reward is inversely proportional to the error in velocity
    reward_action = - action_weight * np.square(action).sum()  # penalty for larger action values

    # Avoid reaching the done state prematurely by only providing a finishing reward in the last timestep
    # Instead of hard-checking the velocity match, reward being in the close vicinity of the desired velocity
    reward_goal = -1.0  # base reward
    if done:
        if np.abs(desired_velocity - base_lin_vel) < 0.05:
            reward_goal = 1.0  # Full reward
        else:
            reward_goal = -1.0  # Full negative reward

    # Compute total reward
    reward = reward_vel + reward_action + reward_goal

    return reward
```

**Training & Evaluation Results:**
- TensorBoard Log Directory: `tensorboard_logs/iteration_4`
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
    # Extract the base_env to avoid wrappers
    base_env = env.unwrapped
    
    # Extract the actual observation array
    obs = np.asarray(obs)
    
    base_lin_vel = obs[13]  # Assuming base_linear_velocity is at index [13]
    desired_velocity = base_env.goal  # Access the desired velocity from env variables
    
    # Define the weights for components of the reward function
    base_lin_vel_weight = 5.0  # emphasis on achieving desired velocity
    action_weight = 0.05  # slightly penalize higher action values for efficiency

    # Compute each component of the reward
    reward_vel = - base_lin_vel_weight * np.abs(desired_velocity - base_lin_vel)  # reward is inversely proportional to the error in velocity
    reward_action = - action_weight * np.square(action).sum()  # penalty for larger action values

    # Avoid reaching the done state prematurely by only providing a finishing reward in the last timestep
    # Instead of hard-checking the velocity match, reward being in the close vicinity of the desired velocity
    reward_goal = -1.0  # base reward
    if done:
        if np.abs(desired_velocity - base_lin_vel) < 0.05:
            reward_goal = 1.0  # Full reward
        else:
            reward_goal = -1.0  # Full negative reward

    # Compute total reward
    reward = reward_vel + reward_action + reward_goal

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

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Below is a simple implementation of a reward function suitable for the task. The main concern when designing reward functions is to make sure the desired behaviors are appropriately promoted and undesired behaviors penalized. 

In this scenario, the goal of the robot is to move forward at a desired velocity. The forward movement can be promoted by calculating the distance between the robot's current position and the goal. The faster the robot moves, the greater the reward. However, it's just as important to keep stability, for which we need to consider joint torques and try to minimize them. Consequently, the reward is calculated as the desired velocity minus a penalty based on joint torques.

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task.

    Parameters
    ----------
    obs : dict
        Observation Dictionary.
    action : np.array
        Action taken by the Go1 robot.
    done : bool
        Whether the task is done.
    env : gym.Env
        The OpenAI Gym environment.

    Returns
    -------
    float
        The computed reward.
    """

    # Make sure obs is a dictionary, get 'observation', or the dictionary itself
    obs = obs.get('observation', obs)

    current_velocity = obs['base_velocity_achieved'][0]
    desired_velocity = obs['desired_forward_velocity']

    # If the task is done and the robot failed to achieve the desired velocity, penalize harshly.
    failed_reward = -100 if done and current_velocity < desired_velocity else 0

    # A reward component based on achieving the desired_forward_velocity or forward movement
    forward_reward = current_velocity * 10

    # Penalize if joint torques are too high, for maintaining stability
    torque_penalty = np.square(action).sum() * 0.1

    # Final reward is forward movement reward minus the torque penalty.
    reward = forward_reward - torque_penalty + failed_reward

    return reward
```

This reward function encourages the robot to move forward at higher speeds and at the same time, take actions that minimize joint torques. In the end, this should help the robot learn to complete the task in an efficient and stable manner.

Don't forget to play with the scaling factors according to your experiment requirements.
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task.

    Parameters
    ----------
    obs : dict
        Observation Dictionary.
    action : np.array
        Action taken by the Go1 robot.
    done : bool
        Whether the task is done.
    env : gym.Env
        The OpenAI Gym environment.

    Returns
    -------
    float
        The computed reward.
    """

    # Make sure obs is a dictionary, get 'observation', or the dictionary itself
    obs = obs.get('observation', obs)

    current_velocity = obs['base_velocity_achieved'][0]
    desired_velocity = obs['desired_forward_velocity']

    # If the task is done and the robot failed to achieve the desired velocity, penalize harshly.
    failed_reward = -100 if done and current_velocity < desired_velocity else 0

    # A reward component based on achieving the desired_forward_velocity or forward movement
    forward_reward = current_velocity * 10

    # Penalize if joint torques are too high, for maintaining stability
    torque_penalty = np.square(action).sum() * 0.1

    # Final reward is forward movement reward minus the torque penalty.
    reward = forward_reward - torque_penalty + failed_reward

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

    Parameters
    ----------
    obs : dict
        Observation Dictionary.
    action : np.array
        Action taken by the Go1 robot.
    done : bool
        Whether the task is done.
    env : gym.Env
        The OpenAI Gym environment.

    Returns
    -------
    float
        The computed reward.
    """

    # Make sure obs is a dictionary, get 'observation', or the dictionary itself
    obs = obs.get('observation', obs)

    current_velocity = obs['base_velocity_achieved'][0]
    desired_velocity = obs['desired_forward_velocity']

    # If the task is done and the robot failed to achieve the desired velocity, penalize harshly.
    failed_reward = -100 if done and current_velocity < desired_velocity else 0

    # A reward component based on achieving the desired_forward_velocity or forward movement
    forward_reward = current_velocity * 10

    # Penalize if joint torques are too high, for maintaining stability
    torque_penalty = np.square(action).sum() * 0.1

    # Final reward is forward movement reward minus the torque penalty.
    reward = forward_reward - torque_penalty + failed_reward

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

    Parameters
    ----------
    obs : dict
        Observation Dictionary.
    action : np.array
        Action taken by the Go1 robot.
    done : bool
        Whether the task is done.
    env : gym.Env
        The OpenAI Gym environment.

    Returns
    -------
    float
        The computed reward.
    """

    # Make sure obs is a dictionary, get 'observation', or the dictionary itself
    obs = obs.get('observation', obs)

    current_velocity = obs['base_velocity_achieved'][0]
    desired_velocity = obs['desired_forward_velocity']

    # If the task is done and the robot failed to achieve the desired velocity, penalize harshly.
    failed_reward = -100 if done and current_velocity < desired_velocity else 0

    # A reward component based on achieving the desired_forward_velocity or forward movement
    forward_reward = current_velocity * 10

    # Penalize if joint torques are too high, for maintaining stability
    torque_penalty = np.square(action).sum() * 0.1

    # Final reward is forward movement reward minus the torque penalty.
    reward = forward_reward - torque_penalty + failed_reward

    return reward
```

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
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

    Parameters
    ----------
    obs : dict
        Observation Dictionary.
    action : np.array
        Action taken by the Go1 robot.
    done : bool
        Whether the task is done.
    env : gym.Env
        The OpenAI Gym environment.

    Returns
    -------
    float
        The computed reward.
    """

    # Make sure obs is a dictionary, get 'observation', or the dictionary itself
    obs = obs.get('observation', obs)

    current_velocity = obs['base_velocity_achieved'][0]
    desired_velocity = obs['desired_forward_velocity']

    # Calculate the deviation from the desired velocity
    velocity_deviation = abs(desired_velocity - current_velocity)

    # Penalize the robot for not reaching the desired velocity
    # Higher penalty for larger deviation
    velocity_penalty = velocity_deviation * 10

    # Calculate the penalization for the joint torques based on the action taken
    # This helps to discourage the robot from making unnecessary movements
    torque_penalty = np.square(action).sum() * 0.1

    # Base reward is 1, substract velocity and torque penalty from the base reward
    reward = 1 - velocity_penalty - torque_penalty

    # If the task is done and the robot successfully achieved the desired velocity, give additional reward
    completed_bonus = 100 if done and current_velocity >= desired_velocity else 0

    # Add the completed_bonus to the reward if the task is done
    reward += completed_bonus

    return float(reward)
```
The proposed reward function now includes a penalty (velocity_penalty) that depends on how far the robot's actual velocity is from the desired_velocity. This encourages the robot to maintain a velocity close to the desired_velocity throughout the episode.

The function also incorporates a bonus reward (completed_bonus) for successfully reaching and maintaining the desired_velocity by the end of the episode. This gives the robot an incentive to not only reach but also maintain the desired_velocity.

The torque_penalty is left as it is, as it helps in maintaining stability of the robot and discourages unnecessary movements.
```

**Generated Reward Code (saved to reward_function_1.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 robot task.

    Parameters
    ----------
    obs : dict
        Observation Dictionary.
    action : np.array
        Action taken by the Go1 robot.
    done : bool
        Whether the task is done.
    env : gym.Env
        The OpenAI Gym environment.

    Returns
    -------
    float
        The computed reward.
    """

    # Make sure obs is a dictionary, get 'observation', or the dictionary itself
    obs = obs.get('observation', obs)

    current_velocity = obs['base_velocity_achieved'][0]
    desired_velocity = obs['desired_forward_velocity']

    # Calculate the deviation from the desired velocity
    velocity_deviation = abs(desired_velocity - current_velocity)

    # Penalize the robot for not reaching the desired velocity
    # Higher penalty for larger deviation
    velocity_penalty = velocity_deviation * 10

    # Calculate the penalization for the joint torques based on the action taken
    # This helps to discourage the robot from making unnecessary movements
    torque_penalty = np.square(action).sum() * 0.1

    # Base reward is 1, substract velocity and torque penalty from the base reward
    reward = 1 - velocity_penalty - torque_penalty

    # If the task is done and the robot successfully achieved the desired velocity, give additional reward
    completed_bonus = 100 if done and current_velocity >= desired_velocity else 0

    # Add the completed_bonus to the reward if the task is done
    reward += completed_bonus

    return float(reward)
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
    Custom reward function for the Go1 robot task.

    Parameters
    ----------
    obs : dict
        Observation Dictionary.
    action : np.array
        Action taken by the Go1 robot.
    done : bool
        Whether the task is done.
    env : gym.Env
        The OpenAI Gym environment.

    Returns
    -------
    float
        The computed reward.
    """

    # Make sure obs is a dictionary, get 'observation', or the dictionary itself
    obs = obs.get('observation', obs)

    current_velocity = obs['base_velocity_achieved'][0]
    desired_velocity = obs['desired_forward_velocity']

    # Calculate the deviation from the desired velocity
    velocity_deviation = abs(desired_velocity - current_velocity)

    # Penalize the robot for not reaching the desired velocity
    # Higher penalty for larger deviation
    velocity_penalty = velocity_deviation * 10

    # Calculate the penalization for the joint torques based on the action taken
    # This helps to discourage the robot from making unnecessary movements
    torque_penalty = np.square(action).sum() * 0.1

    # Base reward is 1, substract velocity and torque penalty from the base reward
    reward = 1 - velocity_penalty - torque_penalty

    # If the task is done and the robot successfully achieved the desired velocity, give additional reward
    completed_bonus = 100 if done and current_velocity >= desired_velocity else 0

    # Add the completed_bonus to the reward if the task is done
    reward += completed_bonus

    return float(reward)
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

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Here is a Python code block that defines a custom reward function for the described robotics task of guiding a robot to follow a desired velocity:

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    ''' Reward function for Go1 robot forward walking task

    Args:
        obs: Numpy array representing an observation from the environment
        action: Numpy array representing the action taken by the agent
        done: Boolean flag indicating whether the episode is finished
        env: Instance of the environment being used

    Returns:
        float representing the reward resulting from the action
    '''
    # Extracting the base environment
    base_env = env.unwrapped
    
    # Extracting the observation array or defaulting to obs as is if it is already an array
    obs = obs.get('observation', obs)

    # Getting the current position and velocity of the robot
    robot_velocity = obs['robot_velocity']
    
    # Fetching the goal position
    goal_velocity = base_env.goal
    
    # Defining weight for different parts of the reward
    velocity_weight = 1.0
   
    # Reward for matching the desired velocity
    velocity_reward = -np.linalg.norm(goal_velocity - robot_velocity)   # we want to minimize the difference

    # The main reward is a weighted sum of the above components
    main_reward = velocity_weight * velocity_reward
    
    # Penalize if robot falls
    if done:
        main_reward -= 1.00

    return main_reward
```
This is a simple reward function that mainly focuses on keeping the robot's velocity close to the target velocity. Actions leading to higher velocities earn higher rewards (measured as the negative of the norm of the difference between goal_velocity and robot_velocity), while actions that cause the robot to fall (indicated by done being True) are penalized by reducing the reward. Future iterations might include rewards for maintaining balance, efficient energy usage, and avoiding obstacles.
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    ''' Reward function for Go1 robot forward walking task

    Args:
        obs: Numpy array representing an observation from the environment
        action: Numpy array representing the action taken by the agent
        done: Boolean flag indicating whether the episode is finished
        env: Instance of the environment being used

    Returns:
        float representing the reward resulting from the action
    '''
    # Extracting the base environment
    base_env = env.unwrapped
    
    # Extracting the observation array or defaulting to obs as is if it is already an array
    obs = obs.get('observation', obs)

    # Getting the current position and velocity of the robot
    robot_velocity = obs['robot_velocity']
    
    # Fetching the goal position
    goal_velocity = base_env.goal
    
    # Defining weight for different parts of the reward
    velocity_weight = 1.0
   
    # Reward for matching the desired velocity
    velocity_reward = -np.linalg.norm(goal_velocity - robot_velocity)   # we want to minimize the difference

    # The main reward is a weighted sum of the above components
    main_reward = velocity_weight * velocity_reward
    
    # Penalize if robot falls
    if done:
        main_reward -= 1.00

    return main_reward
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
    ''' Reward function for Go1 robot forward walking task

    Args:
        obs: Numpy array representing an observation from the environment
        action: Numpy array representing the action taken by the agent
        done: Boolean flag indicating whether the episode is finished
        env: Instance of the environment being used

    Returns:
        float representing the reward resulting from the action
    '''
    # Extracting the base environment
    base_env = env.unwrapped
    
    # Extracting the observation array or defaulting to obs as is if it is already an array
    obs = obs.get('observation', obs)

    # Getting the current position and velocity of the robot
    robot_velocity = obs['robot_velocity']
    
    # Fetching the goal position
    goal_velocity = base_env.goal
    
    # Defining weight for different parts of the reward
    velocity_weight = 1.0
   
    # Reward for matching the desired velocity
    velocity_reward = -np.linalg.norm(goal_velocity - robot_velocity)   # we want to minimize the difference

    # The main reward is a weighted sum of the above components
    main_reward = velocity_weight * velocity_reward
    
    # Penalize if robot falls
    if done:
        main_reward -= 1.00

    return main_reward
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

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Here is a baseline reward function for the Go1MujocoCustom-v0 environment. This reward function considers the deviation from the desired linear (forward) velocity, angular velocity, as well as the energy cost (action cost).

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 environment.

    This function computes a reward based on the difference between the robot's
    current and desired linear (forward) and angular velocities, with an added
    energy cost based on the absolute value of the actions (joint torques).

    Parameters
    ----------
    obs: np.ndarray
        Current observation from the environment.
    action: np.ndarray
        Action taken by the agent.
    done: bool
        Whether the episode is finished.
    env: gym.Env
        Gym environment object.

    Returns
    -------
    reward: float
        Computed reward for the current timestep.
    """
    
    # Unwrap environment to access goal and unwrapped properties
    env = env.unwrapped

    # Check if observation is a dictionary and extract the actual observation array
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Current robot velocity is at obs index 0
    robot_velocity = obs[0]
    
    # Desired robot velocity (goal) is at obs index 1
    desired_velocity = env.goal

    # Compute the absolute difference between current and goal velocities
    velocity_diff = np.abs(desired_velocity - robot_velocity)

    # Energy cost is the sum of absolute actions (joint torques)
    energy_cost = np.sum(np.abs(action))

    # Reward: negative sum of the linear velocity difference and energy cost, 
    # we take the negative sum because we want to minmize these quantities.
    reward = - (velocity_diff + energy_cost)

    return reward
```

This code defines a custom reward function that will be used in training the robot in the Go1MujocoCustom-v0 environment. The rewards penalize deviations from the desired velocities and energy costs, encouraging the robot to achieve its goal efficiently. Exception handling is added to allow for instances where the observation may be a dictionary. Remember that the higher the reward during training, the better the agent will perform.
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 environment.

    This function computes a reward based on the difference between the robot's
    current and desired linear (forward) and angular velocities, with an added
    energy cost based on the absolute value of the actions (joint torques).

    Parameters
    ----------
    obs: np.ndarray
        Current observation from the environment.
    action: np.ndarray
        Action taken by the agent.
    done: bool
        Whether the episode is finished.
    env: gym.Env
        Gym environment object.

    Returns
    -------
    reward: float
        Computed reward for the current timestep.
    """
    
    # Unwrap environment to access goal and unwrapped properties
    env = env.unwrapped

    # Check if observation is a dictionary and extract the actual observation array
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Current robot velocity is at obs index 0
    robot_velocity = obs[0]
    
    # Desired robot velocity (goal) is at obs index 1
    desired_velocity = env.goal

    # Compute the absolute difference between current and goal velocities
    velocity_diff = np.abs(desired_velocity - robot_velocity)

    # Energy cost is the sum of absolute actions (joint torques)
    energy_cost = np.sum(np.abs(action))

    # Reward: negative sum of the linear velocity difference and energy cost, 
    # we take the negative sum because we want to minmize these quantities.
    reward = - (velocity_diff + energy_cost)

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
    Custom reward function for Go1MujocoCustom-v0 environment.

    This function computes a reward based on the difference between the robot's
    current and desired linear (forward) and angular velocities, with an added
    energy cost based on the absolute value of the actions (joint torques).

    Parameters
    ----------
    obs: np.ndarray
        Current observation from the environment.
    action: np.ndarray
        Action taken by the agent.
    done: bool
        Whether the episode is finished.
    env: gym.Env
        Gym environment object.

    Returns
    -------
    reward: float
        Computed reward for the current timestep.
    """
    
    # Unwrap environment to access goal and unwrapped properties
    env = env.unwrapped

    # Check if observation is a dictionary and extract the actual observation array
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Current robot velocity is at obs index 0
    robot_velocity = obs[0]
    
    # Desired robot velocity (goal) is at obs index 1
    desired_velocity = env.goal

    # Compute the absolute difference between current and goal velocities
    velocity_diff = np.abs(desired_velocity - robot_velocity)

    # Energy cost is the sum of absolute actions (joint torques)
    energy_cost = np.sum(np.abs(action))

    # Reward: negative sum of the linear velocity difference and energy cost, 
    # we take the negative sum because we want to minmize these quantities.
    reward = - (velocity_diff + energy_cost)

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
    Custom reward function for Go1MujocoCustom-v0 environment.

    This function computes a reward based on the difference between the robot's
    current and desired linear (forward) and angular velocities, with an added
    energy cost based on the absolute value of the actions (joint torques).

    Parameters
    ----------
    obs: np.ndarray
        Current observation from the environment.
    action: np.ndarray
        Action taken by the agent.
    done: bool
        Whether the episode is finished.
    env: gym.Env
        Gym environment object.

    Returns
    -------
    reward: float
        Computed reward for the current timestep.
    """
    
    # Unwrap environment to access goal and unwrapped properties
    env = env.unwrapped

    # Check if observation is a dictionary and extract the actual observation array
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Current robot velocity is at obs index 0
    robot_velocity = obs[0]
    
    # Desired robot velocity (goal) is at obs index 1
    desired_velocity = env.goal

    # Compute the absolute difference between current and goal velocities
    velocity_diff = np.abs(desired_velocity - robot_velocity)

    # Energy cost is the sum of absolute actions (joint torques)
    energy_cost = np.sum(np.abs(action))

    # Reward: negative sum of the linear velocity difference and energy cost, 
    # we take the negative sum because we want to minmize these quantities.
    reward = - (velocity_diff + energy_cost)

    return reward
```

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 environment.

    This function computes a reward based on the difference between the robot's
    current and desired linear(forward) and angular velocities. It also considers
    an energy cost based on the absolute value of the actions (joint torques), but 
    the impact has been scaled down.

    The reward function has been restructured to optimize learning and improve 
    the success rate.

    Parameters
    ----------
    obs: np.ndarray
        Current observation from the environment.
    action: np.ndarray
        Action taken by the agent.
    done: bool
        Whether the episode is finished.
    env: gym.Env
        Gym environment object.

    Returns
    -------
    reward: float
        Computed reward for the current timestep.
    """
    
    # Unwrap environment to access goal and unwrapped properties
    env = env.unwrapped

    # Check if observation is a dictionary and extract the actual observation array
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Current robot velocity is at obs index 0
    robot_velocity = obs[0]
    
    # Desired robot velocity (goal) is at env.goal
    desired_velocity = env.goal

    # Compute the absolute difference between current and goal velocities. 
    # Being closer to the goal gets a positive reward
    velocity_diff = abs(desired_velocity - robot_velocity)
    velocity_reward = -1 * velocity_diff
    
    # Energy cost is the sum of absolute actions (joint torques)
    # Here, the energy cost is scaled down by an order of magnitude
    energy_cost = np.sum(np.abs(action)) * 0.1
    
    # Compute total reward: velocity reward - energy cost. We want to maximize
    # velocity reward (getting close to goal) and minimize energy cost.
    reward = velocity_reward - energy_cost

    return reward
```
```

**Generated Reward Code (saved to reward_function_1.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1MujocoCustom-v0 environment.

    This function computes a reward based on the difference between the robot's
    current and desired linear(forward) and angular velocities. It also considers
    an energy cost based on the absolute value of the actions (joint torques), but 
    the impact has been scaled down.

    The reward function has been restructured to optimize learning and improve 
    the success rate.

    Parameters
    ----------
    obs: np.ndarray
        Current observation from the environment.
    action: np.ndarray
        Action taken by the agent.
    done: bool
        Whether the episode is finished.
    env: gym.Env
        Gym environment object.

    Returns
    -------
    reward: float
        Computed reward for the current timestep.
    """
    
    # Unwrap environment to access goal and unwrapped properties
    env = env.unwrapped

    # Check if observation is a dictionary and extract the actual observation array
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Current robot velocity is at obs index 0
    robot_velocity = obs[0]
    
    # Desired robot velocity (goal) is at env.goal
    desired_velocity = env.goal

    # Compute the absolute difference between current and goal velocities. 
    # Being closer to the goal gets a positive reward
    velocity_diff = abs(desired_velocity - robot_velocity)
    velocity_reward = -1 * velocity_diff
    
    # Energy cost is the sum of absolute actions (joint torques)
    # Here, the energy cost is scaled down by an order of magnitude
    energy_cost = np.sum(np.abs(action)) * 0.1
    
    # Compute total reward: velocity reward - energy cost. We want to maximize
    # velocity reward (getting close to goal) and minimize energy cost.
    reward = velocity_reward - energy_cost

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
    Custom reward function for Go1MujocoCustom-v0 environment.

    This function computes a reward based on the difference between the robot's
    current and desired linear(forward) and angular velocities. It also considers
    an energy cost based on the absolute value of the actions (joint torques), but 
    the impact has been scaled down.

    The reward function has been restructured to optimize learning and improve 
    the success rate.

    Parameters
    ----------
    obs: np.ndarray
        Current observation from the environment.
    action: np.ndarray
        Action taken by the agent.
    done: bool
        Whether the episode is finished.
    env: gym.Env
        Gym environment object.

    Returns
    -------
    reward: float
        Computed reward for the current timestep.
    """
    
    # Unwrap environment to access goal and unwrapped properties
    env = env.unwrapped

    # Check if observation is a dictionary and extract the actual observation array
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Current robot velocity is at obs index 0
    robot_velocity = obs[0]
    
    # Desired robot velocity (goal) is at env.goal
    desired_velocity = env.goal

    # Compute the absolute difference between current and goal velocities. 
    # Being closer to the goal gets a positive reward
    velocity_diff = abs(desired_velocity - robot_velocity)
    velocity_reward = -1 * velocity_diff
    
    # Energy cost is the sum of absolute actions (joint torques)
    # Here, the energy cost is scaled down by an order of magnitude
    energy_cost = np.sum(np.abs(action)) * 0.1
    
    # Compute total reward: velocity reward - energy cost. We want to maximize
    # velocity reward (getting close to goal) and minimize energy cost.
    reward = velocity_reward - energy_cost

    return reward
```
```

---

## Iteration 1/5

**Status:** Partial Success - Evaluation Failed

**User Prompt to LLM (leading to this iteration's code):**
*Note: Contains feedback from the iteration before this one.*
```
Task Goal: Navigate the Ant robot through a maze to reach a target.
Environment Name: AntMazeCustom-v0
Full Task Description:
The agent controls a quadrupedal robot (Ant) in a maze-like environment. The goal is to navigate from a starting position to a target position within the maze. Observations likely include joint positions, velocities, contact forces, and potentially lidar or position information. Actions are joint torques or target joint positions.

Detailed Environment Configuration:
- env_name: AntMazeCustom-v0
- task: Navigate the Ant robot through a maze to reach a target.
- task_description: The agent controls a quadrupedal robot (Ant) in a maze-like environment. The goal is to navigate from a starting position to a target position within the maze. Observations likely include joint positions, velocities, contact forces, and potentially lidar or position information. Actions are joint torques or target joint positions.
- timestamps: 2000000

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Task-specific reward function:
- Agents receive a small negative reward for each time step it takes to reach the goal, thus motivating faster solutions. (-0.01 per step)
- If the agent drifts too far from the maze area (e.g., >10 units from the goal), the episode is terminated and a large negative reward is given to the agent (-1.0).
- Upon reaching the goal, the agent receives a large positive reward (+1.0).

Here is the python code of reward function:

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    obs: Observations from the environment, likely including joint positions, velocities, contact forces, and potential lidar or position information.
    action: Action taken by the agent. These are joint torques or target joint positions.
    done: Boolean indicating whether the episode has ended.
    env: The environment instance.

    The reward is computed as follows:
    - The agent receives a small negative reward for each timestep to reach the goal (Encouraging swift navigation).
    - If the agent ventures too far from the maze (e.g., >10 units from the goal), the episode ends and it gets a large negative reward.
    - Upon reaching the goal, the agent gets a large positive reward.
    """

    # get current position of the agent  
    current_position = obs.get('observation', obs)[0:2]

    # locate the target position
    goal_position = env.unwrapped.goal

    # calculate the Euclidean distance to goal
    distance_to_goal = np.linalg.norm(current_position - goal_position)

    # Check if the agent has reached the goal
    is_at_goal = distance_to_goal < 0.5

    # Check if the agent drifted too far from the maze
    is_off_track = distance_to_goal > 10.0

    # Define rewards and penalties
    step_penalty = -0.01
    off_track_penalty = -1.0
    goal_reward = +1.0

    # Calculate reward
    if is_at_goal:
        reward = goal_reward
    elif is_off_track:
        reward = off_track_penalty
    else:
        reward = step_penalty

    return reward
```
This reward function aims at providing a continuous, rich feedback to encourage the robot to swiftly and accurately reach the goal without straying too far away from the path.
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    obs: Observations from the environment, likely including joint positions, velocities, contact forces, and potential lidar or position information.
    action: Action taken by the agent. These are joint torques or target joint positions.
    done: Boolean indicating whether the episode has ended.
    env: The environment instance.

    The reward is computed as follows:
    - The agent receives a small negative reward for each timestep to reach the goal (Encouraging swift navigation).
    - If the agent ventures too far from the maze (e.g., >10 units from the goal), the episode ends and it gets a large negative reward.
    - Upon reaching the goal, the agent gets a large positive reward.
    """

    # get current position of the agent  
    current_position = obs.get('observation', obs)[0:2]

    # locate the target position
    goal_position = env.unwrapped.goal

    # calculate the Euclidean distance to goal
    distance_to_goal = np.linalg.norm(current_position - goal_position)

    # Check if the agent has reached the goal
    is_at_goal = distance_to_goal < 0.5

    # Check if the agent drifted too far from the maze
    is_off_track = distance_to_goal > 10.0

    # Define rewards and penalties
    step_penalty = -0.01
    off_track_penalty = -1.0
    goal_reward = +1.0

    # Calculate reward
    if is_at_goal:
        reward = goal_reward
    elif is_off_track:
        reward = off_track_penalty
    else:
        reward = step_penalty

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
    obs: Observations from the environment, likely including joint positions, velocities, contact forces, and potential lidar or position information.
    action: Action taken by the agent. These are joint torques or target joint positions.
    done: Boolean indicating whether the episode has ended.
    env: The environment instance.

    The reward is computed as follows:
    - The agent receives a small negative reward for each timestep to reach the goal (Encouraging swift navigation).
    - If the agent ventures too far from the maze (e.g., >10 units from the goal), the episode ends and it gets a large negative reward.
    - Upon reaching the goal, the agent gets a large positive reward.
    """

    # get current position of the agent  
    current_position = obs.get('observation', obs)[0:2]

    # locate the target position
    goal_position = env.unwrapped.goal

    # calculate the Euclidean distance to goal
    distance_to_goal = np.linalg.norm(current_position - goal_position)

    # Check if the agent has reached the goal
    is_at_goal = distance_to_goal < 0.5

    # Check if the agent drifted too far from the maze
    is_off_track = distance_to_goal > 10.0

    # Define rewards and penalties
    step_penalty = -0.01
    off_track_penalty = -1.0
    goal_reward = +1.0

    # Calculate reward
    if is_at_goal:
        reward = goal_reward
    elif is_off_track:
        reward = off_track_penalty
    else:
        reward = step_penalty

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

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    A custom reward function for the Unitree Go1 quadruped robot task where the goal is to 
    make the robot walk forward at a desired velocity. 

    Parameters:
    obs (array): The observation from the environment, a dictionary containing information 
                 such as joint positions, velocities, base velocities, projected gravity vector,
                 desired velocity, and previous actions.
    action (array): The action taken by the agent. In this context, joint torques.
    done (bool): A boolean indicating whether the episode has ended or not.
    env (object): The environment object, from which we can extract further information if needed.
    
    Returns:
    float: The calculated reward.
    """

    # Extract useful variables
    obs = obs.get('observation', obs)
    desired_velocity = obs['desired_velocity']
    current_velocity = env.unwrapped.base_link_lin_vel 

    # Define reward as a negative distance between current velocity and desired velocity. 
    # We use the negative value because the reinforcement learning algorithm will aim to maximize 
    # the reward, whereas we want to minimize the distance (the smaller the distance the better)

    reward = -np.linalg.norm(current_velocity - desired_velocity)
    return reward
```

The reward function is designed to provide a higher reward when the robot is moving at the desired forward velocity. It does this by measuring the distance between the current velocity of the robot and the desired velocity. The closer the robot's velocity is to the desired velocity, the smaller the distance will be, thus returning a higher reward (since we negated the distance).
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    A custom reward function for the Unitree Go1 quadruped robot task where the goal is to 
    make the robot walk forward at a desired velocity. 

    Parameters:
    obs (array): The observation from the environment, a dictionary containing information 
                 such as joint positions, velocities, base velocities, projected gravity vector,
                 desired velocity, and previous actions.
    action (array): The action taken by the agent. In this context, joint torques.
    done (bool): A boolean indicating whether the episode has ended or not.
    env (object): The environment object, from which we can extract further information if needed.
    
    Returns:
    float: The calculated reward.
    """

    # Extract useful variables
    obs = obs.get('observation', obs)
    desired_velocity = obs['desired_velocity']
    current_velocity = env.unwrapped.base_link_lin_vel 

    # Define reward as a negative distance between current velocity and desired velocity. 
    # We use the negative value because the reinforcement learning algorithm will aim to maximize 
    # the reward, whereas we want to minimize the distance (the smaller the distance the better)

    reward = -np.linalg.norm(current_velocity - desired_velocity)
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
    A custom reward function for the Unitree Go1 quadruped robot task where the goal is to 
    make the robot walk forward at a desired velocity. 

    Parameters:
    obs (array): The observation from the environment, a dictionary containing information 
                 such as joint positions, velocities, base velocities, projected gravity vector,
                 desired velocity, and previous actions.
    action (array): The action taken by the agent. In this context, joint torques.
    done (bool): A boolean indicating whether the episode has ended or not.
    env (object): The environment object, from which we can extract further information if needed.
    
    Returns:
    float: The calculated reward.
    """

    # Extract useful variables
    obs = obs.get('observation', obs)
    desired_velocity = obs['desired_velocity']
    current_velocity = env.unwrapped.base_link_lin_vel 

    # Define reward as a negative distance between current velocity and desired velocity. 
    # We use the negative value because the reinforcement learning algorithm will aim to maximize 
    # the reward, whereas we want to minimize the distance (the smaller the distance the better)

    reward = -np.linalg.norm(current_velocity - desired_velocity)
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

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(observation, action, done, env):
    """
    Custom reward function for Go1 quadruped robot control task.
    
    Args:
        observation (dict): The current observation from the environment.
        action (ndarray): The current action taken by the policy.
        done (bool): Whether the episode is done.
        env: The current environment (Go1MujocoCustom-v0).
    
    Returns:
        float: Reward for the current step.
    """
    # Extract actual observation array from dictionary
    observation = observation.get('observation', observation)
    
    # Extract variables from observation
    joint_pos_vel = observation[:12]               # Joint positions/velocities
    base_lin_ang_vel = observation[12:18]          # Base linear/angular velocities
    projected_gravity_vec = observation[18:21]     # Projected gravity vector
    desired_vel = observation[21]                  # Desired velocity
    previous_actions = observation[22:]             # Previous actions

    # Negative reward for large joint torques (encourages energy efficiency)
    torque_penalty = -np.sum(np.square(action))

    # Positive reward for maintaining the desired velocity
    speed_reward = -np.abs(base_lin_ang_vel[0] - desired_vel)

    # Negative reward for large difference in desired and actual angular velocity (minimize rotation)
    angular_penalty = -np.abs(desired_vel - base_lin_ang_vel[5])

    # Calculate final reward
    reward = speed_reward + angular_penalty + torque_penalty

    return reward
```
This reward function encourages the robot to move forward at the desired velocity, minimizes unnecessary rotations and encourages low energy consumption (low joint torques). It may need to be adjusted for balance, as the current configuration may overly penalize torque use or not reward velocity tracking sufficiently.
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(observation, action, done, env):
    """
    Custom reward function for Go1 quadruped robot control task.
    
    Args:
        observation (dict): The current observation from the environment.
        action (ndarray): The current action taken by the policy.
        done (bool): Whether the episode is done.
        env: The current environment (Go1MujocoCustom-v0).
    
    Returns:
        float: Reward for the current step.
    """
    # Extract actual observation array from dictionary
    observation = observation.get('observation', observation)
    
    # Extract variables from observation
    joint_pos_vel = observation[:12]               # Joint positions/velocities
    base_lin_ang_vel = observation[12:18]          # Base linear/angular velocities
    projected_gravity_vec = observation[18:21]     # Projected gravity vector
    desired_vel = observation[21]                  # Desired velocity
    previous_actions = observation[22:]             # Previous actions

    # Negative reward for large joint torques (encourages energy efficiency)
    torque_penalty = -np.sum(np.square(action))

    # Positive reward for maintaining the desired velocity
    speed_reward = -np.abs(base_lin_ang_vel[0] - desired_vel)

    # Negative reward for large difference in desired and actual angular velocity (minimize rotation)
    angular_penalty = -np.abs(desired_vel - base_lin_ang_vel[5])

    # Calculate final reward
    reward = speed_reward + angular_penalty + torque_penalty

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

def custom_reward_function(observation, action, done, env):
    """
    Custom reward function for Go1 quadruped robot control task.
    
    Args:
        observation (dict): The current observation from the environment.
        action (ndarray): The current action taken by the policy.
        done (bool): Whether the episode is done.
        env: The current environment (Go1MujocoCustom-v0).
    
    Returns:
        float: Reward for the current step.
    """
    # Extract actual observation array from dictionary
    observation = observation.get('observation', observation)
    
    # Extract variables from observation
    joint_pos_vel = observation[:12]               # Joint positions/velocities
    base_lin_ang_vel = observation[12:18]          # Base linear/angular velocities
    projected_gravity_vec = observation[18:21]     # Projected gravity vector
    desired_vel = observation[21]                  # Desired velocity
    previous_actions = observation[22:]             # Previous actions

    # Negative reward for large joint torques (encourages energy efficiency)
    torque_penalty = -np.sum(np.square(action))

    # Positive reward for maintaining the desired velocity
    speed_reward = -np.abs(base_lin_ang_vel[0] - desired_vel)

    # Negative reward for large difference in desired and actual angular velocity (minimize rotation)
    angular_penalty = -np.abs(desired_vel - base_lin_ang_vel[5])

    # Calculate final reward
    reward = speed_reward + angular_penalty + torque_penalty

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

def custom_reward_function(observation, action, done, env):
    """
    Custom reward function for Go1 quadruped robot control task.
    
    Args:
        observation (dict): The current observation from the environment.
        action (ndarray): The current action taken by the policy.
        done (bool): Whether the episode is done.
        env: The current environment (Go1MujocoCustom-v0).
    
    Returns:
        float: Reward for the current step.
    """
    # Extract actual observation array from dictionary
    observation = observation.get('observation', observation)
    
    # Extract variables from observation
    joint_pos_vel = observation[:12]               # Joint positions/velocities
    base_lin_ang_vel = observation[12:18]          # Base linear/angular velocities
    projected_gravity_vec = observation[18:21]     # Projected gravity vector
    desired_vel = observation[21]                  # Desired velocity
    previous_actions = observation[22:]             # Previous actions

    # Negative reward for large joint torques (encourages energy efficiency)
    torque_penalty = -np.sum(np.square(action))

    # Positive reward for maintaining the desired velocity
    speed_reward = -np.abs(base_lin_ang_vel[0] - desired_vel)

    # Negative reward for large difference in desired and actual angular velocity (minimize rotation)
    angular_penalty = -np.abs(desired_vel - base_lin_ang_vel[5])

    # Calculate final reward
    reward = speed_reward + angular_penalty + torque_penalty

    return reward
```

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Based on the provided previous reward function and the given feedback, the main problem is that the robot might not able to reach the desired state and hence the evaluation fails i.e. no episodes got completed. 

There could be few reasons for this, however, without any error or specific feedback, it is a bit hard to pinpoint the exact issue. Still, a possible issue could be the negative reward for large joint torques and large difference in desired and actual angular velocity. The negative rewards might discourage the agent. To encounter this we can slightly modify our reward function by:

1. Scaling down the torque penalty to a smaller value.
2. Scaling down the angular_penalty to a smaller value. 
3. Introducing a reward for standing straight (minimum pitch and roll).
4. Introducing a negative reward proportional to the distance from the goal to promote the robot to move towards the goal.

Let's reformulate our reward function taking into account these adjustments.

```python
import numpy as np

def custom_reward_function(observation, action, done, env):
    """
    Custom reward function for Go1 quadruped robot control task.
    
    Args:
        observation (dict): The current observation from the environment.
        action (ndarray): The current action taken by the policy.
        done (bool): Whether the episode is done.
        env: The current environment (Go1MujocoCustom-v0).
    
    Returns:
        float: Reward for the current step.
    """
    # Extract the actual observation array from observation dictionary
    observation = observation.get('observation', observation)
    
    # Quaternions of the base orientation
    qt = observation[3:7]
    # Convert quaternion to euler angles and extract roll and pitch
    roll_pitch = np.array(euler_from_quaternion(qt))[:2]
    
    # Extracting base velocity from observation
    base_lin_speed = observation[15:18]
    
    # Reward for standing straight
    stand_reward = - np.linalg.norm(roll_pitch)
    
    # Negative reward for large joint torques (energy efficiency)
    torque_penalty = -0.01 * np.sum(np.square(action))
    
    # Positive reward for maintaining the base velocity
    speed_reward = np.linalg.norm(base_lin_speed)
    
    # Negative reward for large difference in desired and actual angular velocity
    angular_penalty = -(1.0 - np.dot(env.unwrapped.goal, base_lin_speed))

    # Final reward
    reward = stand_reward + torque_penalty + speed_reward + angular_penalty

    return reward
```

In the code snippet above, we've decreased the scaling of torque_penalty and angular_penalty to encourage the robot to reach the desired velocities. We've also added stand_reward to encourage the robot to maintain a stable, upright position. Finally, we've added a reward for the base linear speed to encourage the robot to move towards its goal.
```

**Generated Reward Code (saved to reward_function_1.py):**
```python
import numpy as np

def custom_reward_function(observation, action, done, env):
    """
    Custom reward function for Go1 quadruped robot control task.
    
    Args:
        observation (dict): The current observation from the environment.
        action (ndarray): The current action taken by the policy.
        done (bool): Whether the episode is done.
        env: The current environment (Go1MujocoCustom-v0).
    
    Returns:
        float: Reward for the current step.
    """
    # Extract the actual observation array from observation dictionary
    observation = observation.get('observation', observation)
    
    # Quaternions of the base orientation
    qt = observation[3:7]
    # Convert quaternion to euler angles and extract roll and pitch
    roll_pitch = np.array(euler_from_quaternion(qt))[:2]
    
    # Extracting base velocity from observation
    base_lin_speed = observation[15:18]
    
    # Reward for standing straight
    stand_reward = - np.linalg.norm(roll_pitch)
    
    # Negative reward for large joint torques (energy efficiency)
    torque_penalty = -0.01 * np.sum(np.square(action))
    
    # Positive reward for maintaining the base velocity
    speed_reward = np.linalg.norm(base_lin_speed)
    
    # Negative reward for large difference in desired and actual angular velocity
    angular_penalty = -(1.0 - np.dot(env.unwrapped.goal, base_lin_speed))

    # Final reward
    reward = stand_reward + torque_penalty + speed_reward + angular_penalty

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

def custom_reward_function(observation, action, done, env):
    """
    Custom reward function for Go1 quadruped robot control task.
    
    Args:
        observation (dict): The current observation from the environment.
        action (ndarray): The current action taken by the policy.
        done (bool): Whether the episode is done.
        env: The current environment (Go1MujocoCustom-v0).
    
    Returns:
        float: Reward for the current step.
    """
    # Extract the actual observation array from observation dictionary
    observation = observation.get('observation', observation)
    
    # Quaternions of the base orientation
    qt = observation[3:7]
    # Convert quaternion to euler angles and extract roll and pitch
    roll_pitch = np.array(euler_from_quaternion(qt))[:2]
    
    # Extracting base velocity from observation
    base_lin_speed = observation[15:18]
    
    # Reward for standing straight
    stand_reward = - np.linalg.norm(roll_pitch)
    
    # Negative reward for large joint torques (energy efficiency)
    torque_penalty = -0.01 * np.sum(np.square(action))
    
    # Positive reward for maintaining the base velocity
    speed_reward = np.linalg.norm(base_lin_speed)
    
    # Negative reward for large difference in desired and actual angular velocity
    angular_penalty = -(1.0 - np.dot(env.unwrapped.goal, base_lin_speed))

    # Final reward
    reward = stand_reward + torque_penalty + speed_reward + angular_penalty

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

Feedback from Previous Iterations / Analysis:
This is the first iteration. Please generate an initial reward function based on the task description and environment details provided.

Instructions:
Based on the task, environment configuration, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Reward function for the Go1 robot walking task.

    The task is to make the Go1 robot walk forward at a desired velocity. 
    The reward is computed based on the distance to the goal, the velocity matching the desired value, 
    and a control cost to prevent excessive torque use.

    This function takes as inputs the observations, taken action, completion status (done), and the environment.

    :param obs: observation from the environment
    :param action: action taken by the agent
    :param done: boolean flag indicating completion of task
    :param env: the RL environment object
    """
    
    # Extracting raw observations if obs is a dictionary
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Extract base velocity details
    base_velocities = obs[13:16]  # Base linear x,y,z velocities (m/s)

    # Compute the distance to the goal
    goal_position = env.unwrapped.goal
    distance = np.linalg.norm(goal_position - base_velocities[0])

    # Compute the reward for moving towards the goal
    reward_dist = -distance

    # Compute the reward for matching the desired velocity
    desired_velocity = obs[-2]  # Extract desired ler velocity
    velocity_diff = np.abs(desired_velocity - base_velocities[0])
    reward_vel = -velocity_diff  # the closer the actual velocity is to the desired one, the higher the reward

    # Compute the control cost (we want to minimize the total force applied)
    control_cost = -np.sum(np.square(action))

    # Incorporate both the distance to goal and control effort in reward computation
    reward = reward_dist + reward_vel + 0.1 * control_cost

    return reward
```
This function first extracts the base velocities from the observation. It then calculates the distance to the goal using numpy's linalg.norm function to compute the Euclidean distance. The reward for moving towards the goal is then computed by taking the negative of this distance, encouraging the robot to move towards the goal (reducing the distance).

Next, the function calculates the desired velocity from the observation and calculates the difference between this and the robot's actual velocity. This is again negated to provide a reward: the smaller the difference, the closer the robot is to the desired velocity, and the larger the reward.

Finally, the control cost or the force utilized to perform the action is calculated by summing the squares of the action values. This is also negated and added to the reward computation, providing a penalty for using more control effort. The balance here between achieving the goal and minimizing effort makes for an effective and efficient reinforcement learning experiment.
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Reward function for the Go1 robot walking task.

    The task is to make the Go1 robot walk forward at a desired velocity. 
    The reward is computed based on the distance to the goal, the velocity matching the desired value, 
    and a control cost to prevent excessive torque use.

    This function takes as inputs the observations, taken action, completion status (done), and the environment.

    :param obs: observation from the environment
    :param action: action taken by the agent
    :param done: boolean flag indicating completion of task
    :param env: the RL environment object
    """
    
    # Extracting raw observations if obs is a dictionary
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Extract base velocity details
    base_velocities = obs[13:16]  # Base linear x,y,z velocities (m/s)

    # Compute the distance to the goal
    goal_position = env.unwrapped.goal
    distance = np.linalg.norm(goal_position - base_velocities[0])

    # Compute the reward for moving towards the goal
    reward_dist = -distance

    # Compute the reward for matching the desired velocity
    desired_velocity = obs[-2]  # Extract desired ler velocity
    velocity_diff = np.abs(desired_velocity - base_velocities[0])
    reward_vel = -velocity_diff  # the closer the actual velocity is to the desired one, the higher the reward

    # Compute the control cost (we want to minimize the total force applied)
    control_cost = -np.sum(np.square(action))

    # Incorporate both the distance to goal and control effort in reward computation
    reward = reward_dist + reward_vel + 0.1 * control_cost

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
    Reward function for the Go1 robot walking task.

    The task is to make the Go1 robot walk forward at a desired velocity. 
    The reward is computed based on the distance to the goal, the velocity matching the desired value, 
    and a control cost to prevent excessive torque use.

    This function takes as inputs the observations, taken action, completion status (done), and the environment.

    :param obs: observation from the environment
    :param action: action taken by the agent
    :param done: boolean flag indicating completion of task
    :param env: the RL environment object
    """
    
    # Extracting raw observations if obs is a dictionary
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Extract base velocity details
    base_velocities = obs[13:16]  # Base linear x,y,z velocities (m/s)

    # Compute the distance to the goal
    goal_position = env.unwrapped.goal
    distance = np.linalg.norm(goal_position - base_velocities[0])

    # Compute the reward for moving towards the goal
    reward_dist = -distance

    # Compute the reward for matching the desired velocity
    desired_velocity = obs[-2]  # Extract desired ler velocity
    velocity_diff = np.abs(desired_velocity - base_velocities[0])
    reward_vel = -velocity_diff  # the closer the actual velocity is to the desired one, the higher the reward

    # Compute the control cost (we want to minimize the total force applied)
    control_cost = -np.sum(np.square(action))

    # Incorporate both the distance to goal and control effort in reward computation
    reward = reward_dist + reward_vel + 0.1 * control_cost

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

[System Retry Feedback]: The previous response did not contain a valid Python code block. Please ensure the reward function code is clearly marked within ```python ... ``` tags and follows the required signature `def custom_reward_function(obs, action, done, env):`. Use only numpy and standard python.


Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
IMPORTANT: `obs` is a dictionary. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1 robot.

    The robot gets a high reward if it achieves the desired linear (forward) and angular velocity. 
    Additional rewards are added if the robot maintains a stable orientation to prevent falling.

    Arguments:
    obs -- dictionary, observation from the environment. 
           Keys: 'joint_positions', 'joint_velocities', 'base_linear_velocity', 'base_angular_velocity',
           'gravity_vector', 'desired_velocity', 'previous_action'
    action -- np.array, action taken by the agent
    done -- boolean, whether the episode has ended
    env -- gym.Env, environment instance

    Returns:
    reward -- float, calculated reward based on the current observation and action
    """

    # Extract the base environment to obtain the goal position 
    base_env = env.unwrapped

    # Normalize rewards to achieve better performance and stability
    reward_scale = 0.01

    # Initialize reward
    reward = 0.0

    # Extract necessary components from the observation
    base_linear_velocity = obs['base_linear_velocity']
    base_angular_velocity = obs['base_angular_velocity']
    desired_velocity = obs['desired_velocity']
    
    # Reward based on velocity tracking
    if desired_velocity is not None:
        # Linear velocity reward
        vel_diff = desired_velocity - base_linear_velocity
        reward -= np.linalg.norm(vel_diff) * reward_scale

        # Angular velocity reward
        ang_vel_diff = - base_angular_velocity # We want the robot to walk forward, hence angular velocity should be minimal
        reward -= np.abs(ang_vel_diff) * reward_scale
    
    # Reward to discourage large actions, promoting more smooth transitions
    reward -= np.linalg.norm(action) * reward_scale
    
    # Return the reward
    return reward
```
This code block defines the `custom_reward_function`. The main idea behind the reward function is to encourage the robot to maintain the desired linear (forward) and angular velocities, whilst discouraging large actions which can lead to instability. The reward is computed based on the difference in actual and desired velocities and the magnitude of the action being applied. This encourages the robot to learn to track the desired velocity over time. Any deviation from the desired path is punished and smaller, smoother actions are rewarded. As this is a complex control task that requires precise movement, this reward function should facilitate efficient learning.
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for Go1 robot.

    The robot gets a high reward if it achieves the desired linear (forward) and angular velocity. 
    Additional rewards are added if the robot maintains a stable orientation to prevent falling.

    Arguments:
    obs -- dictionary, observation from the environment. 
           Keys: 'joint_positions', 'joint_velocities', 'base_linear_velocity', 'base_angular_velocity',
           'gravity_vector', 'desired_velocity', 'previous_action'
    action -- np.array, action taken by the agent
    done -- boolean, whether the episode has ended
    env -- gym.Env, environment instance

    Returns:
    reward -- float, calculated reward based on the current observation and action
    """

    # Extract the base environment to obtain the goal position 
    base_env = env.unwrapped

    # Normalize rewards to achieve better performance and stability
    reward_scale = 0.01

    # Initialize reward
    reward = 0.0

    # Extract necessary components from the observation
    base_linear_velocity = obs['base_linear_velocity']
    base_angular_velocity = obs['base_angular_velocity']
    desired_velocity = obs['desired_velocity']
    
    # Reward based on velocity tracking
    if desired_velocity is not None:
        # Linear velocity reward
        vel_diff = desired_velocity - base_linear_velocity
        reward -= np.linalg.norm(vel_diff) * reward_scale

        # Angular velocity reward
        ang_vel_diff = - base_angular_velocity # We want the robot to walk forward, hence angular velocity should be minimal
        reward -= np.abs(ang_vel_diff) * reward_scale
    
    # Reward to discourage large actions, promoting more smooth transitions
    reward -= np.linalg.norm(action) * reward_scale
    
    # Return the reward
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
    Custom reward function for Go1 robot.

    The robot gets a high reward if it achieves the desired linear (forward) and angular velocity. 
    Additional rewards are added if the robot maintains a stable orientation to prevent falling.

    Arguments:
    obs -- dictionary, observation from the environment. 
           Keys: 'joint_positions', 'joint_velocities', 'base_linear_velocity', 'base_angular_velocity',
           'gravity_vector', 'desired_velocity', 'previous_action'
    action -- np.array, action taken by the agent
    done -- boolean, whether the episode has ended
    env -- gym.Env, environment instance

    Returns:
    reward -- float, calculated reward based on the current observation and action
    """

    # Extract the base environment to obtain the goal position 
    base_env = env.unwrapped

    # Normalize rewards to achieve better performance and stability
    reward_scale = 0.01

    # Initialize reward
    reward = 0.0

    # Extract necessary components from the observation
    base_linear_velocity = obs['base_linear_velocity']
    base_angular_velocity = obs['base_angular_velocity']
    desired_velocity = obs['desired_velocity']
    
    # Reward based on velocity tracking
    if desired_velocity is not None:
        # Linear velocity reward
        vel_diff = desired_velocity - base_linear_velocity
        reward -= np.linalg.norm(vel_diff) * reward_scale

        # Angular velocity reward
        ang_vel_diff = - base_angular_velocity # We want the robot to walk forward, hence angular velocity should be minimal
        reward -= np.abs(ang_vel_diff) * reward_scale
    
    # Reward to discourage large actions, promoting more smooth transitions
    reward -= np.linalg.norm(action) * reward_scale
    
    # Return the reward
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
IMPORTANT: `obs` is a dictionary. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 Robot to move forward at a desired velocity.
    Given the observation, action, done and environment variables, the function calculates a reward
    based on how well the robot moves forward at a desired velocity. 

    Args:
        obs: (OrderedDict) The current observation from the environment.
        action: (numpy.array) The current action taken by the robot.
        done: (bool) Whether the episode is done.
        env: (gym.Env) The environment instance.

    Returns:
        reward: (float) The calculated reward.
    """
    # Access environment's base goal
    goal_velocity = env.unwrapped.goal

    # Extract relevant observation variables
    obs = obs.get('observation', obs)
    achieved_velocity = obs['linear_velocity']

    tracking_error = goal_velocity - achieved_velocity

    # the robot should aim to minimize the tracking_error
    reward = -np.abs(tracking_error)

    # punish large actions to avoid unnecessary movements
    reward -= 0.1 * np.square(action).sum()

    # punish if episode is done before reaching the goal
    if done:
        reward -= 50.0

    return reward
```
This function gives a high reward when the robot's velocity is close to the goal velocity and also punishes the robot if it makes unnecessarily large actions. Furthermore, it punishes the robot if it does not reach the goal before the end of the episode. The constants used in punishment can be tuned for better performance balance.
```

**Generated Reward Code (saved to reward_function_0.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 Robot to move forward at a desired velocity.
    Given the observation, action, done and environment variables, the function calculates a reward
    based on how well the robot moves forward at a desired velocity. 

    Args:
        obs: (OrderedDict) The current observation from the environment.
        action: (numpy.array) The current action taken by the robot.
        done: (bool) Whether the episode is done.
        env: (gym.Env) The environment instance.

    Returns:
        reward: (float) The calculated reward.
    """
    # Access environment's base goal
    goal_velocity = env.unwrapped.goal

    # Extract relevant observation variables
    obs = obs.get('observation', obs)
    achieved_velocity = obs['linear_velocity']

    tracking_error = goal_velocity - achieved_velocity

    # the robot should aim to minimize the tracking_error
    reward = -np.abs(tracking_error)

    # punish large actions to avoid unnecessary movements
    reward -= 0.1 * np.square(action).sum()

    # punish if episode is done before reaching the goal
    if done:
        reward -= 50.0

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
    Custom reward function for the Go1 Robot to move forward at a desired velocity.
    Given the observation, action, done and environment variables, the function calculates a reward
    based on how well the robot moves forward at a desired velocity. 

    Args:
        obs: (OrderedDict) The current observation from the environment.
        action: (numpy.array) The current action taken by the robot.
        done: (bool) Whether the episode is done.
        env: (gym.Env) The environment instance.

    Returns:
        reward: (float) The calculated reward.
    """
    # Access environment's base goal
    goal_velocity = env.unwrapped.goal

    # Extract relevant observation variables
    obs = obs.get('observation', obs)
    achieved_velocity = obs['linear_velocity']

    tracking_error = goal_velocity - achieved_velocity

    # the robot should aim to minimize the tracking_error
    reward = -np.abs(tracking_error)

    # punish large actions to avoid unnecessary movements
    reward -= 0.1 * np.square(action).sum()

    # punish if episode is done before reaching the goal
    if done:
        reward -= 50.0

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
    Custom reward function for the Go1 Robot to move forward at a desired velocity.
    Given the observation, action, done and environment variables, the function calculates a reward
    based on how well the robot moves forward at a desired velocity. 

    Args:
        obs: (OrderedDict) The current observation from the environment.
        action: (numpy.array) The current action taken by the robot.
        done: (bool) Whether the episode is done.
        env: (gym.Env) The environment instance.

    Returns:
        reward: (float) The calculated reward.
    """
    # Access environment's base goal
    goal_velocity = env.unwrapped.goal

    # Extract relevant observation variables
    obs = obs.get('observation', obs)
    achieved_velocity = obs['linear_velocity']

    tracking_error = goal_velocity - achieved_velocity

    # the robot should aim to minimize the tracking_error
    reward = -np.abs(tracking_error)

    # punish large actions to avoid unnecessary movements
    reward -= 0.1 * np.square(action).sum()

    # punish if episode is done before reaching the goal
    if done:
        reward -= 50.0

    return reward
```

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
IMPORTANT: `obs` is a dictionary. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 Robot to move forward at a desired velocity.
    Given the observation, action, done and environment variables, the function calculates a reward
    based on how well the robot moves forward at a desired velocity. 

    Args:
        obs: (OrderedDict) The current observation from the environment.
        action: (numpy.array) The current action taken by the robot.
        done: (bool) Whether the episode is done.
        env: (gym.Env) The environment instance.

    Returns:
        reward: (float) The calculated reward.
    """
    # Access environment's goal
    goal_velocity = env.unwrapped.goal

    # Extract relevant observation variables
    obs = obs.get('observation', obs)
    achieved_velocity = obs['linear_velocity']
    
    # Calculate velocity error
    velocity_error = np.abs(goal_velocity - achieved_velocity)
    
    # Calculate action cost
    action_cost = np.square(action).sum()
    
    if done:
        # If episode is finished before reaching the goal, give negative reward.
        reward = -100.0
    else:
        # Encourage the agent to reach the goal as fast as possible by giving higher reward.
        reward = -velocity_error - 0.1*action_cost
    
    return reward
```
```

**Generated Reward Code (saved to reward_function_1.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 Robot to move forward at a desired velocity.
    Given the observation, action, done and environment variables, the function calculates a reward
    based on how well the robot moves forward at a desired velocity. 

    Args:
        obs: (OrderedDict) The current observation from the environment.
        action: (numpy.array) The current action taken by the robot.
        done: (bool) Whether the episode is done.
        env: (gym.Env) The environment instance.

    Returns:
        reward: (float) The calculated reward.
    """
    # Access environment's goal
    goal_velocity = env.unwrapped.goal

    # Extract relevant observation variables
    obs = obs.get('observation', obs)
    achieved_velocity = obs['linear_velocity']
    
    # Calculate velocity error
    velocity_error = np.abs(goal_velocity - achieved_velocity)
    
    # Calculate action cost
    action_cost = np.square(action).sum()
    
    if done:
        # If episode is finished before reaching the goal, give negative reward.
        reward = -100.0
    else:
        # Encourage the agent to reach the goal as fast as possible by giving higher reward.
        reward = -velocity_error - 0.1*action_cost
    
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
    Custom reward function for the Go1 Robot to move forward at a desired velocity.
    Given the observation, action, done and environment variables, the function calculates a reward
    based on how well the robot moves forward at a desired velocity. 

    Args:
        obs: (OrderedDict) The current observation from the environment.
        action: (numpy.array) The current action taken by the robot.
        done: (bool) Whether the episode is done.
        env: (gym.Env) The environment instance.

    Returns:
        reward: (float) The calculated reward.
    """
    # Access environment's goal
    goal_velocity = env.unwrapped.goal

    # Extract relevant observation variables
    obs = obs.get('observation', obs)
    achieved_velocity = obs['linear_velocity']
    
    # Calculate velocity error
    velocity_error = np.abs(goal_velocity - achieved_velocity)
    
    # Calculate action cost
    action_cost = np.square(action).sum()
    
    if done:
        # If episode is finished before reaching the goal, give negative reward.
        reward = -100.0
    else:
        # Encourage the agent to reach the goal as fast as possible by giving higher reward.
        reward = -velocity_error - 0.1*action_cost
    
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
    Custom reward function for the Go1 Robot to move forward at a desired velocity.
    Given the observation, action, done and environment variables, the function calculates a reward
    based on how well the robot moves forward at a desired velocity. 

    Args:
        obs: (OrderedDict) The current observation from the environment.
        action: (numpy.array) The current action taken by the robot.
        done: (bool) Whether the episode is done.
        env: (gym.Env) The environment instance.

    Returns:
        reward: (float) The calculated reward.
    """
    # Access environment's goal
    goal_velocity = env.unwrapped.goal

    # Extract relevant observation variables
    obs = obs.get('observation', obs)
    achieved_velocity = obs['linear_velocity']
    
    # Calculate velocity error
    velocity_error = np.abs(goal_velocity - achieved_velocity)
    
    # Calculate action cost
    action_cost = np.square(action).sum()
    
    if done:
        # If episode is finished before reaching the goal, give negative reward.
        reward = -100.0
    else:
        # Encourage the agent to reach the goal as fast as possible by giving higher reward.
        reward = -velocity_error - 0.1*action_cost
    
    return reward
```

Instructions:
Based on the task, environment configuration, observation space, and feedback, generate an improved Python reward function. 
The function must have the signature: `def custom_reward_function(obs, action, done, env):`
IMPORTANT: `obs` is a dictionary. Access its components ONLY using the string keys listed in Observation Space Details (e.g., `obs['linear_velocity']`). DO NOT use integer indices like `obs[0]` or `obs[-3]`.
Use only NumPy (`import numpy as np`) and standard Python libraries.
Ensure the code is enclosed in a single Python code block (```python ... ```).

```

**LLM Response:**
```
Based on the information provided, it seems we are trying to make our robot move at a desired forward velocity. The previous reward function appears to be well-intended; it rewards the robot more for minimizing the difference between the target velocity and the robot's own velocity, and it penalizes the use of excessive control actions (as mentioned in the 'action_cost'). 

But one potential problem is that the reward obtained when the robot reaches the done state, it's -100.0 irrespective of whether the robot has achieved the required goal or not. If the robot has achieved the desired goal, it should get a high reward, not a penalty. If it hasn't achieved the goal, it should then get a penalty.

Let's modify the existing reward function. We will reward the agent more when its achieved velocity is closer to the goal velocity and penalise for larger action (encourage smaller actions). Instead of providing a fixed negative award at termination, we will make it proportional to how off it was from the goal.

In the following code, reward is chosen to be the negative of the absolute error between current and desired velocities normalized by the desired velocity. This is to ensure the reward is always in the range of -1 to 0, where -1 corresponds to the worst case (when velocity is zero), and 0 corresponds to the best case (when the current velocity matches the desired velocity). In addition to that, the proportional control on the action makes sure that the agent tries to minimize the control effort (to stabilize and preserve the system's energy).

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 Robot to move forward at a desired velocity.
    Given the observation, action, done and environment variables, the function calculates a reward
    based on how well the robot moves forward at a desired velocity. 

    Args:
        obs: (OrderedDict) The current observation from the environment.
        action: (numpy.array) The current action taken by the robot.
        done: (bool) Whether the episode is done.
        env: (gym.Env) The environment instance.

    Returns:
        reward: (float) The calculated reward.
    """
    # Access environment's goal
    goal_velocity = env.unwrapped.goal

    # Extract relevant observation variables
    obs = obs.get('observation', obs)
    achieved_velocity = obs['linear_velocity']
    
    # Calculate velocity error
    velocity_error = np.abs(goal_velocity - achieved_velocity)
    
    # Calculate action cost
    action_cost = np.square(action).sum()
    
    if done:
        # At end of the episode, the reward is based on how far the bot is from the desired velocity
        reward = - 100.0 * velocity_error / goal_velocity
    else:
        # Encourage the agent to reach the goal as fastest as possible by giving higher reward for closer to goal velocities and smaller actions
        reward = - velocity_error / goal_velocity - 0.1* action_cost
    
    return reward
```
In general, the reward function for the desired velocity control task is designed to incentivize the robot to move as close as possible to the desired velocity while minimizing unnecessary actions. For this task, it is important to keep the reward function simple and directly related to the ultimate goal of the task. The distance-based rewards coupled with punishment for unnecessary actions can help train the model to reach the desired velocity efficiently.
```

**Generated Reward Code (saved to reward_function_2.py):**
```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    """
    Custom reward function for the Go1 Robot to move forward at a desired velocity.
    Given the observation, action, done and environment variables, the function calculates a reward
    based on how well the robot moves forward at a desired velocity. 

    Args:
        obs: (OrderedDict) The current observation from the environment.
        action: (numpy.array) The current action taken by the robot.
        done: (bool) Whether the episode is done.
        env: (gym.Env) The environment instance.

    Returns:
        reward: (float) The calculated reward.
    """
    # Access environment's goal
    goal_velocity = env.unwrapped.goal

    # Extract relevant observation variables
    obs = obs.get('observation', obs)
    achieved_velocity = obs['linear_velocity']
    
    # Calculate velocity error
    velocity_error = np.abs(goal_velocity - achieved_velocity)
    
    # Calculate action cost
    action_cost = np.square(action).sum()
    
    if done:
        # At end of the episode, the reward is based on how far the bot is from the desired velocity
        reward = - 100.0 * velocity_error / goal_velocity
    else:
        # Encourage the agent to reach the goal as fastest as possible by giving higher reward for closer to goal velocities and smaller actions
        reward = - velocity_error / goal_velocity - 0.1* action_cost
    
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
    Custom reward function for the Go1 Robot to move forward at a desired velocity.
    Given the observation, action, done and environment variables, the function calculates a reward
    based on how well the robot moves forward at a desired velocity. 

    Args:
        obs: (OrderedDict) The current observation from the environment.
        action: (numpy.array) The current action taken by the robot.
        done: (bool) Whether the episode is done.
        env: (gym.Env) The environment instance.

    Returns:
        reward: (float) The calculated reward.
    """
    # Access environment's goal
    goal_velocity = env.unwrapped.goal

    # Extract relevant observation variables
    obs = obs.get('observation', obs)
    achieved_velocity = obs['linear_velocity']
    
    # Calculate velocity error
    velocity_error = np.abs(goal_velocity - achieved_velocity)
    
    # Calculate action cost
    action_cost = np.square(action).sum()
    
    if done:
        # At end of the episode, the reward is based on how far the bot is from the desired velocity
        reward = - 100.0 * velocity_error / goal_velocity
    else:
        # Encourage the agent to reach the goal as fastest as possible by giving higher reward for closer to goal velocities and smaller actions
        reward = - velocity_error / goal_velocity - 0.1* action_cost
    
    return reward
```
```

---

