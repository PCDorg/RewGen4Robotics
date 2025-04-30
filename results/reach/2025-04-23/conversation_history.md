## Iteration 1/5: Failed - OpenAI API Error
**Error:** `Error calling OpenAI API: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}`
**Attempted Prompt:**
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
---

## Iteration 2/5: Failed - OpenAI API Error
**Error:** `Error calling OpenAI API: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}`
**Attempted Prompt:**
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

**Status:** Failed to generate usable code in the previous attempt after 1 tries.
**Last LLM Response:**
Error calling OpenAI API: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}

Please try again, carefully following the instructions and ensuring the code is correctly formatted with the signature `def custom_reward_function(obs, action, done, env):`.


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
---

## Iteration 3/5: Failed - OpenAI API Error
**Error:** `Error calling OpenAI API: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}`
**Attempted Prompt:**
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

**Status:** Failed to generate usable code in the previous attempt after 1 tries.
**Last LLM Response:**
Error calling OpenAI API: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}

Please try again, carefully following the instructions and ensuring the code is correctly formatted with the signature `def custom_reward_function(obs, action, done, env):`.


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
---

