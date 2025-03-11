Below is an example of a reward function for the AntMaze task. The ant gets a reward for moving toward the target and receives penalties for inefficient joint movements.

```python
import numpy as np

def custom_reward_function(obs, action, done, env):
    # If the observation is a dictionary, retrieve the observation array
    obs = obs.get('observation', obs) if isinstance(obs, dict) else obs

    # Extract the ant's current position, velocity and target position
    x, y = obs[0:2]
    vx, vy = obs[2:4]
    goal = env.unwrapped.goal

    # Compute the distance to the target
    distance = np.linalg.norm(goal - np.array([x, y]))

    # Reward is a negative function of the distance to the goal
    reward_dist = -distance

    # Bonus reward for being close to the goal
    goal_threshold = 0.05
    if distance < goal_threshold:
        reward_goal_progess = 200
    else:
        reward_goal_progess = 0

    # Action penalty proportional to the square sum of action, to encourage smoother actions
    reward_action_penalty = -np.sum(np.square(action)) * 0.05

    # Negative reward for not reaching the goal
    if done and distance > goal_threshold:
        reward_completion_penalty = -200
    else:
        reward_completion_penalty = 0

    # Sum up rewards and penalties as the total reward
    reward = reward_dist + reward_action_penalty + reward_goal_progess + reward_completion_penalty

    return reward
```

This reward function mainly focuses on movement towards the target and avoids excessive actions. If the ant gets stuck or terminates the episode before reaching the target, it receives a big negative reward. On the other hand, arriving close to the goal also earns a substantial positive reward. This should encourage the ant to efficiently navigate to the target.