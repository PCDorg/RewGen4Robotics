import json
import matplotlib.pyplot as plt

# Load the data from the JSON files
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

file_paths = [
    "iteration_1_SAC_0.json" ,
    "iteration_4_SAC_0.json",
    "iteration_3_SAC_0.json"
]

data = [load_data(file_path) for file_path in file_paths]

# Extract the step and reward values from the data
def extract_values(data):
    steps = [entry[1] for entry in data]
    rewards = [entry[2] for entry in data]
    return steps, rewards

steps = []
rewards = []

for d in data:
  step, reward = extract_values(d)
  steps.append(step)
  rewards.append(reward)

# Plot the data
plt.figure(figsize=(10, 6))

for i in range(len(file_paths)):
  plt.plot(steps[i], rewards[i], label=f'File: {file_paths[i]}')

plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Reward vs Step for different JSON files')
plt.legend()
plt.grid(True)
plt.show()
