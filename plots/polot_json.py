import json
import matplotlib.pyplot as plt

# Load JSON data
with open('/home/bechir/Downloads/json.json', 'r') as file:
    data = json.load(file)

# Assuming the JSON structure is {"x": [1, 2, 3], "y": [4, 5, 6]}

timesteps = [item[1] for item in data]
reward_per_episode = [ item[2] for item in data]

# Set the style to a clean, professional look
plt.style.use('seaborn-whitegrid')

# Create figure with specific size and DPI for publication quality
plt.figure(figsize=(8, 6), dpi=300)

# Plot with specific styling
plt.plot(timesteps, reward_per_episode, color='#2E64FE', linewidth=2, marker='o', markersize=4)

# Customize axes
plt.xlabel('Timesteps', fontsize=12, fontweight='bold')
plt.ylabel('Reward per Episode', fontsize=12, fontweight='bold')
plt.title('Training Progress', fontsize=14, fontweight='bold', pad=15)

# Customize grid
plt.grid(True, linestyle='--', alpha=0.7)

# Customize ticks
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add borders to the plot
plt.box(True)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure in high resolution
plt.savefig('training_progress.pdf', format='pdf', bbox_inches='tight')
plt.savefig('training_progress.png', format='png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()






