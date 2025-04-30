import json
import matplotlib.pyplot as plt
import os
import glob

def plot_data(file_path):
    """Plot data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract steps and rewards
    steps = [point[1] for point in data]  # Second element is step count
    rewards = [point[2] for point in data]  # Third element is reward
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards)
    plt.title(f'Learning Curve: {os.path.basename(file_path)}')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Save the plot
    output_file = file_path.replace('.json', '.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Created plot: {output_file}")

def main():
    # Find all JSON files in the plots directory
    json_files = glob.glob('plots/*.json')
    
    # Plot each file
    for json_file in json_files:
        try:
            plot_data(json_file)
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

if __name__ == "__main__":
    main() 