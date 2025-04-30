import json
import matplotlib.pyplot as plt
import os
def main() :
# Define the directory containing the JSON files
  plot_dir = "/home/ken2/PCD/plots"

  # --- Configuration ---
  iterations_to_plot = {
      1: "iteration_1_SAC_0.json",
      3: "iteration_3_SAC_0.json",
      4: "iteration_4_SAC_0.json",
  }
  output_filename = "fetchreach_combined_training_plot.png"
  plot_title = "FetchReach: Training Performance (rollout/ep_rew_mean)"
  x_label = "Timesteps"
  y_label = "Mean Episode Reward"
  # --- End Configuration ---

  plt.figure(figsize=(12, 7)) # Adjust figure size if needed

  print("Starting plot generation...")

  all_data_loaded = True
  for iteration, filename in iterations_to_plot.items():
      file_path = os.path.join(plot_dir, filename)
      print(f"Processing Iteration {iteration} from {filename}...")
      try:
          with open(file_path, 'r') as f:
              # Load the JSON data which is a single list of lists
              data = json.load(f)
              if not data:
                  print(f"  Warning: No data found in {filename}. Skipping.")
                  continue
              if not isinstance(data, list) or not all(isinstance(item, list) and len(item) == 3 for item in data):
                  print(f"  Warning: Data in {filename} is not in the expected format (list of [timestamp, step, value]). Skipping.")
                  continue

              # Extract steps (index 1) and reward values (index 2)
              steps = [item[1] for item in data]
              rewards = [item[2] for item in data]

              if not steps or not rewards:
                  print(f"  Warning: Could not extract steps or rewards from {filename}. Skipping.")
                  continue

              # Plot the data for this iteration
              plt.plot(steps, rewards, label=f"Iteration {iteration}")
              print(f"  Successfully plotted {len(steps)} data points.")

      except FileNotFoundError:
            print(f"  Error: File not found: {file_path}. Skipping iteration {iteration}.")
            all_data_loaded = False
      except json.JSONDecodeError:
            print(f"  Error: Could not decode JSON from {file_path}. Skipping iteration {iteration}.")
            all_data_loaded = False
      except Exception as e:
            print(f"  An unexpected error occurred processing {file_path}: {e}. Skipping iteration {iteration}.")
            all_data_loaded = False

    # Add plot labels, title, legend, and grid
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(plot_title)
  plt.legend()
  plt.grid(True)

    # Save the plot
  output_path = os.path.join(plot_dir, output_filename)
  try:
        plt.savefig(output_path)
        print(f"Plot saved successfully to: {output_path}")
        if not all_data_loaded:
            print("Warning: Some iteration files could not be loaded or processed. The plot may be incomplete.")
  except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")

    # Optionally display the plot
    # plt.show()

  print("Plot generation finished.")

if __name__ == "__main__":
    main()
