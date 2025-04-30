import os
import openai
import logging
import math
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import date, datetime
import importlib
from groq import Groq
import re

from utils.file_to_string import file_to_string
from utils.extracct_code import extract_code_from_response
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import traceback
import time

# Dictionary to map env names (from config) to their factory functions and modules
ENV_FACTORIES = {
    'fetchReach': ('envs.fetchReach', 'make_custom_fetch'),
    'antmaze': ('envs.antmaze', 'make_custom_antmaze'),
    'go1': ('envs.Go2Env', 'make_custom_go1'),
}

def get_env_factory(env_name):
    """Dynamically imports and returns the environment factory function."""
    if env_name not in ENV_FACTORIES:
        raise ValueError(f"Unknown environment name: {env_name}. Available: {list(ENV_FACTORIES.keys())}")

    module_name, func_name = ENV_FACTORIES[env_name]
    try:
        module = importlib.import_module(module_name)
        factory_func = getattr(module, func_name)
        return factory_func
    except ImportError as e:
        logging.error(f"Could not import module {module_name} for environment {env_name}: {e}")
        raise
    except AttributeError as e:
        logging.error(f"Could not find factory function {func_name} in module {module_name}: {e}")
        raise

def get_all_scalars(tb_log_dir):
    """Reads all scalar data from a TensorBoard log directory."""
    try:
        logging.info(f"Attempting to read TensorBoard logs from: {tb_log_dir}")
        # Add a small delay for filesystem sync, especially on networked drives
        time.sleep(2)
        if not os.path.exists(tb_log_dir):
            logging.warning(f"TensorBoard log directory does not exist: {tb_log_dir}")
            return {}
        event_acc = EventAccumulator(tb_log_dir)
        event_acc.Reload() # Load data
        scalar_dict = {}
        tags = event_acc.Tags().get('scalars', [])
        logging.info(f"Found scalar tags: {tags}")
        for tag in tags:
            # Store pairs of (step, value)
            scalar_dict[tag] = [(s.step, s.value) for s in event_acc.Scalars(tag)]
        if not scalar_dict:
            logging.warning(f"No scalar data found in {tb_log_dir}. Check if SAC logged correctly and training ran.")
        return scalar_dict
    except Exception as e:
        logging.error(f"Error reading TensorBoard logs from {tb_log_dir}: {e}\n{traceback.format_exc()}")
        return {}

def get_final_scalar_value(scalar_dict, tag_name):
    """Gets the latest value for a specific scalar tag."""
    if tag_name in scalar_dict and scalar_dict[tag_name]:
        # Sort by step (just in case) and return the last value
        try:
            sorted_scalars = sorted(scalar_dict[tag_name], key=lambda x: x[0])
            return sorted_scalars[-1][1] # Return the value of the last entry
        except IndexError:
            logging.warning(f"Scalar tag '{tag_name}' was found but contained no data.")
            return None
    else:
        # Reduced severity from warning to info as missing tags can be expected if logging changes
        logging.info(f"Scalar tag '{tag_name}' not found or empty in TensorBoard data.")
        return None

# Helper function to sanitize env names for use in paths
def sanitize_path_component(name):
    # Replace common problematic characters with underscores
    # Allow alphanumeric, underscore, hyphen
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    # Avoid starting or ending with underscore/hyphen if possible
    name = name.strip('_-')
    # Handle empty or invalid names after sanitization
    if not name: return "unknown_env"
    return name

def get_llm_response(llm_provider, model, system_prompt_path, user_prompt_path, code_tip_path, cfg, feedback_content, previous_reward_code):
    logging.info(f"Generating new reward function with {llm_provider}...")
    system_prompt = file_to_string(system_prompt_path)
    user_prompt_template = file_to_string(user_prompt_path)
    code_tip = file_to_string(code_tip_path)

    # Get the environment specific config
    env_config = cfg.env
    if not isinstance(env_config, DictConfig):
        logging.error(f"Configuration error: cfg.env is type {type(env_config)}, expected DictConfig. Value: {env_config}")
        raise TypeError(f"cfg.env is not a DictConfig object. Check Hydra configuration loading.")

    # Prepare environment details string from env_config
    env_details_dict = OmegaConf.to_container(env_config, resolve=True)
    env_details_str = "\n".join([f"- {k}: {v}" for k, v in env_details_dict.items()])

    # --- Add Observation Space Details --- #
    observation_details_str = "Observation space structure not available."
    # Try to get env_key to check if it's go1
    env_key = None
    if cfg._metadata and hasattr(cfg._metadata, 'defaults_list'):
        for default_item in cfg._metadata.defaults_list:
            if isinstance(default_item, dict) and 'env' in default_item:
                env_key = default_item['env']
                break
    # If it is go1, provide the specific structure
    if env_key == 'go1':
        # Keys based on Go2Env.py _get_obs method
        go1_obs_keys = [
            "linear_velocity", "angular_velocity", "projected_gravity",
            "desired_velocity", "dofs_position", "dofs_velocity", "last_action"
        ]
        observation_details_str = (
            f"The observation ('obs') passed to the reward function is a dictionary representing the state *after* the action was taken.\n"
            f"Access its components using these string keys: {go1_obs_keys}\n"
            f"  - `obs['linear_velocity']`: Current velocity [vx, vy, vz] of the robot base.\n"
            f"  - `obs['angular_velocity']`: Current angular velocity [wx, wy, wz] of the robot base.\n"
            f"  - `obs['projected_gravity']`: Gravity vector projected onto the robot's base frame [gx, gy, gz]. Useful for orientation penalties (penalize non-zero gx, gy).\n"
            f"  - `obs['dofs_position']`: Current joint positions (relative to default).\n"
            f"  - `obs['dofs_velocity']`: Current joint velocities.\n"
            f"  - `obs['last_action']`: The action taken in the previous step.\n"
            f"Example: `forward_velocity = obs['linear_velocity'][0]`\n"
            f"\nTARGET VELOCITY: The target/goal velocity for the Go1 environment can be accessed directly from the environment object using `env.desired_velocity`.\n"
            f"Use `env.desired_velocity` to get the target [vx, vy, wz] vector.\n"
            f"Example: `target_vx = env.desired_velocity[0]`\n"
            f"\nIMPORTANT: Do NOT use 'velocity' or 'base_velocity' as keys - these do not exist. Use 'linear_velocity'.\n"
            f"\nREWARD FUNCTION SIGNATURE: `def custom_reward_function(obs, action, done, env):`\n"
            f"  - `obs`: The dictionary described above (state *after* action).\n"
            f"  - `action`: The action taken in the current step.\n"
            f"  - `done`: Boolean indicating if the episode terminated due to failure conditions (e.g., falling). Time limits are handled separately.\n"
            f"  - `env`: The environment instance, useful for accessing things like `env.desired_velocity`."
        )
    elif env_key == 'fetchReach': # Example for FetchReach if needed
        observation_details_str = (
             f"The observation ('obs') passed to the reward function is a dictionary.\n"
             f"Keys include: 'observation', 'achieved_goal', 'desired_goal'.\n"
             f"Example: `gripper_pos = obs['observation'][0:3]`\n"
             f"TARGET POSITION: Access the desired goal position using `env.unwrapped.goal`."
        )
    elif env_key == 'antmaze': # Example for AntMaze
         observation_details_str = (
             f"The observation ('obs') passed to the reward function is likely a numpy array.\n"
             f"Consult the specific AntMaze variant documentation for the exact structure and how to access goal information."
             # Or provide a known structure if available
         )
    # --- End Add Observation Space Details --- #

    # Explicitly access required values from the env_config object
    try:
        task_desc_val = env_config.task_description
        task_val = env_config.task
        env_name_val = env_config.env_name
    except AttributeError as e:
        logging.error(f"Missing expected key in environment config ('{cfg._metadata.defaults_list[0]['env']}.yaml'): {e}")
        raise e

    # Update user prompt formatting using the fetched values
    try:
        user_prompt_formatted = user_prompt_template.format(
            task_description=task_desc_val,
            task=task_val,
            env_name=env_name_val,
            env_details=env_details_str,
            observation_details=observation_details_str, # Pass the new details
        feedback=feedback_content,
    )
    except KeyError as e:
        logging.error(f"Placeholder key {e} not found in user_prompt.txt template. Available keys: task_description, task, env_name, env_details, observation_details, feedback.")
        raise e

    combined_system_prompt = system_prompt + "\n" + code_tip

    messages = [
        {"role": "system", "content": combined_system_prompt},
        {"role": "user", "content": user_prompt_formatted}
    ]

    try:
        response_text = "" # Initialize
        if llm_provider == 'openai':
            # --- OpenAI API Call ---
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                 logging.error("OpenAI API key not found in environment variable OPENAI_API_KEY.")
                 return "", "Error: OpenAI API key not found.", user_prompt_formatted
            openai.api_key = openai_api_key
            client = openai # Use the legacy client object structure if needed, or initialize openai.OpenAI()
            logging.info("Sending request to OpenAI API...")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                # temperature=cfg.get("temperature", 0.7) # Uncomment if using temperature
            )
            response_text = response.choices[0].message.content
            logging.info("Received response from OpenAI.")
            # --- End OpenAI API Call ---

        elif llm_provider == 'groq':
            # --- Groq API Call ---
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                logging.error("Groq API key not found in environment variable GROQ_API_KEY.")
                return "", "Error: Groq API key not found.", user_prompt_formatted
            client = Groq(api_key=groq_api_key)
            logging.info(f"Sending request to Groq API (Model: {model})...")
            # Note: Groq model names might differ (e.g., 'llama3-8b-8192')
            # Ensure cfg.model is set appropriately in your config for groq
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                # temperature=cfg.get("temperature", 0.7), # Optional
                # max_tokens=1024, # Optional
                # top_p=1, # Optional
                # stop=None, # Optional
                # stream=False, # Optional
            )
            response_text = chat_completion.choices[0].message.content
            logging.info("Received response from Groq.")
            # --- End Groq API Call ---

        else:
            logging.error(f"Unsupported LLM provider: {llm_provider}")
            return "", f"Error: Unsupported LLM provider '{llm_provider}'", user_prompt_formatted

    except Exception as e:
        logging.error(f"Error calling {llm_provider} API: {e}")
        # Log traceback for more details during debugging
        logging.error(traceback.format_exc())
        return "", f"Error calling {llm_provider} API: {e}", user_prompt_formatted # Return prompt used

    try:
        clean_code = extract_code_from_response(response_text)
        if not clean_code:
             logging.warning("LLM response did not contain a valid Python code block.")
    except ValueError as e:
        logging.error(f"Error extracting code: {e}")
        clean_code = ""

    return clean_code, response_text, user_prompt_formatted


def train_and_evaluate(cfg, py_reward_path, tb_log_dir_iter, results_folder_iter, iteration):
    """Trains the SAC agent, saves the model, and evaluates it."""
    final_train_reward = None
    avg_eval_reward = None
    error_message = None
    model_save_path = None

    try:
        # --- Determine Environment Factory Key (Revised Logic) ---
        matched_key = None

        # 1. Prioritize the key specified in the Hydra defaults list
        try:
            if cfg._metadata and hasattr(cfg._metadata, 'defaults_list'):
                 for default_item in cfg._metadata.defaults_list:
                     if isinstance(default_item, dict) and 'env' in default_item:
                         potential_key = default_item['env']
                         if potential_key in ENV_FACTORIES:
                             matched_key = potential_key
                             logging.info(f"Using environment key '{matched_key}' from Hydra defaults list.")
                             break # Found valid key
                         else:
                             logging.warning(f"Key '{potential_key}' found in defaults list but is not in ENV_FACTORIES: {list(ENV_FACTORIES.keys())}")
        except Exception as meta_e:
            logging.warning(f"Could not reliably access Hydra defaults list to determine env key: {meta_e}. Attempting fallback.")

        # 2. Fallback: If not found in defaults, try inferring from cfg.env.env_name
        if not matched_key:
            logging.warning("Could not determine environment key from defaults list. Attempting inference from cfg.env.env_name...")
            if hasattr(cfg, 'env') and hasattr(cfg.env, 'env_name'):
                env_name_from_cfg = cfg.env.env_name.lower()
                # Try matching known keys (case-insensitive)
                possible_matches = []
                for key in ENV_FACTORIES.keys():
                    if key.lower() in env_name_from_cfg:
                        possible_matches.append(key)

                if len(possible_matches) == 1:
                    matched_key = possible_matches[0]
                    logging.info(f"Inferred environment key '{matched_key}' from cfg.env.env_name ('{cfg.env.env_name}').")
                elif len(possible_matches) > 1:
                    logging.warning(f"Ambiguous environment name '{cfg.env.env_name}'. Matches multiple keys: {possible_matches}. Using the first match: {possible_matches[0]}")
                    matched_key = possible_matches[0]
                else:
                    logging.warning(f"Could not infer environment key from cfg.env.env_name '{cfg.env.env_name}'. No match in {list(ENV_FACTORIES.keys())}.)")
            else:
                logging.warning("cfg.env or cfg.env.env_name not found. Cannot infer environment key.")

        # 3. Final Check: If no key found, raise error
        if not matched_key:
            raise ValueError(
                f"Failed to determine a valid environment factory key. "
                f"Checked Hydra defaults and cfg.env.env_name ('{getattr(cfg.env, 'env_name', 'N/A')}'). "
                f"Known environments: {list(ENV_FACTORIES.keys())}"
            )
        # --- End Determine Environment Factory Key ---

        env_factory = get_env_factory(matched_key)
        logging.info(f"Iteration {iteration+1}: Creating environment '{matched_key}' with reward from: {py_reward_path}")

        env_kwargs = {}
        # Add specific args for certain environments based on config
        if matched_key == 'go1':
             env_kwargs['ctrl_type'] = cfg.env.get('ctrl_type', 'torque') # Use ctrl_type from config

        # Add render_mode for human rendering during training
        env_kwargs['render_mode'] = "human"

        try:
            # Pass collected kwargs (including render_mode)
            env = env_factory(reward_function_path=py_reward_path, **env_kwargs)
        except Exception as env_e:
            logging.error(f"Error creating environment '{matched_key}' with factory {ENV_FACTORIES[matched_key][1]}: {env_e}")
            logging.error(traceback.format_exc()) # Log full traceback for env creation error

        # --- End Dynamic Environment Creation ---

        env = Monitor(env)

        logging.info(f"Iteration {iteration+1}: Initializing SAC model. Logging TensorBoard to: {tb_log_dir_iter}")
        # Use total_timesteps from the global config (which interpolates from env config)
        total_timesteps = cfg.total_timesteps
        # Ensure total_timesteps is an integer
        if not isinstance(total_timesteps, int):
             try:
                 total_timesteps = int(total_timesteps)
                 logging.warning(f"Converted total_timesteps from {type(cfg.total_timesteps)} to int: {total_timesteps}")
             except ValueError:
                 logging.error(f"Invalid total_timesteps value: {cfg.total_timesteps}. Must be an integer.")
                 raise ValueError(f"Invalid total_timesteps: {cfg.total_timesteps}")

        model = SAC("MultiInputPolicy", env, verbose=0, tensorboard_log=tb_log_dir_iter)

        logging.info(f"Iteration {iteration+1}: Starting training for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps, tb_log_name="SAC", reset_num_timesteps=False)
        logging.info(f"Iteration {iteration+1}: Training complete.")

        model_save_path = os.path.join(results_folder_iter, f"sac_model_{iteration}.zip")
        logging.info(f"Iteration {iteration+1}: Saving trained model to: {model_save_path}")
        try:
            model.save(model_save_path)
            logging.info(f"Iteration {iteration+1}: Model saved successfully.")
        except Exception as save_e:
            logging.error(f"Iteration {iteration+1}: Failed to save model: {save_e}")
            model_save_path = None


        # Determine the specific SAC log path (usually env_name/SAC_1 or just SAC_1)
        potential_log_path = os.path.join(tb_log_dir_iter, "SAC_1")
        if not os.path.exists(potential_log_path):
             # Sometimes it might just be the tb_log_dir_iter itself if tb_log_name wasn't used as subdir
             if os.path.exists(tb_log_dir_iter) and any(f.startswith('events.out.tfevents') for f in os.listdir(tb_log_dir_iter)):
                sac_log_path = tb_log_dir_iter
                logging.warning(f"SAC_1 subdir not found, but events file found in parent: {tb_log_dir_iter}. Reading from parent.")
             else:
                 sac_log_path = tb_log_dir_iter # Still use parent path for error reporting
                 logging.warning(f"Expected SAC log directory '{potential_log_path}' not found, and no events file in parent {tb_log_dir_iter}. Reading attempt may fail.")
        else:
             sac_log_path = potential_log_path


        logging.info(f"Iteration {iteration+1}: Reading scalars from TensorBoard log: {sac_log_path}")
        scalar_data = get_all_scalars(sac_log_path)
        final_train_reward = get_final_scalar_value(scalar_data, 'rollout/ep_rew_mean')

        if final_train_reward is not None:
            logging.info(f"Iteration {iteration+1}: Final training mean episode reward (from TB): {final_train_reward:.2f}")
        else:
            logging.warning(f"Iteration {iteration+1}: Could not extract 'rollout/ep_rew_mean' from TensorBoard logs at {sac_log_path}.")


        logging.info(f"Iteration {iteration+1}: Starting evaluation for {cfg.eval_episodes} episodes...")
        # Use the same monitored env for evaluation (it resets automatically)
        eval_env = env
        total_reward = 0.0
        num_successful_evals = 0

        for ep in range(cfg.eval_episodes):
            obs, info = eval_env.reset()
            done = False
            ep_reward = 0.0
            step_count = 0
            # Get max_steps from Monitor wrapper if available, else default
            max_steps = getattr(env, '_max_episode_steps', 1000)

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                try:
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                     # Check done flags from Monitor (info['TimeLimit.truncated']) if direct flags unreliable
                     # done = terminated or truncated
                    done = info.get('TimeLimit.truncated', False) or terminated or truncated

                    ep_reward += reward
                    step_count += 1
                     # Optional: Add step limit check if Monitor doesn't handle it via TimeLimit wrapper
                     # if step_count >= max_steps:
                     #     done = True # Truncate if step limit reached
                except Exception as step_e:
                     logging.error(f"Error during evaluation step {step_count} in episode {ep+1}: {step_e}")
                     logging.error(traceback.format_exc())
                     done = True # End episode on error
                     ep_reward = -float('inf') # Penalize episodes ending in error

            # Only count reward if episode didn't end due to error
            if ep_reward > -float('inf'):
                total_reward += ep_reward
                num_successful_evals += 1
            else:
                 logging.warning(f"Episode {ep+1} ended due to an error, excluding from average reward calculation.")


        if num_successful_evals > 0:
             avg_eval_reward = total_reward / num_successful_evals
             logging.info(f"Iteration {iteration+1}: Evaluation complete. Average reward over {num_successful_evals}/{cfg.eval_episodes} successful episodes: {avg_eval_reward:.2f}")
        else:
             logging.error(f"Iteration {iteration+1}: No evaluation episodes completed successfully.")
             avg_eval_reward = None

        # Close the environment
        try:
             env.close()
        except Exception as close_e:
             logging.warning(f"Error closing environment: {close_e}")

    except Exception as e:
        logging.error(f"Error during training/evaluation in Iteration {iteration+1}: {e}")
        logging.error(traceback.format_exc())
        error_message = f"Error during training/evaluation: {e}\nTraceback:\n{traceback.format_exc()}"
        # Ensure rewards are None if error occurred before they were calculated
        avg_eval_reward = avg_eval_reward if 'avg_eval_reward' in locals() and avg_eval_reward is not None else None
        final_train_reward = final_train_reward if 'final_train_reward' in locals() and final_train_reward is not None else None


    return avg_eval_reward, final_train_reward, error_message, model_save_path


FEEDBACK_ANALYSIS_PROMPT = """
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
"""

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Setup Run-Specific Directories --- #
    # Get environment name from config - essential for directory structure
    env_name_str = "unknown_env" # Initialize with fallback
    try:
        # Try to get a user-friendly name if available
        env_name_raw = cfg.env.env_name
        env_name_str = sanitize_path_component(env_name_raw)
        logging.info(f"Determined environment name for paths: {env_name_str} (from cfg.env.env_name: '{env_name_raw}')")
    except AttributeError:
        logging.error("Could not determine environment name from cfg.env.env_name. Trying fallback...")
        # Fallback logic
        try:
            if cfg._metadata and hasattr(cfg._metadata, 'defaults_list'):
                for default_item in cfg._metadata.defaults_list:
                    if isinstance(default_item, dict) and 'env' in default_item:
                        potential_key = default_item['env']
                        env_name_str = sanitize_path_component(potential_key) # Sanitize the key itself
                        logging.info(f"Using fallback env name for paths: {env_name_str} (from defaults list key '{potential_key}')")
                        break # Exit loop once found
                # Check if fallback succeeded
                if env_name_str == "unknown_env":
                   logging.warning("Fallback failed to find env key in defaults list.")
            else:
                logging.warning("Could not access Hydra metadata for fallback env name determination.")
        except Exception as e:
            logging.error(f"Error during fallback env name determination: {e}. Using default '{env_name_str}'.")
    except Exception as e:
        # Catch any other unexpected errors during env name access
        logging.error(f"Unexpected error determining environment name: {e}. Using default '{env_name_str}'.")

    # Generate a unique timestamp for this run
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define base paths from config
    base_results_dir = cfg.paths.results_dir
    base_models_dir = cfg.paths.model_dir

    # Create unique directories for this run: base/env_name/timestamp
    current_run_results_dir = os.path.join(base_results_dir, env_name_str, run_timestamp)
    current_run_models_dir = os.path.join(base_models_dir, env_name_str, run_timestamp)
    os.makedirs(current_run_results_dir, exist_ok=True)
    os.makedirs(current_run_models_dir, exist_ok=True)

    # Define TensorBoard base directory for this run
    tb_log_dir_base = os.path.join(current_run_results_dir, "tensorboard_logs")
    os.makedirs(tb_log_dir_base, exist_ok=True)

    # Define conversation history file path for this run
    conversation_file = os.path.join(current_run_results_dir, "conversation_history.md")

    logging.info(f"Starting Run: {run_timestamp}")
    logging.info(f"Results will be saved in: {current_run_results_dir}")
    logging.info(f"Models will be saved in: {current_run_models_dir}")
    logging.info(f"TensorBoard logs base: {tb_log_dir_base}")
    # --- End Setup Run-Specific Directories --- #

    # Determine LLM provider and check API key
    # Access provider from the llm section
    try:
        llm_provider = cfg.llm.provider.lower()
    except AttributeError:
        logging.error("LLM provider setting ('llm.provider') not found in configuration. Please ensure it exists in config.yaml under the 'llm' section.")
        return
    logging.info(f"Using LLM Provider: {llm_provider}")
    if llm_provider == 'openai':
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logging.error("OpenAI API key not found in environment variable OPENAI_API_KEY. Please set it.")
            return
        # OpenAI key is set globally within get_llm_response if needed by legacy client
    elif llm_provider == 'groq':
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logging.error("Groq API key not found in environment variable GROQ_API_KEY. Please set it.")
            return
    else:
        logging.error(f"Invalid llm_provider specified in config: '{llm_provider}'. Use 'openai' or 'groq'.")
        return

    # Setup prompt paths
    prompt_base_path = os.path.abspath("/home/ken2/PCD/utils/prompts")
    system_prompt_path = os.path.join(prompt_base_path, "system_prompt.txt")
    user_prompt_path = os.path.join(prompt_base_path, "user_prompt.txt")
    code_tip_path = os.path.join(prompt_base_path, "code_output_tip.txt")
    for f_path in [system_prompt_path, user_prompt_path, code_tip_path]:
         if not os.path.exists(f_path):
              logging.error(f"Required prompt file not found: {f_path}")
              return

    # Initial feedback and code placeholders
    current_feedback_content = "This is the first iteration. Please generate an initial reward function based on the task description and environment details provided."
    previous_reward_code = "None (first iteration)"
    all_results_summary = []

    for i in range(cfg.iterations):
        iteration_str = f"Iteration {i+1}/{cfg.iterations}"
        logging.info(f"========== Starting {iteration_str} ==========")

        # Save reward function in the run-specific results directory
        reward_py_path = os.path.join(current_run_results_dir, f"reward_function_{i}.py")
        tb_log_dir_iter = os.path.join(tb_log_dir_base, f"iteration_{i}")
        os.makedirs(tb_log_dir_iter, exist_ok=True)

        reward_function_code = ""
        conversation_text = ""
        user_prompt_for_llm = ""
        llm_attempts = 0
        max_llm_attempts = 10

        while not reward_function_code and llm_attempts < max_llm_attempts:
             llm_attempts += 1
             logging.info(f"{iteration_str}: LLM Attempt {llm_attempts}/{max_llm_attempts}")

             # Construct feedback for LLM, including previous code if available
             feedback_for_llm = current_feedback_content
             # The previous code is now added within the feedback string generation logic below
             # No need to add it separately here unless the structure changes

             # Pass the llm_provider and config
             reward_function_code, conversation_text, user_prompt_for_llm = get_llm_response(
                 llm_provider=llm_provider, # Pass the selected provider
                 model=cfg.llm.model, # Access model from the llm section
                 system_prompt_path=system_prompt_path,
                 user_prompt_path=user_prompt_path,
                 code_tip_path=code_tip_path,
                 cfg=cfg, # Pass the full config
                 feedback_content=feedback_for_llm, # Pass the constructed feedback
                 previous_reward_code=previous_reward_code # Pass code separately if needed by template
             )

             # Check for generic API error message
             if f"Error calling {llm_provider} API" in conversation_text:
                  logging.error(f"{iteration_str}: LLM API call failed. Aborting iteration.")
                  # Log failure details more concisely
                  with open(conversation_file, "a", encoding="utf-8") as conv_file:
                        conv_file.write(f"## {iteration_str}: Failed - {llm_provider.upper()} API Error\n")
                        conv_file.write(f"**Error:** `{conversation_text}`\n")
                        conv_file.write(f"**Attempted Prompt:**\n```\n{user_prompt_for_llm}\n```\n---\n\n")
                        all_results_summary.append(f"Iter {i+1}: Failed - {llm_provider.upper()} API Error")
                  # Ensure reward_function_code remains empty to trigger skip logic below
                  reward_function_code = ""
                  break # Exit the LLM attempt loop for this iteration

             elif not reward_function_code:
                  logging.warning(f"{iteration_str}: Could not extract valid Python code from LLM response (Attempt {llm_attempts}).")
                  if llm_attempts < max_llm_attempts:
                       logging.info("Retrying LLM call.")
                       # Modify feedback slightly for retry
                       # Make sure this doesn't get added multiple times if retries happen across iterations
                       retry_feedback = "\n\n[System Retry Feedback]: The previous response did not contain a valid Python code block. Please ensure the reward function code is clearly marked within ```python ... ``` tags and follows the required signature `def custom_reward_function(obs, action, done, env):`. Use only numpy and standard python.\n"
                       if retry_feedback not in current_feedback_content:
                            current_feedback_content += retry_feedback
                       time.sleep(2) # Small delay before retry
                  else:
                       logging.error(f"{iteration_str}: Failed to get valid reward code from LLM after {max_llm_attempts} attempts.")
                       # Log failure details more concisely
                       with open(conversation_file, "a", encoding="utf-8") as conv_file:
                            conv_file.write(f"## {iteration_str}: Failed - No Valid Code from LLM\n")
                            conv_file.write(f"**Final LLM Response:**\n```\n{conversation_text}\n```\n\n")
                            conv_file.write(f"**Attempted Prompt:**\n```\n{user_prompt_for_llm}\n```\n---\n\n")
                       all_results_summary.append(f"Iter {i+1}: Failed - No valid code from LLM")
                       # Ensure reward_function_code remains empty to trigger skip logic below
                       reward_function_code = ""
                       break # Exit LLM attempt loop
             else:
                  logging.info(f"{iteration_str}: Successfully generated and extracted reward code.")
                  # Break the LLM attempt loop if code is found
                  break

        # --- After LLM attempt loop --- #

        # Check if we failed to get code after all attempts
        if not reward_function_code:
             logging.warning(f"{iteration_str}: Skipping Training/Evaluation due to failure in LLM response generation.")
             # Prepare feedback for the *next* iteration's LLM call
             # This feedback was already prepared during the retry logic or initial failure logging
             # We just need to ensure `current_feedback_content` reflects the failure state
             current_feedback_content = FEEDBACK_ANALYSIS_PROMPT + (
                  f"\n**Status:** Failed to generate usable code in the previous attempt after {llm_attempts} tries.\n"
                  f"**Last LLM Response:**\n{conversation_text}\n\n"
                  "Please try again, carefully following the instructions and ensuring the code is correctly formatted with the signature `def custom_reward_function(obs, action, done, env):`.\n"
             )
             # No valid code from this iteration to become the 'previous' code
             # Keep the existing `previous_reward_code` for the next iteration's context
             # Make sure summary reflects failure reason
             if f"Iter {i+1}: Failed - No valid code from LLM" not in all_results_summary and f"Iter {i+1}: Failed - {llm_provider.upper()} API Error" not in all_results_summary:
                 # Summarize specific failure type if possible
                 failure_reason = f"Failed - {llm_provider.upper()} API Error" if f"Iter {i+1}: Failed - {llm_provider.upper()} API Error" in all_results_summary else "Failed - LLM Code Gen"
                 all_results_summary.append(f"Iter {i+1}: {failure_reason}")
             continue # Skip to the next iteration


        # --- Code generated, proceed with saving and training --- #
        logging.info(f"{iteration_str}: Saving generated reward function to: {reward_py_path}")
        try:
            with open(reward_py_path, "w", encoding="utf-8") as f:
                f.write("import numpy as np\n") # Ensure numpy is imported
                # Add other potential common imports if needed by LLM code
                # f.write("import math\n")
                f.write(reward_function_code)
            # Update previous_reward_code *only* if saving was successful and code is valid
            # This is the code that will be included in the *next* iteration's feedback prompt
            previous_reward_code = reward_function_code
        except IOError as e:
             logging.error(f"{iteration_str}: Failed to write reward function file: {e}. Skipping training/evaluation.")
             # Log failure details concisely
             with open(conversation_file, "a", encoding="utf-8") as conv_file:
                  conv_file.write(f"## {iteration_str}: Failed - Cannot Save Code\n")
                  conv_file.write(f"**Error:** `{str(e)}`\n")
                  conv_file.write(f"**Generated Code (Unsaved):**\n```python\n{reward_function_code}\n```\n---\n\n")
             # Prepare feedback for the *next* iteration's LLM call OUTSIDE the with block
             # Explicitly closing parenthesis after all strings are concatenated.
             current_feedback_content = FEEDBACK_ANALYSIS_PROMPT + (
                 f"\n**Status:** Internal error: Failed to save the previously generated code.\n"
                 f"**Error:** {e}\n"
                 "Please regenerate the reward function."
             )
             # Keep the existing `previous_reward_code` as the generation failed at saving stage
             all_results_summary.append(f"Iter {i+1}: Failed - Cannot save code")
             continue # Skip to next iteration

        logging.info(f"{iteration_str}: Starting training and evaluation...")
        avg_eval_reward, final_train_reward, error_message, saved_model_path = train_and_evaluate(
            cfg, # Pass cfg object
            reward_py_path, # Path to reward function
            tb_log_dir_iter, # Tensorboard log dir for this iteration
            current_run_models_dir, # Pass the run-specific *models* directory for saving
            i # Iteration number
        )

        # Precompute evaluation and training reward strings
        eval_reward_str = f"{avg_eval_reward:.2f}" if avg_eval_reward is not None else "N/A"
        train_reward_str = f"{final_train_reward:.2f}" if final_train_reward is not None else "N/A"

        if avg_eval_reward is None:
             status = "Partial Success - Evaluation Failed"
             logging.warning(f"{iteration_str}: Evaluation did not complete successfully (avg_eval_reward is None).")
             current_status_message = (
                 f"**Status:** {status}\n"
                 f"**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `{train_reward_str}`\n"
                 f"**Evaluation Result:** Failed to get an average reward (likely no episodes completed).\n\n"
                 "Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals)."
             )
        else:
             status = "Success"
             logging.info(f"{iteration_str}: Training/Evaluation successful.")
             current_status_message = (
                 f"**Status:** {status}\n"
                 f"**Results:**\n"
                 f"- Average Evaluation Reward: `{eval_reward_str}`\n"
                 f"- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `{train_reward_str}`\n\n"
                 f"Based on these results and the task goal ('{cfg.env.task}'), analyze the reward function code (below) and suggest improvements."
             )

        current_feedback_content = FEEDBACK_ANALYSIS_PROMPT + \
                                   f"\n{current_status_message}" + \
                                   f"\n\n**Previous Reward Function Code:**\n```python\n{previous_reward_code}\n```"

        logging.info(f"{iteration_str}: Logging results to {conversation_file}")
        with open(conversation_file, "a", encoding="utf-8") as conv_file:
             conv_file.write(f"## {iteration_str}\n\n")
             conv_file.write(f"**Status:** {status}\n\n")
             conv_file.write(
                 f"**User Prompt to LLM (leading to this iteration's code):**\n"
                 f"*Note: Contains feedback from the iteration before this one.*\n```\n{user_prompt_for_llm}\n```\n\n"
             )
             conv_file.write(f"**LLM Response:**\n```\n{conversation_text}\n```\n\n")
             conv_file.write(
                 f"**Generated Reward Code (saved to {os.path.basename(reward_py_path)}):**\n"
                 f"```python\n{reward_function_code}\n```\n\n"
             )
             conv_file.write("**Training & Evaluation Results:**\n")
             # Use relative paths for logging within the run directory
             relative_tb_path = os.path.relpath(tb_log_dir_iter, current_run_results_dir)
             conv_file.write(f"- TensorBoard Log Directory: `{relative_tb_path}`\n")
             if saved_model_path:
                 # Model path is now relative to the base models dir, log the run-specific model path
                 # Path relative to workspace root: models/env_name/timestamp/model_iter_N.zip
                 workspace_rel_model_path = os.path.relpath(saved_model_path, os.getcwd()) # Or adjust base path if needed
                 conv_file.write(f"- Saved Model: `{workspace_rel_model_path}`\n")
             else:
                 conv_file.write("- Saved Model: Failed or Skipped\n")
             conv_file.write(f"- Average Evaluation Reward: `{eval_reward_str}`\n")
             conv_file.write(f"- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `{train_reward_str}`\n")
             if error_message:
                 conv_file.write("- Error Encountered: Yes (See Status/Feedback)\n")
             conv_file.write("\n")
             conv_file.write(
                 f"**Feedback Content Generated for Next Iteration:**\n```\n{current_feedback_content}\n```\n\n"
             )
             conv_file.write("---\n\n")

        eval_reward_summary = f"{avg_eval_reward:.2f}" if avg_eval_reward is not None else "N/A"
        train_reward_summary = f"{final_train_reward:.2f}" if final_train_reward is not None else "N/A"
        summary = f"Iter {i+1}: Status='{status}', Eval Reward={eval_reward_summary}, Train Reward={train_reward_summary}, Model Saved='{bool(saved_model_path)}', Error='{bool(error_message)}'"
        all_results_summary.append(summary)
        logging.info(f"========== Finished {iteration_str} ==========\n")

    logging.info("Iterative reward function generation complete.")
    logging.info("Final Results Summary:")
    if not all_results_summary:
         logging.info("  No iterations completed fully.")
    else:
         for result_line in all_results_summary:
              logging.info(f"  {result_line}")
    logging.info(f"Detailed logs and artifacts for run {run_timestamp} (env: {env_name_str}) saved in: {current_run_results_dir}")
    logging.info(f"Models for run {run_timestamp} (env: {env_name_str}) saved in: {current_run_models_dir}")


if __name__ == "__main__" :
             main()