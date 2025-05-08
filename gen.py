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
import numpy as np
from pathlib import Path

from utils.file_to_string import file_to_string
from utils.extracct_code import extract_code_from_response
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from tensorboard.backend.event_processing import event_accumulator
import traceback
import time
from stable_baselines3.common.callbacks import BaseCallback

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

# --- Add Callback Definition --- #
class ProgressCallback(BaseCallback):
    """
    A simple callback that logs progress during training.
    """
    def __init__(self, check_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self._last_log_step = 0

    def _on_step(self) -> bool:
        """Log progress every check_freq steps."""
        # Check if it's time to log (or first step)
        if self.num_timesteps % self.check_freq == 0 or self.num_timesteps == 1:
            # Avoid logging the same step multiple times if check_freq is small or steps are skipped
            if self.num_timesteps > self._last_log_step:
                total_steps = self.model.num_timesteps # Get total timesteps from the model if available
                if hasattr(self.training_env, 'spec') and self.training_env.spec is not None:
                    max_steps = self.training_env.spec.max_episode_steps
                else:
                    # Fallback if max_episode_steps isn't available directly
                    max_steps = getattr(self.model, 'total_timesteps', '?') # Or use the planned total if spec missing

                logging.info(f"  Training Progress: Timestep {self.num_timesteps}/{total_steps}")
                self._last_log_step = self.num_timesteps
        return True # Continue training
# --- End Callback Definition --- #

def get_all_scalars(tb_log_dir):
    """Reads all scalar data from a TensorBoard log directory."""
    try:
        logging.info(f"Waiting 3 seconds for filesystem sync before reading: {tb_log_dir}")
        time.sleep(3)

        if not os.path.exists(tb_log_dir):
            logging.warning(f"TensorBoard log directory does not exist: {tb_log_dir}")
            return {}

        logging.info(f"Initializing EventAccumulator for: {tb_log_dir}")
        try:
            ea = event_accumulator.EventAccumulator(
                tb_log_dir,
                size_guidance={event_accumulator.SCALARS: 0}
            )
            ea.Reload()
            logging.info(f"EventAccumulator loaded successfully for: {tb_log_dir}")
        except Exception as ea_e:
             logging.error(f"Failed to initialize or load EventAccumulator for {tb_log_dir}: {ea_e}")
             logging.error(traceback.format_exc())
             return {}

        scalar_dict = {}
        tags = ea.Tags().get('scalars', [])
        logging.info(f"Found scalar tags in {tb_log_dir}: {tags}")

        if not tags:
             logging.warning(f"No scalar tags found in {tb_log_dir}. Ensure training ran and generated logs.")
             return {}

        for tag in tags:
            try:
                scalar_events = ea.Scalars(tag)
                scalar_dict[tag] = [(s.step, s.value) for s in scalar_events]
                logging.debug(f"Read {len(scalar_dict[tag])} events for tag '{tag}'")
            except Exception as tag_e:
                 logging.error(f"Error reading data for tag '{tag}' in {tb_log_dir}: {tag_e}")
                 scalar_dict[tag] = []

        if not scalar_dict:
            logging.warning(f"No scalar data could be extracted despite finding tags in {tb_log_dir}. Check log file integrity.")

        return scalar_dict
    except Exception as e:
        logging.error(f"Unexpected error reading TensorBoard logs from {tb_log_dir}: {e}\n{traceback.format_exc()}")
        return {}

def get_final_scalar_value(scalar_dict, tag_name):
    """Gets the latest value for a specific scalar tag."""
    if tag_name in scalar_dict and scalar_dict[tag_name]:
        try:
            valid_scalars = [(step, value) for step, value in scalar_dict[tag_name] if np.isfinite(value)]
            if not valid_scalars:
                 logging.warning(f"Scalar tag '{tag_name}' contained only non-finite (NaN/inf) values.")
                 return None
            sorted_scalars = sorted(valid_scalars, key=lambda x: x[0])
            final_value = sorted_scalars[-1][1]
            logging.info(f"Final value for tag '{tag_name}': {final_value:.4f} at step {sorted_scalars[-1][0]}")
            return final_value
        except IndexError:
             logging.warning(f"Scalar tag '{tag_name}' was found but contained no valid finite data after filtering.")
             return None
        except Exception as e:
             logging.error(f"Error processing scalar tag '{tag_name}': {e}")
            return None
    else:
        logging.info(f"Scalar tag '{tag_name}' not found or empty in provided scalar_dict.")
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
    # Determine the environment key to provide specific observation details
    # This logic mirrors the key determination in train_and_evaluate for consistency
    env_key = None
    try:
    if cfg._metadata and hasattr(cfg._metadata, 'defaults_list'):
        for default_item in cfg._metadata.defaults_list:
            if isinstance(default_item, dict) and 'env' in default_item:
                    potential_key = default_item['env']
                    if potential_key in ENV_FACTORIES: # Check if it's a known key
                        env_key = potential_key
                break
        if not env_key and hasattr(cfg, 'env') and hasattr(cfg.env, 'env_name'):
             env_name_from_cfg = cfg.env.env_name.lower()
             possible_matches = [key for key in ENV_FACTORIES if key.lower() in env_name_from_cfg]
             if len(possible_matches) >= 1:
                  env_key = possible_matches[0] # Use first match if ambiguous or unique
    except Exception as e:
         logging.warning(f"Could not robustly determine env_key for prompt generation: {e}. Proceeding with generic observation details.")
         env_key = None # Ensure it's None if determination failed

    # Initialize observation details string
    observation_details_str = "Observation details are environment-specific. Ensure reward function correctly interprets the 'obs' structure."

    # If it is go1, provide the specific structure
    if env_key == 'go1':
        observation_details_str = (
            "**IMPORTANT: Observation Space Structure (Go1 Environment)**\n"
            "The 'obs' argument passed to the function is a Python **dictionary**, representing the state *after* the action was taken.\n"
            "Access its components using these string keys:\n"
            "  - `obs['linear_velocity']`: Current base linear velocity [vx, vy, vz] (NumPy array).\n"
            "  - `obs['angular_velocity']`: Current base angular velocity [wx, wy, wz] (NumPy array).\n"
            "  - `obs['projected_gravity']`: Gravity vector projected onto base frame [gx, gy, gz] (NumPy array). Useful for orientation penalties.\n"
            "  - `obs['dofs_position']`: Current joint positions (relative to default) (NumPy array).\n"
            "  - `obs['dofs_velocity']`: Current joint velocities (NumPy array).\n"
            "  - `obs['last_action']`: The action (torques) taken in the previous step (NumPy array).\n"
            "\n**Accessing Data:**\n"
            "- Use dictionary key access, e.g., `current_lin_vel = obs['linear_velocity']`. \n"
            "- Remember that values like velocity might be scaled; check the environment code if needed, but accessing `env.data.qvel` might provide unscaled values directly.\n"
            "- Do NOT use array slicing (e.g., `obs[0:3]`) as `obs` is a dictionary.\n"
            "\n**Target Velocity:**\n"
            "- Access the target velocity vector [vx, vy, wz] using `env.desired_velocity`.\n"
            "\n**Available Environment Attributes/Methods in `env`:**\n"
            "You can access useful attributes from the `env` object:\n"
            "  - `env.desired_velocity`: Target velocity [vx, vy, wz] (unscaled).\n"
            "  - `env.dt`: Simulation timestep.\n"
            "  - `env.data.qvel`: Full velocity vector [base_linear, base_angular, joint_velocities] (unscaled).\n"
            "  - `env.is_healthy`: Boolean indicating if the robot is stable (check height/orientation limits).\n"
            "  - `env.non_flat_base_cost`: Property for non-flat base penalty.\n"
            "  - `env.torque_cost`: Property for torque penalty.\n"
            "  - (And others listed in the original Go1MujocoEnv code example...)\n"
            "\n**Reward Objectives:**\n"
            "Create a reward function that encourages:\n"
            "  1. **Velocity Tracking:** Match desired linear (XY) and angular (Z) velocity (`env.desired_velocity`) using current velocities (`obs['linear_velocity']`, `obs['angular_velocity']`).\\n"
            "  2. **Forward Progress:** Reward actual forward velocity (`obs['linear_velocity'][0]`) ONLY when desired forward velocity (`env.desired_velocity[0]`) is positive.\\n"
            "  3. **Stability:** Reward staying healthy (`env.is_healthy`) AND penalize non-flat orientation (use `env.non_flat_base_cost`). Consider adding a small constant 'alive' bonus per step.\\n"
            "  4. **Efficiency/Smoothness:** Penalize high torques (`env.torque_cost`), high joint velocities (`obs['dofs_velocity']`), and potentially high joint accelerations (`env.data.qacc[6:]` if needed, but focus on torque/velocity first).\\n"
            "\n**Implementation Guidance:**\n"
            "- **Use Explicit Weights:** Define weights (e.g., `W_FORWARD = 1.5`, `W_TORQUE = -0.0002`) for each component and sum the weighted terms. This is better than implicit scaling.\\n"
            "- **Return Value:** Ensure the function returns a single float value. Returning `None` or `NaN` will cause errors.\\n"
            "\n**Reference: Original Reward Logic (Example):**\n"
            "```python\\n"
            "# --- Define Weights --- \\n"
            "W_LINEAR_VEL = 2.0\\n"
            "W_ANGULAR_VEL = 1.0\\n"
            "W_FORWARD = 1.5\\n"
            "W_STABILITY = 1.0 # For env.is_healthy check\\n"
            "W_ALIVE = 0.05 # Small bonus per step\\n"
            "W_ORIENT = -1.0 # Penalty for non-flat base\\n"
            "W_TORQUE = -0.0002 # Penalty for torque\\n"
            "W_JOINT_VEL = -0.01 # Penalty for joint velocity\\n"
            "\\n"
            "# --- Get Data --- \\n"
            "current_lin_vel = obs['linear_velocity']\\n"
            "current_ang_vel = obs['angular_velocity']\\n"
            "joint_velocities = obs['dofs_velocity']\\n"
            "\\n"
            "# --- Calculate Components --- \\n"
            "# Velocity Tracking\\n"
            "lin_vel_error = np.sum(np.square(env.desired_velocity[:2] - current_lin_vel[:2]))\\n"
            "ang_vel_error = np.square(env.desired_velocity[2] - current_ang_vel[2])\\n"
            "linear_tracking_reward = np.exp(-lin_vel_error / 0.25) # Sigma = 0.25\\n"
            "angular_tracking_reward = np.exp(-ang_vel_error / 0.25)\\n"
            "\\n"
            "# Forward Progress\\n"
            "current_forward_vel = obs['linear_velocity'][0]\\n"
            "forward_progress_reward = W_FORWARD * current_forward_vel if env.desired_velocity[0] > 0.1 else 0.0\\n"
            "\\n"
            "# Stability\\n"
            "stability_reward = W_STABILITY if env.is_healthy else 0.0 # Reward for being healthy\\n"
            "alive_bonus = W_ALIVE # Constant bonus per step\\n"
            "orientation_penalty = W_ORIENT * env.non_flat_base_cost\\n"
            "\\n"
            "# Efficiency/Smoothness\\n"
            "torque_penalty = W_TORQUE * env.torque_cost\\n"
            "joint_vel_penalty = W_JOINT_VEL * np.sum(np.square(joint_velocities))\\n"
            "\\n"
            "# --- Combine Terms --- \\n"
            "reward = ( \\n"
            "    linear_tracking_reward * W_LINEAR_VEL + \\n"
            "    angular_tracking_reward * W_ANGULAR_VEL + \\n"
            "    forward_progress_reward + \\n"
            "    stability_reward + \\n"
            "    alive_bonus + \\n"
            "    orientation_penalty + \\n"
            "    torque_penalty + \\n"
            "    joint_vel_penalty \\n"
            ")\\n"
            "```"
        )
    elif env_key == 'fetchReach': # Example for FetchReach if needed
        observation_details_str = (
             "The observation ('obs') passed to the reward function is a dictionary.\n"
             "Keys include: 'observation', 'achieved_goal', 'desired_goal'. \n"
             "Example: `gripper_pos = obs['observation'][0:3]` \n"
             "TARGET POSITION: Access the desired goal position using `env.unwrapped.goal`."
        )
    elif env_key == 'antmaze': # Example for AntMaze
         observation_details_str = (
             "The observation ('obs') passed to the reward function for AntMaze variants is often a NumPy array.\n"
             "Consult the specific AntMaze variant documentation for the exact structure and how to access goal information."
             # Or provide a known structure if available
         )
    else:
         logging.warning(f"No specific observation details provided in prompt logic for environment key: {env_key}")
         # Keep the default generic message
         observation_details_str = (
              "Observation details are environment-specific. "
              "Ensure reward function correctly interprets the 'obs' structure."
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
    env = None # Initialize env to None

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
            env_instance = env_factory(reward_function_path=py_reward_path, **env_kwargs)
            # Wrap with Monitor for SB3 logging
            env = Monitor(env_instance)
            logging.info(f"Environment '{matched_key}' created and wrapped with Monitor.")
        except Exception as env_e:
            logging.error(f"Error creating environment '{matched_key}': {env_e}")
            logging.error(traceback.format_exc())
            # Set specific error message for feedback
            error_message = f"Failed to create environment '{matched_key}'. Check factory and reward function compatibility. Error: {env_e}"
            raise # Re-raise to stop the iteration

        # --- End Dynamic Environment Creation ---

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

        # --- Instantiate Callback --- #
        log_freq = cfg.get('log_freq', 10000) # Get logging frequency from config, default 10k
        progress_callback = ProgressCallback(check_freq=log_freq)
        # --- End Instantiate Callback --- #

        logging.info(f"Iteration {iteration+1}: Starting training for {total_timesteps} timesteps... (Logging progress every {log_freq} steps)")
        # --- Training with Error Handling for Reward ---
        try:
            # --- Pass Callback to model.learn --- #
            model.learn(total_timesteps=total_timesteps, tb_log_name="SAC", reset_num_timesteps=False, callback=progress_callback)
            # --- End Pass Callback --- #
            logging.info(f"Iteration {iteration+1}: Training finished normally.")
        except (TypeError, ValueError) as reward_error:
             # Catch errors potentially caused by non-numeric rewards during SB3 updates
             if "unhashable type" in str(reward_error) or "must be real number" in str(reward_error) or "numpy() argument must be" in str(reward_error):
                 logging.error(f"Iteration {iteration+1}: Training stopped likely due to invalid reward value returned by custom function: {reward_error}")
                 error_message = f"Training failed: Custom reward function returned non-numeric value ({type(reward_error).__name__}: {reward_error}). Check reward calculation logic."
                 logging.error(traceback.format_exc()) # Log full traceback for debugging
                 # Skip saving and evaluation if training failed due to bad reward
             else:
                 # Re-raise other unexpected TypeErrors/ValueErrors
                 logging.error(f"Iteration {iteration+1}: Unexpected training error: {reward_error}")
                 error_message = f"Unexpected Training Error: {reward_error}"
                 logging.error(traceback.format_exc())
                 raise reward_error # Re-raise to stop iteration

        except Exception as train_e:
             # Catch other potential training errors
             logging.error(f"Iteration {iteration+1}: Training failed with unexpected error: {train_e}")
             error_message = f"Unexpected Training Error: {train_e}"
             logging.error(traceback.format_exc())
             raise train_e # Re-raise other errors

        # --- End Training with Error Handling ---

        # Only proceed to save/evaluate if training didn't raise a critical error
        if error_message and "Training failed: Custom reward function" in error_message:
            logging.warning("Skipping model saving and evaluation due to invalid reward function error during training.")
        else:
            # Save Model
        model_save_path = os.path.join(results_folder_iter, f"sac_model_{iteration}.zip")
        logging.info(f"Iteration {iteration+1}: Saving trained model to: {model_save_path}")
            try: model.save(model_save_path)
        except Exception as save_e:
            logging.error(f"Iteration {iteration+1}: Failed to save model: {save_e}")
                model_save_path = None # Mark as not saved

            # Read Scalars (improved logging inside function)
            sac_log_path = os.path.join(tb_log_dir_iter, "SAC_1")
            if not os.path.exists(sac_log_path):
             if os.path.exists(tb_log_dir_iter) and any(f.startswith('events.out.tfevents') for f in os.listdir(tb_log_dir_iter)):
                sac_log_path = tb_log_dir_iter
                     logging.info(f"Reading TB scalars from parent directory: {sac_log_path}")
             else:
                     sac_log_path = tb_log_dir_iter
                     logging.warning(f"TB directory '{potential_log_path}' or parent '{tb_log_dir_iter}' seems empty or missing. Check SB3 logging setup (tensorboard_log path, tb_log_name='SAC'). Reading attempt may fail.")
        else:
                 logging.info(f"Reading TB scalars from: {sac_log_path}")

        scalar_data = get_all_scalars(sac_log_path)
        final_train_reward = get_final_scalar_value(scalar_data, 'rollout/ep_rew_mean')
            if final_train_reward is None:
                 logging.warning(f"Iteration {iteration+1}: Could not extract final 'rollout/ep_rew_mean' from {sac_log_path}. Check tag name and log file contents.")

            # Evaluation
        logging.info(f"Iteration {iteration+1}: Starting evaluation for {cfg.eval_episodes} episodes...")
            eval_env = env # Re-use the monitored env
        total_reward = 0.0
        num_successful_evals = 0
        for ep in range(cfg.eval_episodes):
            obs, info = eval_env.reset()
            done = False
            ep_reward = 0.0
            step_count = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                try:
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                         # --- Check for non-numeric reward during eval ---
                         if not isinstance(reward, (int, float)) or not np.isfinite(reward):
                              logging.error(f"Evaluation Step Error: Reward function returned non-numeric value: {reward} (type: {type(reward)})")
                              # Provide specific error message for feedback
                              error_message = f"Evaluation failed: Reward function returned non-numeric value '{reward}' (type: {type(reward)})."
                              ep_reward = -float('inf') # Mark episode as failed
                              done = True # End episode
                              continue # Skip rest of step logic
                         # --- End Check ---
                    done = info.get('TimeLimit.truncated', False) or terminated or truncated
                    ep_reward += reward
                    step_count += 1
                except Exception as step_e:
                     logging.error(f"Error during evaluation step {step_count} in episode {ep+1}: {step_e}")
                     logging.error(traceback.format_exc())
                          # Provide specific error message for feedback
                          error_message = f"Evaluation failed during env.step(): {step_e}"
                          ep_reward = -float('inf') # Mark episode as failed
                          done = True # End episode

            if ep_reward > -float('inf'):
                total_reward += ep_reward
                num_successful_evals += 1
            else:
                     logging.warning(f"Episode {ep+1} failed (reward={ep_reward}), excluding from average reward calculation.")

        if num_successful_evals > 0:
             avg_eval_reward = total_reward / num_successful_evals
             logging.info(f"Iteration {iteration+1}: Evaluation complete. Average reward over {num_successful_evals}/{cfg.eval_episodes} successful episodes: {avg_eval_reward:.2f}")
        else:
             logging.error(f"Iteration {iteration+1}: No evaluation episodes completed successfully.")
                 # Keep existing error message if one occurred, otherwise set a new one
                 if not error_message:
                      error_message = "Evaluation failed: No episodes completed successfully."
                 avg_eval_reward = None # Ensure it's None

    except Exception as e:
        # General catch-all for errors *outside* the specific training/eval logic (e.g., env creation)
        logging.error(f"Critical error during Iteration {iteration+1} setup or execution: {e}")
        logging.error(traceback.format_exc())
        # Set error message if not already set by inner try-except blocks
        if not error_message:
            error_message = f"Critical Iteration Error: {e}\nTraceback:\n{traceback.format_exc()}"
        # Ensure rewards are None if error occurred before they were calculated
        avg_eval_reward = avg_eval_reward if 'avg_eval_reward' in locals() and avg_eval_reward is not None else None
        final_train_reward = final_train_reward if 'final_train_reward' in locals() and final_train_reward is not None else None

    finally:
        # Ensure environment is closed even if errors occurred
        if env is not None:
             try:
                 logging.info(f"Closing environment for Iteration {iteration+1}")
                 env.close()
             except Exception as close_e:
                 logging.warning(f"Error closing environment: {close_e}")

    # Return results including the specific error message
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

    # Setup prompt paths using Hydra's CWD interpolation (relative to where script is run)
    # Assumes prompts are in 'utils/prompts' relative to the project root where you run the script.
    # Accessing Hydra's CWD requires it to be passed or accessed differently.
    # A simpler approach is relative to the script file itself, assuming standard structure.
    # However, using relative paths directly in the config is often cleaner.

    # Let's assume paths are defined relative to project root in config (cfg.paths.prompt_dir)
    try:
        prompt_base_path = Path(cfg.paths.prompt_dir).resolve()
        logging.info(f"Resolved prompt base path: {prompt_base_path}")
    except AttributeError:
        logging.error("Config missing 'paths.prompt_dir'. Please define the relative path to the prompts directory (e.g., 'utils/prompts') in your config.")
        # Fallback using __file__ (less reliable if structure changes)
        script_dir = Path(__file__).parent.resolve()
        prompt_base_path = (script_dir / "../utils/prompts").resolve()
        logging.warning(f"Falling back to prompt path relative to script: {prompt_base_path}")
    except Exception as e:
        logging.error(f"Error resolving prompt path: {e}. Using fallback.")
        script_dir = Path(__file__).parent.resolve()
        prompt_base_path = (script_dir / "../utils/prompts").resolve()

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
        max_llm_attempts = 3 # Reduce retries for faster failure if needed

        while not reward_function_code and llm_attempts < max_llm_attempts:
             llm_attempts += 1
             logging.info(f"{iteration_str}: LLM Attempt {llm_attempts}/{max_llm_attempts}")
             feedback_for_llm = current_feedback_content # Use feedback from previous iteration

             # Call get_llm_response (already updated with correct prompt logic)
             reward_function_code, conversation_text, user_prompt_for_llm = get_llm_response(
                 llm_provider=llm_provider, model=cfg.llm.model,
                 system_prompt_path=system_prompt_path, user_prompt_path=user_prompt_path,
                 code_tip_path=code_tip_path, cfg=cfg,
                 feedback_content=feedback_for_llm,
                 previous_reward_code=previous_reward_code
             )

             # Check for API errors
             if f"Error calling {llm_provider} API" in conversation_text:
                  logging.error(f"{iteration_str}: LLM API call failed. Aborting iteration.")
                  # Log failure details concisely...
                  with open(conversation_file, "a", encoding="utf-8") as conv_file:
                       conv_file.write(f"## {iteration_str}: Failed - {llm_provider.upper()} API Error\n**Error:** `{conversation_text}`\n**Attempted Prompt:**\n```\n{user_prompt_for_llm}\n```\n---\n\n")
                  current_feedback_content = FEEDBACK_ANALYSIS_PROMPT + f"\n**Status:** Failed - {llm_provider.upper()} API Error.\n**Error:** {conversation_text}\nPlease try again."
                        all_results_summary.append(f"Iter {i+1}: Failed - {llm_provider.upper()} API Error")
                  reward_function_code = "" # Ensure skip
                  break # Exit attempt loop

             # Check for empty/invalid code extraction
             elif not reward_function_code:
                  logging.warning(f"{iteration_str}: Could not extract valid Python code from LLM response (Attempt {llm_attempts}).")
                  if llm_attempts < max_llm_attempts:
                       logging.info("Retrying LLM call.")
                       retry_feedback = "\n\n[System Retry Feedback]: Previous response lacked valid code. Ensure ```python ... ``` tags and correct signature `def custom_reward_function(obs, action, done, env):`.\n"
                       if retry_feedback not in current_feedback_content: # Avoid duplicate messages
                            current_feedback_content += retry_feedback
                       time.sleep(2)
                  else:
                       logging.error(f"{iteration_str}: Failed to get valid reward code from LLM after {max_llm_attempts} attempts.")
                       # Log failure details...
                       with open(conversation_file, "a", encoding="utf-8") as conv_file:
                           conv_file.write(f"## {iteration_str}: Failed - No Valid Code from LLM\n**Final LLM Response:**\n```\n{conversation_text}\n```\n**Attempted Prompt:**\n```\n{user_prompt_for_llm}\n```\n---\n\n")
                       current_feedback_content = FEEDBACK_ANALYSIS_PROMPT + f"\n**Status:** Failed - No valid code after {max_llm_attempts} attempts.\n**Last LLM Response:**\n{conversation_text}\nPlease try again."
                       all_results_summary.append(f"Iter {i+1}: Failed - No valid code from LLM")
                       reward_function_code = "" # Ensure skip
                       break # Exit attempt loop
             else:
                  logging.info(f"{iteration_str}: Successfully generated and extracted reward code.")
                  break # Exit attempt loop

        # --- Check if LLM attempts failed ---
        if not reward_function_code:
             logging.warning(f"{iteration_str}: Skipping Training/Evaluation due to failure in LLM response generation.")
             # current_feedback_content is already set from the failure logic above
             # previous_reward_code remains unchanged
             continue # Skip to the next iteration

        # --- Save generated code ---
        logging.info(f"{iteration_str}: Saving generated reward function to: {reward_py_path}")
        try:
            with open(reward_py_path, "w", encoding="utf-8") as f:
                f.write("import numpy as np\n")
                # Make sure generated code doesn't have duplicate imports if LLM adds them
                cleaned_code = re.sub(r"^\s*import\s+numpy\s+as\s+np\s*\n?", "", reward_function_code, flags=re.MULTILINE)
                f.write(cleaned_code)
            # This generated code becomes the context for the *next* iteration's feedback
            previous_reward_code = cleaned_code
        except IOError as e:
             logging.error(f"{iteration_str}: Failed to write reward function file: {e}. Skipping training/evaluation.")
             # Log failure...
             with open(conversation_file, "a", encoding="utf-8") as conv_file:
                  conv_file.write(f"## {iteration_str}: Failed - Cannot Save Code\n**Error:** `{str(e)}`\n**Generated Code (Unsaved):**\n```python\n{reward_function_code}\n```\n---\n\n")
             current_feedback_content = FEEDBACK_ANALYSIS_PROMPT + f"\n**Status:** Internal error: Failed to save the previously generated code.\n**Error:** {e}\nPlease regenerate the reward function."
             # previous_reward_code remains unchanged
             all_results_summary.append(f"Iter {i+1}: Failed - Cannot save code")
             continue # Skip to next iteration

        # --- Train and Evaluate (uses the updated train_and_evaluate function) ---
        logging.info(f"{iteration_str}: Starting training and evaluation...")
        avg_eval_reward, final_train_reward, error_message, saved_model_path = train_and_evaluate(
            cfg, reward_py_path, tb_log_dir_iter, current_run_models_dir, i
        )

        # --- Process results and generate feedback for NEXT iteration ---
        eval_reward_str = f"{avg_eval_reward:.2f}" if avg_eval_reward is not None else "N/A"
        train_reward_str = f"{final_train_reward:.2f}" if final_train_reward is not None else "N/A"
        status = "Success" # Default status

        if error_message:
            # Prioritize specific error messages
            if "returned non-numeric value" in error_message:
                 status = "Failed - Invalid Reward Type"
             current_status_message = (
                 f"**Status:** {status}\n"
                     f"**Error:** {error_message}\n\n"
                     f"The reward function (below) returned an invalid (non-numeric) value during {'training' if 'Training failed' in error_message else 'evaluation'}. "
                     "Please correct the function to ensure it always returns a single float or integer."
                 )
            elif "Training failed" in error_message or "Evaluation failed" in error_message or "Iteration Error" in error_message:
                 status = "Failed - Training/Evaluation Error"
                 current_status_message = (
                     f"**Status:** {status}\n"
                     f"**Error:** {error_message}\n\n"
                     "An error occurred during training or evaluation. Review the error message and the reward function (below) for potential issues."
                 )
            else: # General error
                 status = "Failed - Unknown Error"
                 current_status_message = (
                     f"**Status:** {status}\n"
                     f"**Error Details:**\n```\n{error_message}\n```\n\n"
                     "An unexpected error occurred. Please review the reward function (below)."
                 )
            logging.error(f"{iteration_str}: {status}. Error: {error_message}")
        elif avg_eval_reward is None and final_train_reward is None:
             status = "Failed - No Results"
             current_status_message = (
                 f"**Status:** {status}\n"
                 f"**Reason:** Could not obtain evaluation reward or final training reward (check logs for errors during TB reading or evaluation episode failures).\n\n"
                 "Review the reward function (below) and the simulation behavior."
             )
             logging.warning(f"{iteration_str}: {status}. Check logs for TB/eval issues.")
        elif avg_eval_reward is None:
             status = "Partial Success - Eval Failed/Inconclusive"
             current_status_message = (
                 f"**Status:** {status}\n"
                 f"**Training Result:** Final Mean Training Reward (TB): `{train_reward_str}`\n"
                 f"**Evaluation Result:** Failed or no episodes completed successfully.\n\n"
                 "Training progressed, but evaluation failed. Review reward function (below) for issues causing evaluation failures (e.g., instability, unreachable goals)."
             )
             logging.warning(f"{iteration_str}: {status}. Eval reward N/A.")
        else:
             status = "Success"
             current_status_message = (
                 f"**Status:** {status}\n"
                 f"**Results:**\n"
                 f"- Average Evaluation Reward: `{eval_reward_str}`\n"
                 f"- Final Mean Training Reward (TB): `{train_reward_str}`\n\n"
                 f"Based on these results and the task goal ('{cfg.env.task}'), analyze the reward function code (below) and suggest improvements for the next iteration."
             )
             logging.info(f"{iteration_str}: {status}. Eval: {eval_reward_str}, Train: {train_reward_str}")

        # Construct the full feedback content for the *next* LLM call
        # Use the `previous_reward_code` which holds the code from *this* iteration
        current_feedback_content = FEEDBACK_ANALYSIS_PROMPT + \
                                   f"\n{current_status_message}" + \
                                   f"\n\n**Previous Reward Function Code (Iteration {i+1}):**\n```python\n{previous_reward_code}\n```" # Use the code that was just run

        # --- Log Iteration Summary ---
        logging.info(f"{iteration_str}: Logging results to {conversation_file}")
        with open(conversation_file, "a", encoding="utf-8") as conv_file:
             conv_file.write(f"## {iteration_str}\n\n")
             conv_file.write(f"**Status:** {status}\n\n")
             # Log the prompt that *led* to this iteration's code
             conv_file.write(f"**User Prompt to LLM:**\n*Note: Contains feedback from Iteration {i}*\n```\n{user_prompt_for_llm}\n```\n\n")
             # Log the raw LLM response
             conv_file.write(f"**LLM Response:**\n```\n{conversation_text}\n```\n\n")
             # Log the *cleaned* code that was actually saved and run
             run_code_for_log = previous_reward_code if status != "Failed - Cannot Save Code" else reward_function_code # Show unsaved if saving failed
             conv_file.write(f"**Generated/Executed Reward Code (saved as {os.path.basename(reward_py_path)}):**\n```python\n{run_code_for_log}\n```\n\n")
             conv_file.write("**Training & Evaluation Results:**\n")
             relative_tb_path = os.path.relpath(tb_log_dir_iter, current_run_results_dir)
             conv_file.write(f"- TensorBoard Log: `{relative_tb_path}`\n")
             if saved_model_path:
                 workspace_rel_model_path = os.path.relpath(saved_model_path, os.getcwd())
                 conv_file.write(f"- Saved Model: `{workspace_rel_model_path}`\n")
             else: conv_file.write("- Saved Model: Failed or Skipped\n")
             conv_file.write(f"- Avg Eval Reward: `{eval_reward_str}`\n")
             conv_file.write(f"- Final Train Reward (TB): `{train_reward_str}`\n")
             if error_message:
                 # Log concise error here, full details are in the feedback section
                 concise_error = error_message.split('\\n')[0] # Get first line
                 conv_file.write(f"- Error Encountered: Yes (`{concise_error}`)\n")
             conv_file.write("\n")
             # Log the feedback that will be used for the *next* iteration
             conv_file.write(f"**Feedback Content Generated for Next Iteration ({i+2}):**\n```\n{current_feedback_content}\n```\n\n")
             conv_file.write("---\n\n")

        # --- Update Summary ---
        eval_reward_summary = f"{avg_eval_reward:.2f}" if avg_eval_reward is not None else "N/A"
        train_reward_summary = f"{final_train_reward:.2f}" if final_train_reward is not None else "N/A"
        summary_error = "Yes" if error_message else "No"
        summary_model = "Yes" if saved_model_path else "No"
        summary = f"Iter {i+1}: Status='{status}', Eval={eval_reward_summary}, Train(TB)={train_reward_summary}, Model Saved={summary_model}, Error={summary_error}"
        all_results_summary.append(summary)
        logging.info(f"========== Finished {iteration_str} ==========\n")

    logging.info("Iterative reward function generation complete.")
    logging.info("Final Results Summary:")
    if not all_results_summary: logging.info("  No iterations completed fully.")
    else:
         for result_line in all_results_summary: logging.info(f"  {result_line}")
    logging.info(f"Detailed logs for run {run_timestamp} (env: {env_name_str}) saved in: {current_run_results_dir}")
    logging.info(f"Models for run {run_timestamp} (env: {env_name_str}) saved in: {current_run_models_dir}")

if __name__ == "__main__" :
             main()