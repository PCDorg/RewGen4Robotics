import os
import openai
import logging
import math
import hydra
from omegaconf import DictConfig
from datetime import date

from utils.file_to_string import file_to_string
from utils.extracct_code import extract_code_from_response
from envs.fetchReach import make_custom_fetch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import traceback
import time


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
            logging.warning(f"No scalar data found in {tb_log_dir}. Check if PPO logged correctly and training ran.")
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


def get_llm_response(model, system_prompt_path, user_prompt_path, code_tip_path, cfg, feedback_content, previous_reward_code):
    logging.info("Generating new reward function with LLM...")
    system_prompt = file_to_string(system_prompt_path)
    user_prompt_template = file_to_string(user_prompt_path)
    code_tip = file_to_string(code_tip_path)

    formatted_task_desc = (
        f"Task Goal: {cfg.env.task}\n"
        f"Environment Name: {cfg.env.env_name}\n"
        f"Task Description: {cfg.env.task_description}\n"
        f"Target Timesteps: {cfg.env.timestamps}" # Renamed for clarity
    )


    user_prompt_formatted = user_prompt_template.format(
        task_description=formatted_task_desc,
        feedback=feedback_content,
        env_details=cfg.env  # Changed this line
    )
    combined_system_prompt = system_prompt + "\n" + code_tip

    messages = [
        {"role": "system", "content": combined_system_prompt},
        {"role": "user", "content": user_prompt_formatted}
    ]

    logging.info("Sending request to OpenAI API...")
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            #temperature=cfg.get("temperature", 0.7)
        )
        response_text = response.choices[0].message.content
        logging.info("Received response from LLM.")
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return "", f"Error calling OpenAI API: {e}", user_prompt_formatted

    try:
        clean_code = extract_code_from_response(response_text)
        if not clean_code:
             logging.warning("LLM response did not contain a valid Python code block.")
    except ValueError as e:
        logging.error(f"Error extracting code: {e}")
        clean_code = ""


    return clean_code, response_text, user_prompt_formatted



def train_and_evaluate(py_reward_path, tb_log_dir_iter, results_folder_iter, total_timesteps, eval_episodes, iteration):
    """Trains the PPO agent, saves the model, and evaluates it."""
    final_train_reward = None
    avg_eval_reward = None
    error_message = None
    model_save_path = None

    try:
        logging.info(f"Iteration {iteration+1}: Creating environment with reward from: {py_reward_path}")

        try:

             env = make_custom_fetch(py_reward_path)
        except NameError:
             logging.error("`make_custom_fetch` function not found. Ensure it's defined or imported.")
             raise
        except Exception as env_e:
             logging.error(f"Error creating environment with {py_reward_path}: {env_e}")
             raise


        env = Monitor(env)

        logging.info(f"Iteration {iteration+1}: Initializing PPO model. Logging TensorBoard to: {tb_log_dir_iter}")
        model = PPO("MultiInputPolicy", env, verbose=0, tensorboard_log=tb_log_dir_iter)

        logging.info(f"Iteration {iteration+1}: Starting training for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps, tb_log_name="PPO", reset_num_timesteps=False)
        logging.info(f"Iteration {iteration+1}: Training complete.")

        model_save_path = os.path.join(results_folder_iter, f"ppo_model_{iteration}.zip")
        logging.info(f"Iteration {iteration+1}: Saving trained model to: {model_save_path}")
        try:
            model.save(model_save_path)
            logging.info(f"Iteration {iteration+1}: Model saved successfully.")
        except Exception as save_e:
            logging.error(f"Iteration {iteration+1}: Failed to save model: {save_e}")
            model_save_path = None


        potential_log_path = os.path.join(tb_log_dir_iter, "PPO_1")
        if not os.path.exists(potential_log_path):
             logging.warning(f"Expected PPO log directory '{potential_log_path}' not found. Reading from parent: {tb_log_dir_iter}")
             ppo_log_path = tb_log_dir_iter
        else:
             ppo_log_path = potential_log_path


        logging.info(f"Iteration {iteration+1}: Reading scalars from TensorBoard log: {ppo_log_path}")
        scalar_data = get_all_scalars(ppo_log_path)
        final_train_reward = get_final_scalar_value(scalar_data, 'rollout/ep_rew_mean')

        if final_train_reward is not None:
            logging.info(f"Iteration {iteration+1}: Final training mean episode reward (from TB): {final_train_reward:.2f}")
        else:
            logging.warning(f"Iteration {iteration+1}: Could not extract 'rollout/ep_rew_mean' from TensorBoard.")


        logging.info(f"Iteration {iteration+1}: Starting evaluation for {eval_episodes} episodes...")
        total_reward = 0.0
        eval_env = env
        num_successful_evals = 0

        for ep in range(eval_episodes):
            obs, info = eval_env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated
            total_reward += ep_reward
            num_successful_evals += 1

        if num_successful_evals > 0:
             avg_eval_reward = total_reward / num_successful_evals
             logging.info(f"Iteration {iteration+1}: Evaluation complete. Average reward over {num_successful_evals} episodes: {avg_eval_reward:.2f}")
        else:
             logging.error(f"Iteration {iteration+1}: No evaluation episodes completed successfully.")
             avg_eval_reward = None
        try:
             env.close()
        except Exception as close_e:
             logging.warning(f"Error closing environment: {close_e}")

    except Exception as e:
        logging.error(f"Error during training/evaluation in Iteration {iteration+1}: {e}")
        logging.error(traceback.format_exc())
        error_message = f"Error during training/evaluation: {e}\nTraceback:\n{traceback.format_exc()}"
        avg_eval_reward = avg_eval_reward if avg_eval_reward is not None else None
        final_train_reward = final_train_reward if final_train_reward is not None else None


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

@hydra.main(config_path="/home/ken2/PCD/cfg", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    results_folder_base = "/home/ken2/PCD/results/reach"
    os.makedirs(results_folder_base, exist_ok=True)

    current_date = date.today().strftime("%Y-%m-%d")
    iteration_results_folder = os.path.join(results_folder_base, current_date)
    os.makedirs(iteration_results_folder, exist_ok=True)

    logging.info(f"Using results directory: {iteration_results_folder}")

    tb_log_dir_base = os.path.join(iteration_results_folder, "tensorboard_logs")
    os.makedirs(tb_log_dir_base, exist_ok=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_api_key


    prompt_base_path = os.path.abspath("/home/ken2/PCD/utils/prompts") # Absolute path
    system_prompt_path = os.path.join(prompt_base_path, "system_prompt.txt")
    user_prompt_path = os.path.join(prompt_base_path, "user_prompt.txt") # This is the template file
    code_tip_path = os.path.join(prompt_base_path, "code_output_tip.txt")

    for f_path in [system_prompt_path, user_prompt_path, code_tip_path]:
         if not os.path.exists(f_path):
              logging.error(f"Required prompt file not found: {f_path}")
              return

    conversation_file = os.path.join(iteration_results_folder, "conversation_history.md") # Use Markdown

    # --- Iteration Loop ---
    current_feedback_content = "This is the first iteration. Please generate an initial reward function based on the task description." # Initial feedback
    previous_reward_code = "None (first iteration)"
    all_results_summary = []

    for i in range(cfg.iterations):
        iteration_str = f"Iteration {i+1}/{cfg.iterations}"
        logging.info(f"========== Starting {iteration_str} ==========")

        reward_py_path = os.path.join(iteration_results_folder, f"reward_function_{i}.py")
        tb_log_dir_iter = os.path.join(tb_log_dir_base, f"iteration_{i}")
        os.makedirs(tb_log_dir_iter, exist_ok=True) # Ensure TB dir exists for this iter

        reward_function_code = ""
        conversation_text = ""
        user_prompt_for_llm = ""
        llm_attempts = 0
        max_llm_attempts = 10

        while not reward_function_code and llm_attempts < max_llm_attempts:
             llm_attempts += 1
             logging.info(f"{iteration_str}: LLM Attempt {llm_attempts}/{max_llm_attempts}")
             reward_function_code, conversation_text, user_prompt_for_llm = get_llm_response(
                 model=cfg.model,
                 system_prompt_path=system_prompt_path,
                 user_prompt_path=user_prompt_path,
                 code_tip_path=code_tip_path,
                 cfg=cfg,
                 feedback_content=current_feedback_content,
                 previous_reward_code=previous_reward_code
             )

             if "Error calling OpenAI API" in conversation_text:
                  logging.error(f"{iteration_str}: LLM API call failed. Aborting iteration.")
                  with open(conversation_file, "a", encoding="utf-8") as conv_file:
                       conv_file.write(f"## {iteration_str}\n\n")
                       conv_file.write("**Status:** Failed - OpenAI API Error\n\n")
                       conv_file.write("**Error:**\n```\n{conversation_text}\n```\n\n")
                       conv_file.write("**Attempted User Prompt to LLM:**\n```\n{user_prompt_for_llm}\n```\n\n")
                       conv_file.write("---\n\n")
                  all_results_summary.append(f"Iter {i+1}: Failed - OpenAI API Error")
                  break

             elif not reward_function_code:
                  logging.warning(f"{iteration_str}: Could not extract valid Python code from LLM response (Attempt {llm_attempts}).")
                  if llm_attempts < max_llm_attempts:
                       logging.info("Retrying LLM call.")

                       current_feedback_content += "\n\n[System Retry Feedback]: The previous response did not contain a valid Python code block. Please ensure the reward function code is clearly marked within ```python ... ``` tags and follows the required signature."

                       time.sleep(2)
                  else:
                       logging.error(f"{iteration_str}: Failed to get valid reward code from LLM after {max_llm_attempts} attempts.")

                       with open(conversation_file, "a", encoding="utf-8") as conv_file:
                            conv_file.write(f"## {iteration_str}\n\n")
                            conv_file.write(f"**Status:** Failed - Could not generate valid reward code after {max_llm_attempts} attempts\n\n")
                            conv_file.write("**Final LLM Response:**\n```\n{conversation_text}\n```\n\n")
                            conv_file.write("**Attempted User Prompt to LLM:**\n```\n{user_prompt_for_llm}\n```\n\n")
                            conv_file.write("---\n\n")
                       all_results_summary.append(f"Iter {i+1}: Failed - No valid code from LLM")

             else:
                  logging.info(f"{iteration_str}: Successfully generated and extracted reward code.")




        if not reward_function_code:
             logging.warning(f"{iteration_str}: Skipping Training/Evaluation due to failure in LLM response generation.")

             current_feedback_content = FEEDBACK_ANALYSIS_PROMPT + (
                  f"\n**Status:** Failed to generate usable code in the previous attempt after {llm_attempts} tries.\n"
                  f"**Last LLM Response:**\n{conversation_text}\n\n"
                  "Please try again, carefully following the instructions and ensuring the code is correctly formatted."
             )
             previous_reward_code = "None (Code generation failed)"


        logging.info(f"{iteration_str}: Saving generated reward function to: {reward_py_path}")
        try:
            with open(reward_py_path, "w", encoding="utf-8") as f:
                f.write(reward_function_code)
            previous_reward_code = reward_function_code
        except IOError as e:
             logging.error(f"{iteration_str}: Failed to write reward function file: {e}. Skipping training/evaluation.")
             with open(conversation_file, "a", encoding="utf-8") as conv_file:
                 conv_file.write(f"## {iteration_str}\n\n")
                 conv_file.write("**Status:** Failed - Could not save reward code\n\n")
                 conv_file.write("**Error:**\n```\n{str(e)}\n```\n\n")
                 conv_file.write("**Generated Code (Unsaved):**\n```python\n{reward_function_code}\n```\n\n")
                 conv_file.write("---\n\n")
             current_feedback_content = FEEDBACK_ANALYSIS_PROMPT + (
                 f"\n**Status:** Internal error: Failed to save the previously generated code.\n"
                 f"**Error:** {e}\n"
                 "Please regenerate the reward function."
             )
             previous_reward_code = "None (Code saving failed)"
             all_results_summary.append(f"Iter {i+1}: Failed - Cannot save code")
             continue


        logging.info(f"{iteration_str}: Starting training and evaluation...")
        avg_eval_reward, final_train_reward, error_message, saved_model_path = train_and_evaluate(
            py_reward_path=reward_py_path,
            tb_log_dir_iter=tb_log_dir_iter,
            results_folder_iter=iteration_results_folder,
            total_timesteps=cfg.total_timesteps,
            eval_episodes=cfg.eval_episodes,
            iteration=i
        )
        current_status_message = ""
        if error_message:
            status = "Failed - Error during Training/Evaluation"
            logging.warning(f"{iteration_str}: Training/Evaluation failed. Error logged.")
            current_status_message = (
                f"**Status:** {status}\n"
                f"**Error details:**\n```\n{error_message}\n```\n"
                "Please analyze the reward function code (below) and the error message to fix the issue."
            )
        elif avg_eval_reward is None:
             status = "Partial Success - Evaluation Failed"
             logging.warning(f"{iteration_str}: Evaluation did not complete successfully (avg_eval_reward is None).")
             current_status_message = (
                 f"**Status:** {status}\n"
                 f"**Training Result:** Final Mean Training Reward (from TensorBoard `rollout/ep_rew_mean`): `{final_train_reward:.2f if final_train_reward is not None else 'N/A'}`\n"
                 f"**Evaluation Result:** Failed to get an average reward (likely no episodes completed).\n\n"
                 "Review the reward function (below) for issues that might prevent episode completion during evaluation (e.g., infinite loops, unreachable goals)."
             )
        else:
            status = "Success"
            logging.info(f"{iteration_str}: Training/Evaluation successful.")
            current_status_message = (
                f"**Status:** {status}\n"
                f"**Results:**\n"
                f"- Average Evaluation Reward: `{avg_eval_reward:.2f}`\n"
                f"- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): `{final_train_reward:.2f if final_train_reward is not None else 'N/A'}`\n\n"
                f"Based on these results and the task goal ('{cfg.env.task}'), analyze the reward function code (below) and suggest improvements."
            )

        current_feedback_content = FEEDBACK_ANALYSIS_PROMPT + \
                                   f"\n{current_status_message}" + \
                                   f"\n\n**Previous Reward Function Code:**\n```python\n{previous_reward_code}\n```"


        logging.info(f"{iteration_str}: Logging results to {conversation_file}")
        with open(conversation_file, "a", encoding="utf-8") as conv_file:
            conv_file.write(f"## {iteration_str}\n\n")
            conv_file.write(f"**Status:** {status}\n\n")
            conv_file.write("**User Prompt to LLM (leading to this iteration's code):**\n*Note: Contains feedback from the iteration before this one.*\n```\n{user_prompt_for_llm}\n```\n\n")
            conv_file.write("**LLM Response:**\n```\n{conversation_text}\n```\n\n")
            conv_file.write(f"**Generated Reward Code (saved to {os.path.basename(reward_py_path)}):**\n```python\n{reward_function_code}\n```\n\n") # Use the code generated in *this* iter
            conv_file.write("**Training & Evaluation Results:**\n")
            conv_file.write(f"- TensorBoard Log Directory: `{os.path.relpath(tb_log_dir_iter, iteration_results_folder)}`\n") # Relative path
            if saved_model_path:
                conv_file.write(f"- Saved Model: `{os.path.relpath(saved_model_path, iteration_results_folder)}`\n") # Relative path
            else:
                conv_file.write("- Saved Model: Failed or Skipped\n")
            eval_reward_str = f"`{avg_eval_reward:.2f}`" if avg_eval_reward is not None else "`N/A`"
            conv_file.write(f"- Average Evaluation Reward: {eval_reward_str}\n")

            train_reward_str = f"`{final_train_reward:.2f}`" if final_train_reward is not None else "`N/A`"
            conv_file.write(f"- Final Mean Training Reward (TensorBoard `rollout/ep_rew_mean`): {train_reward_str}\n")
            if error_message:
                conv_file.write("- Error Encountered: Yes (See Status/Feedback)\n")
            conv_file.write("\n")
            conv_file.write("**Feedback Content Generated for Next Iteration:**\n```\n{current_feedback_content}\n```\n\n")
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

    logging.info(f"Detailed logs and artifacts saved in: {iteration_results_folder}")


if __name__ == "__main__" :
             main()