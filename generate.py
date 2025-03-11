# scripts/reward_generator.py
import os
import openai
import logging
import hydra
from omegaconf import DictConfig
from utils.file_to_string import file_to_string
from utils.extracct_code import extract_code_from_response
from envs.antmaze import make_custom_antmaze
from stable_baselines3 import PPO

def get_llm_response(model, system_prompt_path, user_prompt_path, code_tip_path, task_description, feedback,  iteration):
    system_prompt = file_to_string(system_prompt_path)
    user_prompt = file_to_string(user_prompt_path)
    code_tip = file_to_string(code_tip_path)
    
    # Load feedback guidelines
    feedback_guidelines_path = "/home/ken2/PCD/utils/prompts/feedback_guidelines.txt"
    guidelines = file_to_string(feedback_guidelines_path)

    # Combine feedback with guidelines
    combined_feedback = feedback + "\n" + guidelines
    
    # Format user prompt with task description and feedback
    user_prompt_formatted = user_prompt.format(task_description=task_description, feedback=combined_feedback)
    combined_prompt = system_prompt + "\n" + code_tip
    
    messages = [
        {"role": "system", "content": combined_prompt},
        {"role": "user", "content": user_prompt_formatted}
    ]
    
    response = openai.chat.completions.create(
        model=model,
        messages=messages
    )
    
    response_text = response.choices[0].message.content  # Correctly extract response text

    try:
        clean_code = extract_code_from_response(response_text)  # Extract only Python code
    except ValueError as e:
        print("Error extracting code:", e)
        clean_code = ""  # Fallback to an empty string

    # Save extracted reward function
    reward_file_path = f"/home/ken2/PCD/results/reward_function_{iteration}.py"
    os.makedirs(os.path.dirname(reward_file_path), exist_ok=True)  # Ensure directory exists

    with open(reward_file_path, "w") as f:
        f.write(clean_code)

    print(f" Reward function for iteration {iteration} saved to {reward_file_path}")

    return response_text  # Return full response in case you need it

def train_and_evaluate(py_reward_path, total_timesteps, eval_episodes):
    # Create environment with the custom reward function loaded from py_reward_path
    env = make_custom_antmaze(py_reward_path)
    model = PPO("MultiInputPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    
    total_reward = 0.0
    for ep in range(eval_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_reward += ep_reward
    avg_reward = total_reward / eval_episodes
    return avg_reward

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    base_results_folder = os.path.abspath("/home/ken2/PCD/results")
    os.makedirs(base_results_folder, exist_ok=True)
    
    # Set OpenAI API key from environment variables
    openai.api_key = os.getenv("OPEN_AI_KEY")
    
    # Define paths to prompt files (assumed to be in utils/prompts)
    prompt_path = os.path.abspath("/home/ken2/PCD/utils/prompts")
    system_prompt_path = os.path.join(prompt_path, "system_prompt.txt")
    user_prompt_path = os.path.join(prompt_path, "user_prompt.txt")
    code_tip_path = os.path.join(prompt_path, "code_output_tip.txt")
    
    feedback = ""  # Initial feedback can be empty
    for i in range(cfg.iterations):
        logging.info(f"=== Iteration {i+1}/{cfg.iterations} ===")
        
        # Generate reward function code using the LLM
        reward_function_response = get_llm_response(
            model=cfg.model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            code_tip_path=code_tip_path,
            task_description=cfg.task_description,
            feedback=feedback ,
            iteration=i
        )
        
        # Save generated reward function as .txt and .py
        reward_txt_path = os.path.join(base_results_folder, f"reward_function_{i}.txt")
        reward_py_path = os.path.join(base_results_folder, f"reward_function_{i}.py")
        with open(reward_txt_path, "w") as f:
            f.write(reward_function_response)

    # Extract only the Python code
        try:
            reward_function_code = extract_code_from_response(reward_function_response)
        except ValueError as e:
            print(f"Error extracting Python code: {e}")
            reward_function_code = ""  # Fallback in case extraction fails

        # Save extracted Python code to .py
        reward_py_path = os.path.join(base_results_folder, f"reward_function_{i}.py")
        with open(reward_py_path, "w") as f:
            f.write(reward_function_code)
            
            # Train and evaluate using the new reward function
        avg_reward = train_and_evaluate(
                py_reward_path=reward_py_path,
                total_timesteps=cfg.total_timesteps,
                eval_episodes=cfg.eval_episodes
            )
            
            # Save evaluation result
        eval_result_path = os.path.join(base_results_folder, f"eval_results_{i}.txt")
        with open(eval_result_path, "w") as f:
                f.write(f"Average reward: {avg_reward}\n")
            
            # Generate dynamic feedback for next iteration
        dynamic_feedback = f"Iteration {i+1}: Average reward = {avg_reward:.2f}. "
        if avg_reward < cfg.target_reward:
                dynamic_feedback += "Reward function performance is below target. It needs to better encourage efficient movement toward the goal and penalize unnecessary actions."
        else:
                dynamic_feedback += "Reward function performs well."
            
            # Save dynamic feedback to a file for record
        feedback_path = os.path.join(base_results_folder, f"dynamic_feedback_{i+1}.txt")
        with open(feedback_path, "w") as f:
                f.write(dynamic_feedback)
            
            # Use dynamic feedback for the next iteration
        feedback = dynamic_feedback
        logging.info(f"Iteration {i+1} complete. Avg reward: {avg_reward:.2f}")
        
        logging.info("Iterative reward function generation complete.")
        
        # Optionally run a final training step using a selected reward function.
        if cfg.final_training:
            logging.info("Starting final training using the selected reward function.")
            reward_py_path_final = os.path.join(base_results_folder, cfg.selected_reward_function)
            # Create environment with the selected reward function
            env = make_custom_antmaze(reward_py_path_final)
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=cfg.total_timesteps)
            model_save_path = os.path.join(base_results_folder, "trained_model.zip")
            model.save(model_save_path)
            logging.info(f"Final training complete. Model saved at {model_save_path}.")

if __name__ == "__main__":
    main()
