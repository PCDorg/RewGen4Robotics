import os
import openai
import logging
import hydra
from omegaconf import DictConfig
from utils.file_to_string import file_to_string
from utils.extracct_code import extract_code_from_response
from envs.antmaze import make_custom_fetch
from stable_baselines3 import PPO

def get_llm_response(model, system_prompt_path, user_prompt_path, code_tip_path, task_description, feedback, previous_reward_path=None):
    logging.info("Generating new reward function...")
    
    system_prompt = file_to_string(system_prompt_path)
    user_prompt = file_to_string(user_prompt_path)
    code_tip = file_to_string(code_tip_path)
    
    if previous_reward_path and os.path.exists(previous_reward_path):
        previous_reward_code = file_to_string(previous_reward_path)
    else:
        previous_reward_code = "None (first iteration)"
    
    guidelines = file_to_string("/home/ken2/PCD/utils/prompts/feedback_guidelines.txt")
    combined_feedback = f"Previous reward function:\n{previous_reward_code}\n\nFeedback:\n{feedback}\n\n{guidelines}"
    
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
    response_text = response.choices[0].message.content
    
    try:
        clean_code = extract_code_from_response(response_text)
    except ValueError as e:
        logging.error(f"Error extracting code: {e}")
        clean_code = ""
    
    return clean_code

def train_and_evaluate(py_reward_path, results_folder,total_timesteps, eval_episodes , iteration ):
    env = make_custom_fetch(py_reward_path)
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

    model_save_path = os.path.join(results_folder, f"model_iteration_{iteration}.zip")
    model.save(model_save_path)
    logging.info(f"Model saved at: {model_save_path}")
        
    
    avg_reward = total_reward / eval_episodes
    logging.info(f"Evaluation complete. Average reward: {avg_reward:.2f}")
    return avg_reward

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    base_results_folder = os.path.abspath("/home/ken2/PCD/results")
    os.makedirs(base_results_folder, exist_ok=True)
    
    openai.api_key = os.getenv("OPEN_AI_KEY")
    
    prompt_path = os.path.abspath("/home/ken2/PCD/utils/prompts")
    system_prompt_path = os.path.join(prompt_path, "system_prompt.txt")
    user_prompt_path = os.path.join(prompt_path, "user_prompt.txt")
    code_tip_path = os.path.join(prompt_path, "code_output_tip.txt")
    
    feedback = ""
    previous_reward_path = None
    
    for i in range(cfg.iterations):
        logging.info(f"Starting iteration {i+1}...")
        
        reward_function_code = get_llm_response(
            model=cfg.model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            code_tip_path=code_tip_path,
            task_description=cfg.task_description,
            feedback=feedback,
            previous_reward_path=previous_reward_path
        )
        
        reward_txt_path = os.path.join(base_results_folder, f"reward_function_{i}.txt")
        reward_py_path = os.path.join(base_results_folder, f"reward_function_{i}.py")
        
        with open(reward_txt_path, "w") as f:
            f.write(reward_function_code)
        
        with open(reward_txt_path, "r") as txt_file, open(reward_py_path, "w") as py_file:
            py_file.write(txt_file.read())
        
        previous_reward_path = reward_py_path
        
        avg_reward = train_and_evaluate(
            py_reward_path=reward_py_path,
            total_timesteps=cfg.total_timesteps,
            eval_episodes=cfg.eval_episodes , 
            iteration=i,
            results_folder=base_results_folder
        )
        
        eval_result_path = os.path.join(base_results_folder, f"eval_results_{i}.txt")
        with open(eval_result_path, "w") as f:
            f.write(f"Average reward: {avg_reward}\n")
        
        dynamic_feedback = f"Iteration {i+1}: Average reward = {avg_reward:.2f}. "
        if avg_reward < cfg.target_reward:
            dynamic_feedback += "Reward function performance is below target. It needs improvement."
        else:
            dynamic_feedback += "Reward function performs well."
        
        feedback_path = os.path.join(base_results_folder, f"dynamic_feedback_{i+1}.txt")
        with open(feedback_path, "w") as f:
            f.write(dynamic_feedback)
        
        feedback = dynamic_feedback
        logging.info(f"Iteration {i+1} complete. Avg reward: {avg_reward:.2f}")
    
    logging.info("Iterative reward function generation complete.")
    
    if cfg.final_training:
        logging.info("Starting final training using the selected reward function.")
        reward_py_path_final = os.path.join(base_results_folder, cfg.selected_reward_function)
        env = make_custom_antmaze(reward_py_path_final)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=cfg.total_timesteps)
        model_save_path = os.path.join(base_results_folder, "trained_model.zip")
        model.save(model_save_path)
        logging.info(f"Final training complete. Model saved at {model_save_path}.")

if __name__ == "__main__":
    main()
