import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
from openai import OpenAI
import traceback

import re
import subprocess
from pathlib import Path
import shutil
import time 
from utils import create_task 
import importlib
from omegaconf import DictConfig

from utils import Conversation 
from utils.file import *
from utils.create_task import create_task
from model_test_env import train

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
EUREKA_ROOT_DIR = os.getcwd()
OUTPUTS_DIR = f"{EUREKA_ROOT_DIR}/results"

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    suffix = cfg.suffix
    task=cfg.task
    task_description= cfg.task_description
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)


    #workspace_dir = str(os.getcwd())
    #workspace_dir = Path.cwd()
    workspace_dir = '/home/bechir/RewGen4Robotic'
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")


    env_name = "walker2d_v5"

    test_obs_file_path = f'model_test_env/{env_name}_obs.py'
    task_obs_file = f'{workspace_dir}/{test_obs_file_path}'
    task_obs_file = f"{workspace_dir}/model_test_env/{env_name}_obs.py"
    logging.info(f"task_obs_file : {task_obs_file}")
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_file = f'{workspace_dir}/model_test_env/{env_name}.py'
    task_obs_code_string = file_to_string(task_obs_file)
    task_code_string = file_to_string(task_file)
    output_file = f"{OUTPUTS_DIR}/tasks/{env_name}{suffix.lower()}.py"

    #loading the prompts
    prompt_path= f"{workspace_dir}/utils/prompts"
    system_prompt = file_to_string(f'{workspace_dir}/utils/walker_prompts/system_prompt.txt')
    user_prompt = file_to_string(f'{prompt_path}/user_prompt.txt')
    reward_signature = file_to_string(f'{prompt_path}/reward_signature.txt')
    code_output_tip = file_to_string(f'{prompt_path}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_path}/code_feedback.txt')
    policy_feedback = file_to_string(f'{prompt_path}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_path}/execution_error_feedback.txt')

    system_prompt = system_prompt.format(task_reward_signature_string=reward_signature) + code_output_tip
    user_prompt = user_prompt.format(task_description=task_description, task_obs_code_string=task_file)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    logging.info( f"Generating samples with {cfg.model}")

    task_code_string = task_code_string.replace(task, task+suffix)

    max_rewards = list()
    best_code_paths = list()


    #create_task(OUTPUTS_DIR, cfg.env.task, cfg.env.env_name, suffix)
    DUMMY_FAILURE = -10000.
    execute_rates = []
    best_code_paths = []
    max_reward_overall = DUMMY_FAILURE
    max_reward_code_path = None 

    for iter in range(cfg.iteration) :
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = 4

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        while True :
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = client.chat.completions.create(model=model,
                    messages=messages,
                    temperature=cfg.temperature,
                    n=chunk_size)
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur.choices)
            prompt_tokens = response_cur.usage.prompt_tokens
            total_completion_token += response_cur.usage.completion_tokens
            total_token += response_cur.usage.total_tokens

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

        code_runs = [] 
        rl_runs = []
        mean_reward_per_sample = []
        # logging.info(responses[0])
        code_feedbacks = []
        contents = []
        code_paths = []
        for response_id in range(cfg.sample):

                response_cur = responses[response_id].message.content
                logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

                # extract code block from gpt response
                code_string = extract_code(response_cur)

                # Remove unnecessary imports
                lines = code_string.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("def "):
                        code_string = "\n".join(lines[i:])

                # Add the Eureka Reward Signature to the environment code
                try:
                    gpt_reward_signature, input_lst = get_function_signature2(code_string)
                except Exception as e:
                    logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
                    continue

                code_runs.append(code_string)

                reward_signature = [
                f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
                ]

                indent = " " * 8
                reward_signature = "\n".join([indent + line for line in reward_signature])
                if "def compute_reward(self, x_velocity: float,observation, action):" in task_code_string:
                    task_code_string_iter = task_code_string.replace("def compute_reward(self, x_velocity: float,observation, action):", "def compute_reward(self, x_velocity: float,observation, action):\n" + reward_signature)
                elif "def compute_reward(self)" in task_code_string:
                    task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
                else:
                    raise NotImplementedError

                # Save the new environment code when the output contains valid code string!
                with open(output_file,'w') as file :
                    file.writelines(task_code_string_iter)
                    file.writelines("import math"+'\n')
                    file.writelines(code_string+ '\n')

                with open(f"{OUTPUTS_DIR}/tasks/env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                    file.writelines(code_string + '\n')

                # Copy the generated environment code to hydra output directory for bookkeeping
                full_file_path = f"{OUTPUTS_DIR}/tasks/env_iter{iter}_response{response_id}.py"
                shutil.copy(output_file, full_file_path)

                env_iter_file = f'results/tasks/env_iter{iter}_response{response_id}.py'
                
                # convert file path to module name
                module_name = env_iter_file.replace('/','.').replace('.py','').lstrip()
                env_module = importlib.import_module(module_name)

                # Instantiate environment
                env = env_module.Walker2dEnv()
                env.reset()
                code_paths.append(env_iter_file) 
                rl_runs.append(env_iter_file) 
                traceback_msg = ""
                try :
                    # Training the environment
                    trainer = train.TrainingManager(env=env,root_dir=workspace_dir,iter=iter,reponse_id=response_id)
                    model = trainer.run()
                    
                except :
                    content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                    content += code_output_tip  
                    contents.append(content)
                    mean_reward_per_sample.append(DUMMY_FAILURE)
                    traceback_msg=traceback.print_exc()
                    continue

                content = ""
                if traceback_msg == "":
                    # extracting rward stats from tensorboard 
                    tensorboard_logs = load_tensorboard_logs(trainer.get_logs_path()) 
                    ep_reward_mean = np.array(tensorboard_logs["rollout/ep_rew_mean"]).mean()
                    ep_length_mean = np.array(tensorboard_logs["rollout/ep_len_mean"]).mean()

                    logging.info( f"iteration [{iter}], sample number [{response_id}]// episode reward mean : {ep_reward_mean}")
                    logging.info( f"iteration [{iter}], sample number [{response_id}]// episode length mean : {ep_length_mean}")

                    mean_reward_per_sample.append(ep_reward_mean) 
                    max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                    epoch_freq = max(int(max_iterations // 10), 1)        
                    content += policy_feedback.format(epoch_freq=epoch_freq)
                    for metric in tensorboard_logs:
                            metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                            metric_cur_max = max(tensorboard_logs[metric])
                            metric_cur_mean = np.array(tensorboard_logs[metric]).mean()
                            metric_cur_min = min(tensorboard_logs[metric])
                            
                            metric_name = "task_score"
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                    code_feedbacks.append(code_feedback)
                    content += code_feedback
                else :
                    # Otherwise, provide execution traceback error feedback
                    content+= execution_error_feedback(traceback_msg= traceback_msg)

                    
                content += code_output_tip
                contents.append(content)
        
        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(mean_reward_per_sample))
        best_content = contents[best_sample_idx]
        max_reward = mean_reward_per_sample[best_sample_idx]

        if max_reward > max_reward_overall :
            max_reward_overall = max_reward
            max_reward_code_path = code_paths[best_sample_idx]

        max_rewards.append(max_reward)
        best_code_paths.append(max_reward_code_path)
        logging.info(f"Iteration {iter}: Max Success: {max_reward}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx].message.content + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        messages += [{"role": "assistant", "content": responses[best_sample_idx].message.content}]
        messages += [{"role": "user", "content": best_content}]

        # Save dictionary as JSON file
        with open('messages.json', 'w') as file:
            json.dump(messages, file, indent=4)
    
    # Evaluate the best reward code many times
    if max_reward_code_path is None: 
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()

    logging.info(f"Task: {task}, Max Training Success {max_reward_overall}, Best Reward Code Path: {max_reward_code_path}")
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    shutil.copy(max_reward_code_path, output_file)

    eval_runs = []

    for i in range(cfg.num_eval):
        # convert file path to module name
        module_name = max_reward_code_path.replace('/','.').replace('.py','').lstrip()
        env_module = importlib.import_module(module_name)

        # Instantiate environment
        env = env_module.Walker2dEnv()
        env.reset()
        # Training the environment
        trainer_max_reward = train.TrainingManager(env=env,root_dir=workspace_dir,iter=iter,reponse_id=response_id)
        model_max_reward = trainer_max_reward.run()
        # extracting rward stats from tensorboard 
        tensorboard_logs = load_tensorboard_logs(trainer.get_logs_path()) 
        eval_runs.append(tensorboard_logs)

    reward_code_final_rewards = list()
    for i, tb_logs in enumerate(eval_runs) :
        max_reward = max(tb_logs["cumulative_reward"]) 
        reward_code_final_rewards.append(max_reward)

    logging.info(f"Final Success Mean: {np.mean(reward_code_final_rewards)}, Std: {np.std(reward_code_final_rewards)}, Raw: {reward_code_final_rewards}")
    np.savez('final_eval.npz', reward_code_final_successes=reward_code_final_rewards)

if __name__ == "__main__":
    main()
