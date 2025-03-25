import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
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
    #code_feedback = file_to_string(f'{prompt_path}/code_feedback.txt')
    #policy_feedback = file_to_string(f'{prompt_path}/policy_feedback.txt')
    #execution_error_feedback = file_to_string(f'{prompt_path}/execution_error_feedback.txt')

    system_prompt = system_prompt.format(task_reward_signature_string=reward_signature) + code_output_tip
    user_prompt = user_prompt.format(task_description=task_description, task_obs_code_string=task_file)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    logging.info( f"Generating samples with {cfg.model}")

    task_code_string = task_code_string.replace(task, task+suffix)

    #create_task(OUTPUTS_DIR, cfg.env.task, cfg.env.env_name, suffix)
    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
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
        # logging.info(responses[0])
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
                shutil.copy(output_file, f"{OUTPUTS_DIR}/tasks/env_iter{iter}_response{response_id}.py")

                env_iter_file = f'results/tasks/env_iter{iter}_response{response_id}.py'
                # convert file path to module name
                module_name = env_iter_file.replace('/','.').replace('.py','').lstrip()
                env_module = importlib.import_module(module_name)

                # Instantiate environment
                env = env_module.Walker2dEnv()
                env.reset()
                # Training the environment 
                trainer = train.TrainingManager(env=env)
                model = trainer.run()
                print("completion succeded")      



if __name__ == "__main__":
    main()
