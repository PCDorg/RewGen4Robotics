import hydra
import numpy as np 
import logging 
import os
from openai import OpenAI
import traceback
from pathlib import Path
import shutil
import time 
import importlib
from omegaconf import DictConfig
from utils.file import *
from groq import Groq
from model_test_env import train
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from utils.environment_init import *
import glfw

EUREKA_ROOT_DIR = os.getcwd()
OUTPUTS_DIR = f"{EUREKA_ROOT_DIR}/results"


client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
client_GROQ = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

@hydra.main(config_path="/home/bechir/RewGen4Robotic/cfg", config_name="config", version_base="1.1")
def main(cfg):
    suffix = cfg.llm.suffix
    task=cfg.env.task
    task_description= cfg.env.task_description
    
    logging.info(f"Using LLM: {cfg.llm.model}")
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
    output_file = f"{OUTPUTS_DIR}/tasks/{env_name}{cfg.llm.suffix.lower()}.py"

    #loading the prompts
    prompt_path= f"{workspace_dir}/utils/prompts"
    system_prompt = file_to_string(f'{workspace_dir}/utils/walker_prompts/system_prompt.txt')
    user_prompt = file_to_string(f'{prompt_path}/user_prompt.txt')
    reward_signature = file_to_string(f'{prompt_path}/reward_signature.txt')
    code_output_tip = file_to_string(f'{prompt_path}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_path}/code_feedback.txt')
    policy_feedback = file_to_string(f'{prompt_path}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_path}/execution_error_feedback.txt')

    code_output_tip = code_output_tip.format(task_reward_signature_string=reward_signature)
    system_prompt = system_prompt.format(task_reward_signature_string=reward_signature) + code_output_tip
    user_prompt = user_prompt.format(task_description=task_description, task_obs_code_string=task_obs_code_string)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    logging.info(f"Generating samples with {cfg.llm.model}")

    task_code_string = task_code_string.replace(task, task+suffix)

    max_rewards = list()


    #create_task(OUTPUTS_DIR, cfg.env.task, cfg.env.env_name, suffix)
    DUMMY_FAILURE = -10000.
    execute_rates = []
    best_code_paths = []
    max_reward_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    model_paths = []

    for iter in range(cfg.iteration) :
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = 1

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.llm.model}")

        while True :
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    if cfg.llm.provider == "groq":
                        response_cur = client_GROQ.chat.completions.create(model=cfg.llm.model,
                        messages=messages,
                        temperature=cfg.llm.temperature,
                        n=chunk_size# groq only supports 1 sample at a time
                        )
                    elif cfg.llm.provider == "openai":
                        response_cur = client.chat.completions.create(model=cfg.llm.model,
                        messages=messages,
                        temperature=cfg.llm.temperature,
                        n=chunk_size)
                    else:
                        raise ValueError(f"Invalid provider: {cfg.llm.provider}")
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)# reduce the chunk size by half in case of error
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
                f"self.extras['gpt_reward'] = self.rew_buf.mean() if len(self.rew_buf)>0 else 0",
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
                initialize_glfw()
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
                    trainer = train.TrainingManager(env=env,root_dir=workspace_dir,iter=iter,reponse_id=response_id,config={'timesteps':cfg.timesteps})
                    trained_model = trainer.run()
                    if iter == cfg.iteration-1 :
                        model_paths.append(trainer.get_model_path())
                    
                except Exception as e:
                    logging.error("failed to train the model ! " + str(e))
                    content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                    content += code_output_tip  
                    contents.append(content)
                    mean_reward_per_sample.append(DUMMY_FAILURE)
                    traceback_msg=traceback.print_exc()
                    continue
                
                try:
                    env.close()
                except Exception as e:
                    logging.warning(f"Error closing environment: {str(e)}")
                finally:
                    # Force cleanup of GLFW context
                    if glfw.get_current_context():
                        glfw.terminate()
                content = ""
                if traceback_msg == "":
                    logging.info("Traceback message is empty")
                    # extracting rward stats from tensorboard
                    logging.info( "extracting reward stats from tensorboard")
                    try:
                        tensorboard_logs = load_tensorboard_logs(trainer.get_logs_path())
                        #logging.info(f"tensorboard_logs : {tensorboard_logs}")
                        
                        
                        ep_reward_mean = np.array(tensorboard_logs["rollout/ep_rew_mean"]).mean()
                        ep_length_mean = np.array(tensorboard_logs["rollout/ep_len_mean"]).mean()
                        cumulative_reward = np.array(tensorboard_logs["cumulative_rewards"])

                        logging.info( f"iteration [{iter}], sample number [{response_id}]// episode reward mean : {ep_reward_mean}")
                        logging.info( f"iteration [{iter}], sample number [{response_id}]// episode length mean : {ep_length_mean}")
                        

                        mean_reward_per_sample.append(ep_reward_mean) 
                        max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                        epoch_freq = max(int(max_iterations // 10), 1)        
                        content += policy_feedback.format(epoch_freq=epoch_freq)
                        tensorboard_metrics = ['rollout/ep_len_mean', 'rollout/ep_rew_mean']
                        for metric in tensorboard_metrics:
                                # Limit to 20 values by taking evenly spaced samples
                                metric_values = tensorboard_logs[metric]
                                if len(metric_values) > 20:
                                    step = len(metric_values) // 20
                                    metric_values = metric_values[::step][:20]
                                metric_cur = ['{:.2f}'.format(x) for x in metric_values]
                                metric_cur_max = max(tensorboard_logs[metric])
                                metric_cur_mean = np.array(tensorboard_logs[metric]).mean()
                                metric_cur_min = min(tensorboard_logs[metric])
                                
                                content += f"{metric}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                        code_feedbacks.append(code_feedback)
                        content += code_feedback
                        logging.info("*********** evaluation is done ***********")
                    except Exception as e:
                        logging.error(f"Error loading TensorBoard logs: {str(e)}")
                        mean_reward_per_sample.append(DUMMY_FAILURE)
                        content += f"Error loading training metrics: {str(e)}\n"
                else :
                    logging.info("Traceback message is not empty")
                    logging.info("Traceback message is not empty")
                    logging.info("Traceback message is not empty")
                    logging.info("Traceback message is not empty")
                    content+= execution_error_feedback(traceback_msg= traceback_msg)

                content += code_output_tip
                contents.append(content)

        #iteration terminated
        logging.info(f"************** Iteration {iter} : Terminated **************")
        
        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(mean_reward_per_sample))
        best_content = contents[best_sample_idx]
        max_reward = mean_reward_per_sample[best_sample_idx]

        
        if max_reward > max_reward_overall or max_reward_code_path is None:
            max_reward_overall = max_reward
            max_reward_code_path = code_paths[best_sample_idx]
            if iter == cfg.iteration -1 : 
                max_reward_model_path = model_paths[best_sample_idx]

        try :
            logging.info("this is the content of the code paths %s" , code_paths)
            logging.info(f"max reward code path length: {len(max_reward_code_path)} **********")
        except Exception as e:
            logging.error(f"Error logging max reward code path: {str(e)}")
            exit()

        max_rewards.append(max_reward)
        best_code_paths.append(max_reward_code_path)
        logging.info(f"Iteration {iter}: Max Success: {max_reward}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        #logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx].message.content + "\n")
        #logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        messages += [{"role": "assistant", "content": str(responses[best_sample_idx].message.content)}]
        sanitized_content = str(best_content)
        messages += [{"role": "user", "content": sanitized_content}]

        
        try:
            # create the conversation file if it doesn't exist
            messages_file = os.path.join(OUTPUTS_DIR, 'messages.txt')
            os.makedirs(os.path.dirname(messages_file), exist_ok=True)
        
            with open(messages_file, 'a') as file:
                file.write('\n'.join('Role :' + str(msg['role'])+ '\n' + 'Content :' + str(msg['content']) + '\n' for msg in messages))
            logging.info(f"Messages saved to {messages_file}")
        except Exception as e:
            logging.error(f"Error saving messages: {str(e)}")

    #log that the evaluation phase is starting
    logging.info("*********** Evaluation phase is starting ***********")

    
    # Evaluate the best reward code many times
    if max_reward_code_path is None: 
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()

    logging.info(f"Task: {task}, Max Training reward {max_reward_overall}, Best Reward Code Path: {max_reward_code_path}")
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    #shutil.copy(max_reward_code_path, output_file)

    reward_code_final_rewards = []
    x_axis = list(range(1, 101))
    for i in range(cfg.num_eval):
        # convert file path to module name
        module_name = max_reward_code_path.replace('/','.').replace('.py','').lstrip()
        env_module = importlib.import_module(module_name)

        # Instantiate environment
        try :
            env = env_module.Walker2dEnv()
        except Exception as e:
            logging.error(f"Error instantiating environment: {str(e)}")
            continue
        env.reset()
        max_reward_model_path = model_paths[best_sample_idx]
        model = PPO.load(max_reward_model_path)
        # Load the model but make sure it doesn't get mixed with message content
        evaluation_model = PPO.load(max_reward_model_path)
        # Don't use the variable name 'model' which is already used for OpenAI model config
        episode_rewards = []

        try:
            episodes_rewards = []
            # evaluation for 100 episodes
            for _ in range(cfg.num_eval_episodes): 
                obs, info = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = evaluation_model.predict(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                episodes_rewards.append(episode_reward)

            # Plot all episodes at once after collecting all rewards
            if episodes_rewards:  # Only plot if we have rewards
                plt.style.use('seaborn-whitegrid')
                plt.figure(figsize=(8, 6), dpi=300)
                plt.plot(range(1, len(episodes_rewards) + 1), episodes_rewards, 
                        color='#2E64FE', linewidth=2, marker='o', markersize=4)
                plt.xlabel('Episode',fontsize=12, fontweight='bold')
                plt.ylabel('Reward',fontsize=12, fontweight='bold')
                plt.title('Evaluation Progress',fontsize=14, fontweight='bold', pad=15)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.box(True)
                plt.tight_layout()
                plt.savefig(f'{workspace_dir}/plots/evaluation_progress_n°{i}.pdf', format='pdf', bbox_inches='tight')
                plt.savefig(f'{workspace_dir}/plots/evaluation_progress_n°{i}.png', format='png', dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to free memory

            if episodes_rewards:  # Only append if we have rewards
                reward_code_final_rewards.append(np.mean(episodes_rewards))
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            logging.error(traceback.format_exc())
        finally:
            # Clean up
            try:
                env.close()
            except Exception as e:
                logging.error(f"Error closing environment: {str(e)}")
            """finally:
                    # Force cleanup of GLFW context
                    if glfw.get_current_context():
                        glfw.terminate()"""
        

    logging.info(f"Final Reward Mean: {np.mean(reward_code_final_rewards)}")
    logging.info(f"Final Reward Std: {np.std(reward_code_final_rewards)}")
    logging.info(f"Final Reward Min: {np.min(reward_code_final_rewards)}")
    logging.info(f"Final Reward Max: {np.max(reward_code_final_rewards)}")
    logging.info("******************** Process finished ********************")

if __name__ == "__main__":
    main()