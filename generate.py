
import os
import openai 
import logging 
from utils import file, Conversation  

import hydra 



@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):


    task=cfg.task
    task_description= cfg.task_description

    #loading the prompts
    workspace_dir = str(os.getcwd())

    prompt_path= f"{workspace_dir}/utils/prompts"
    
    system_prompt = file.file_to_string(f'{prompt_path}/system_prompt.txt')
    user_prompt = file.file_to_string(f'{prompt_path}/user_prompt.txt')
    reward_signature = file.file_to_string(f'{prompt_path}/reward_signature.txt')
    code_output_tip = file.file_to_string(f'{prompt_path}/code_output_tip.txt')
    


    system_prompt = system_prompt.format(task_reward_signature_string=reward_signature) + code_output_tip
    user_prompt = user_prompt.format(task_description=task_description)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    openai.api_key = os.getenv("OPEN_AI_KEY")
    logging.info( f"Generating samples with {cfg.model}")

    """if openai.api_key:
        print("API Key found:")
    
       
    else:
        print("API Key not found")"""

    try: 
        response= openai.chat.completions.create(
            model="gpt-4o", 
            messages = messages
        )
    except Exception as e : 
        logging.exception("Attempt failed with an error")
    #print(response.choices[0].message.content)   
    
    # save the response to a file
    content = response.choices[0].message.content
    text_filename = f"{workspace_dir}/results/text.txt"
    code_filename = f"{workspace_dir}/results/code.txt"
    Conversation.save_content(content, code_filename, text_filename)
    logging.info(f"Content saved to {code_filename} and {text_filename}")



    



if __name__ == "__main__":
    main()
