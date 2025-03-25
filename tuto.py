import openai
import os
import gymnasium as gym 
from pathlib import Path
# Load API key from environment variable
#api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
#client = openai.OpenAI(api_key=api_key)

# List available models
#models = client.models.list()

# Print model IDs
#for model in models.data:
 #   print(model.id)
from model_test_env.walker2d_v5 import Walker2dEnv

#env = Walker2dEnv()

# obs = env.reset()
