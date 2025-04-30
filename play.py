import os
import argparse
import logging
import hydra
from omegaconf import DictConfig
import importlib
import traceback

# Assume SAC is the agent used, adjust if PPO or others are needed
from stable_baselines3 import SAC
# from stable_baselines3 import PPO # Uncomment if PPO models are used

# Dictionary mapping environment names to their factory functions and modules
# (Should match the one in gen.py)
ENV_FACTORIES = {
    'fetchReach': ('envs.fetchReach', 'make_custom_fetch'),
    'antmaze': ('envs.antmaze', 'make_custom_antmaze'),
    'go1': ('envs.Go2Env', 'make_custom_go1'),
}

def get_env_factory(env_name):
    """Dynamically imports and returns the environment factory function."""
    if env_name not in ENV_FACTORIES:
        raise ValueError(f'Unknown environment name: {env_name}. Available: {list(ENV_FACTORIES.keys())}')

    module_name, func_name = ENV_FACTORIES[env_name]
    try:
        module = importlib.import_module(module_name)
        factory_func = getattr(module, func_name)
        return factory_func
    except ImportError as e:
        logging.error(f'Could not import module {module_name} for environment {env_name}: {e}')
        raise
    except AttributeError as e:
        logging.error(f'Could not find factory function {func_name} in module {module_name}: {e}')
        raise

def play(args):
    """Loads the environment and model, then runs the simulation."""
    logging.info(f'Playing with environment: {args.env}')
    logging.info(f'Using model: {args.model}')
    logging.info(f'Using reward function: {args.reward}')

    # Check if files exist
    if not os.path.exists(args.model):
        logging.error(f'Model file not found: {args.model}')
        return
    if not os.path.exists(args.reward):
        logging.error(f'Reward function file not found: {args.reward}')
        return

    try:
        # Initialize Hydra to load the environment config
        # Assumes config structure is cfg/config.yaml and cfg/env/<env_name>.yaml
        hydra.initialize(config_path="./cfg", version_base="1.1")
        # Load config, overriding the default env with the one specified
        cfg = hydra.compose(config_name="config", overrides=[f'env={args.env}'])
        logging.info(f'Loaded configuration for \'{args.env}\' environment.')

        # Get the environment factory function
        env_factory = get_env_factory(args.env)

        # Prepare environment arguments
        env_kwargs = {}
        if args.env == 'go1':
             env_kwargs['ctrl_type'] = cfg.env.get('ctrl_type', 'torque') # Use ctrl_type from config

        # Create the environment using the factory and reward function path
        logging.info("Creating environment...")
        env = env_factory(reward_function_path=args.reward, **env_kwargs)
        logging.info("Environment created successfully.")

        # Load the trained model (assuming SAC, change if needed)
        logging.info("Loading model...")
        # Use the appropriate class (SAC, PPO, etc.) based on how the model was saved
        model = SAC.load(args.model, env=env) # Pass env for model compatibility check
        # model = PPO.load(args.model, env=env) # If using PPO
        logging.info("Model loaded successfully.")

        # Run the simulation loop
        logging.info("Starting simulation... Press Ctrl+C to exit.")
        episodes = args.episodes
        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0
            step = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True) # Use deterministic actions for eval
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                step += 1
                try:
                    env.render()
                except Exception as render_e:
                     # Handle potential rendering issues gracefully (e.g., no display)
                     if step == 0 and ep == 0: # Log only once per run
                          logging.warning(f'Rendering failed: {render_e}. Will continue without rendering.')
                     pass # Continue without rendering
            logging.info(f'Episode {ep+1} finished. Reward: {ep_reward:.2f}, Steps: {step}')

    except ValueError as e:
        logging.error(f'Configuration or setup error: {e}')
    except FileNotFoundError as e:
        logging.error(f'File not found error: {e}')
    except Exception as e:
        logging.error(f'An unexpected error occurred during playback: {e}')
        logging.error(traceback.format_exc())
    finally:
        if 'env' in locals() and env is not None:
            try:
                env.close()
                logging.info("Environment closed.")
            except Exception as close_e:
                logging.warning(f'Error closing environment: {close_e}')

def main():
    parser = argparse.ArgumentParser(description="Play a trained RL agent in a specified environment.")
    parser.add_argument("--env", type=str, required=True, choices=list(ENV_FACTORIES.keys()),
                        help="Name of the environment to run (e.g., fetchReach, antmaze, go1).")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained agent model (.zip file).")
    parser.add_argument("--reward", type=str, required=True,
                        help="Path to the Python file defining the custom reward function.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to play (default: 10).")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    play(args)

if __name__ == "__main__":
    main()
