# Reinforcement Learning with LLM-Generated Reward Functions

This project combines reinforcement learning with large language models to automatically generate and optimize reward functions for robotic tasks.

## Features

- Uses SAC (Soft Actor-Critic) for reinforcement learning
- Integrates with OpenAI's GPT models for reward function generation
- Supports the FetchReach environment from Gymnasium
- Configurable through Hydra configuration files
- TensorBoard logging for training visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Configuration

The project uses Hydra for configuration management. The main configuration file is located at `cfg/config.yaml`. You can modify the following sections:

- `training`: SAC algorithm parameters
- `env`: Environment settings
- `llm`: OpenAI model settings
- `experiment`: Experiment parameters
- `paths`: Directory paths

### Overriding Configuration

You can override any configuration parameter using command-line arguments:

```bash
python gen.py training.total_timesteps=200000 experiment.iterations=10
```

## Usage

1. Basic training with default configuration:
```bash
python gen.py
```

2. Training with custom configuration:
```bash
python gen.py training.total_timesteps=200000 experiment.iterations=10
```

3. Training with a specific environment:
```bash
python gen.py env=your_custom_env
```

## Directory Structure

```
.
├── cfg/                    # Configuration files
│   ├── config.yaml        # Main configuration
│   └── env/              # Environment-specific configurations
├── envs/                  # Custom environments
├── models/               # Saved models
├── results/              # Training results
├── utils/               # Utility functions
├── gen.py              # Main training script
└── README.md           # This file
```

## Output

The training process generates:
- TensorBoard logs in the specified tensorboard directory
- Saved models in the models directory
- Training results and logs in the results directory
- Generated reward functions for each iteration

## Monitoring Training

You can monitor the training progress using TensorBoard:
```bash
tensorboard --logdir results/reach/tensorboard_logs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license]

## Acknowledgments

- OpenAI for the GPT models
- Stable Baselines3 for the RL implementation
- Hydra for configuration management