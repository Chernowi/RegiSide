# Regicide MARL Training Framework

A modular training framework for Multi-Agent Reinforcement Learning on the Regicide environment.

## Quick Start

### Installation

```bash
# Install dependencies (PyTorch + CUDA by default)
pip install -r requirements.txt

# Optional: Install JAX backend (if you want to use backend: "jax")
# pip install jax jaxlib flax optax chex
```

### Basic Training

```bash
# Train with default MAPPO configuration
python train.py --config configs/mappo_default.yaml

# Override parameters
python train.py --config configs/mappo_default.yaml --seed 123 --run-name my_experiment

# Quick test run
python train.py --config configs/mappo_default.yaml --timesteps 1000000 --num-envs 32
```

### Monitor Training

```bash
# View tensorboard logs
tensorboard --logdir=runs

# Access at http://localhost:6006
```

## Framework Structure

```
RegiSide/
├── train.py                 # Main training script
├── configs/                 # Configuration files
│   └── mappo_default.yaml  # Default MAPPO config
├── src/
│   ├── runners/            # Algorithm runners
│   │   ├── base_runner.py  # Abstract base class
│   │   └── mappo_runner.py # MAPPO implementation
│   └── ...                 # Environment implementations
├── runs/                   # Tensorboard logs (auto-created)
└── checkpoints/            # Model checkpoints (auto-created)
```

## Configuration

All training parameters are specified in YAML configuration files in the `configs/` directory.

### Key Configuration Sections

- **algorithm**: Algorithm name (e.g., "mappo")
- **backend**: Backend to use ("pytorch" [default] or "jax")
- **env**: Environment settings (num_players, num_envs, etc.)
- **mappo**: Algorithm-specific hyperparameters
- **logging**: Logging intervals and settings
- **checkpoint**: Model saving configuration
- **evaluation**: Evaluation settings

### Example Configuration

```yaml
algorithm: "mappo"
backend: "pytorch"  # Options: pytorch (default), jax

env:
  num_players: 4
  num_envs: 64

mappo:
  total_timesteps: 50_000_000
  learning_rate: 3.0e-4
  gamma: 0.99
  
logging:
  use_tensorboard: true
  log_interval: 10
  
checkpoint:
  save_dir: "checkpoints"
  save_best: true
```

## Command-Line Options

```bash
python train.py --help

Options:
  --config PATH         Path to YAML configuration file (required)
  --seed INT           Random seed (overrides config)
  --run-name STR       Name for this training run
  --device STR         Device: auto, cpu, cuda, gpu
  --num-envs INT       Number of parallel environments
  --timesteps INT      Total training timesteps
  --no-tensorboard     Disable tensorboard logging
  --wandb              Enable Weights & Biases logging
  --wandb-project STR  W&B project name
  --resume PATH        Resume training from checkpoint
```

## Adding New Algorithms

To add a new algorithm (e.g., QMIX):

1. Create configuration: `configs/qmix_default.yaml`
2. Implement runner: `src/runners/qmix_runner.py`
   - Inherit from `BaseRunner`
   - Implement abstract methods: `setup()`, `train()`, `evaluate()`, `_save_checkpoint_impl()`
3. Register in `train.py`:
   ```python
   from src.runners import QMIXRunner
   
   ALGORITHM_REGISTRY = {
       'mappo': MAPPORunner,
       'qmix': QMIXRunner,  # Add here
   }
   ```

## Runner Class Interface

All runners inherit from `BaseRunner` and must implement:

- `setup()`: Initialize environment, networks, optimizer
- `train()`: Main training loop
- `evaluate()`: Policy evaluation
- `_save_checkpoint_impl()`: Save algorithm-specific state

The base class provides:
- Configuration management
- Logging (console + tensorboard)
- Checkpointing
- Metrics tracking

## Logging

### Tensorboard
- Enabled by default
- Logs saved to `runs/<run_name>/`
- Tracks: losses, rewards, policy metrics

### Console
- Configurable verbosity
- Log intervals in config
- Progress tracking

### Checkpoints
- Saved to `checkpoints/<run_name>/`
- Best model saved separately
- Includes full training state for resumption

## Examples

### Train MAPPO with custom settings
```bash
python train.py --config configs/mappo_default.yaml \
  --seed 42 \
  --run-name mappo_4players_64envs \
  --num-envs 64
```

### Quick test run (1M timesteps)
```bash
python train.py --config configs/mappo_default.yaml \
  --timesteps 1000000 \
  --run-name quick_test
```

### Resume training
```bash
python train.py --config configs/mappo_default.yaml \
  --resume checkpoints/mappo_regicide_20251002_120000/checkpoint_step_1000000.pt
```

### Use W&B for logging
```bash
python train.py --config configs/mappo_default.yaml \
  --wandb \
  --wandb-project regicide-experiments
```
