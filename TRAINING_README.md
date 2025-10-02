# MAPPO Training for Regicide (PyTorch)

This training script implements **Multi-Agent Proximal Policy Optimization (MAPPO)** for the Regicide cooperative card game environment using **PyTorch** (no JAX dependencies).

## Features

✅ **Pure PyTorch Implementation** - No JAX required  
✅ **TensorBoard Logging** - Monitor training in real-time  
✅ **Action Masking** - Handles invalid actions properly  
✅ **Parameter Sharing** - Agents share network weights  
✅ **GAE (Generalized Advantage Estimation)** - Improved advantage computation  
✅ **Checkpointing** - Save and resume training  
✅ **Evaluation Mode** - Test policy performance periodically  

## Installation

Install the required dependencies:

```bash
pip install -r requirements_pytorch.txt
```

Or manually install:

```bash
pip install torch>=2.0.0 tensorboard>=2.14.0 gymnasium>=0.28.0 numpy>=1.21.0
```

## Quick Start

### Basic Training

Train with default settings (2 players, 10,000 episodes):

```bash
python train_mappo_pytorch.py
```

### Custom Configuration

Train with custom parameters:

```bash
python train_mappo_pytorch.py \
    --num-players 2 \
    --episodes 20000 \
    --batch-size 512 \
    --lr 0.0003 \
    --hidden-dim 256 \
    --log-dir runs/my_experiment
```

### Resume from Checkpoint

Continue training from a saved checkpoint:

```bash
python train_mappo_pytorch.py \
    --checkpoint runs/regicide_mappo/checkpoints/checkpoint_5000.pt \
    --episodes 15000
```

## Command-Line Arguments

### Environment Settings
- `--num-players` (int): Number of players in the game (default: 2)
- `--max-steps` (int): Maximum steps per episode (default: 200)

### Training Hyperparameters
- `--episodes` (int): Number of training episodes (default: 10000)
- `--batch-size` (int): Batch size for PPO updates (default: 256)
- `--lr` (float): Learning rate for actor and critic (default: 3e-4)
- `--gamma` (float): Discount factor (default: 0.99)
- `--hidden-dim` (int): Hidden layer size (default: 256)

### Logging and Checkpointing
- `--log-dir` (str): Directory for TensorBoard logs (default: 'runs/regicide_mappo')
- `--log-interval` (int): Episodes between logging (default: 10)
- `--eval-interval` (int): Episodes between evaluations (default: 100)
- `--save-interval` (int): Episodes between checkpoints (default: 500)

### Other
- `--seed` (int): Random seed for reproducibility (default: None)
- `--checkpoint` (str): Path to checkpoint to resume from (default: None)

## Monitoring Training

### TensorBoard

Start TensorBoard to monitor training:

```bash
tensorboard --logdir runs/
```

Then open your browser to `http://localhost:6006`

### Metrics Tracked

**Training Metrics:**
- `train/reward` - Average episode reward
- `train/episode_length` - Average episode length
- `train/win_rate` - Percentage of games won
- `train/policy_loss` - PPO policy loss
- `train/value_loss` - Value function loss
- `train/entropy` - Policy entropy (exploration measure)
- `train/total_steps` - Total environment steps

**Evaluation Metrics:**
- `eval/reward` - Evaluation episode reward
- `eval/episode_length` - Evaluation episode length
- `eval/win_rate` - Evaluation win rate

## Architecture

### Network Structure

```
Input (48 features)
    ↓
Shared Feature Extractor (3 layers, 256 hidden)
    ├─→ Actor Head → Action Logits (30 actions)
    └─→ Critic Head → Value Estimate (1 value)
```

### MAPPO Algorithm

1. **Collect Experience**: Run episodes and store trajectories
2. **Compute Advantages**: Use GAE for advantage estimation
3. **Update Policy**: Optimize using PPO with clipped objective
4. **Repeat**: Continue until convergence

### Key Components

- **Actor-Critic Network**: Shared feature extractor with separate actor/critic heads
- **Experience Buffer**: Stores transitions with GAE computation
- **PPO Optimizer**: Clipped surrogate objective with entropy bonus
- **Action Masking**: Prevents selection of invalid actions

## Configuration

The `MAPPOConfig` class contains all hyperparameters:

```python
config = MAPPOConfig(
    # Environment
    num_players=2,
    max_steps=200,
    
    # Network
    hidden_dim=256,
    num_layers=3,
    
    # Training
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    
    # PPO
    ppo_epochs=4,
    batch_size=256,
)
```

## Example Output

```
Environment: 2 players
Observation dim: 48, Action dim: 30
Device: cuda

============================================================
Starting MAPPO Training
============================================================

Episode     0 | Reward:   -5.10 | Length:  12.0 | Win%:   0.0 | FPS:  245.3
Episode    10 | Reward:   -3.20 | Length:  15.3 | Win%:   5.0 | FPS:  312.1
Episode    20 | Reward:   -1.50 | Length:  18.7 | Win%:  10.0 | FPS:  345.7

Evaluating...
Eval: Reward=2.30, Length=25.4, Win%=15.0

Episode   100 | Reward:    8.20 | Length:  32.1 | Win%:  35.0 | FPS:  401.2
Episode   200 | Reward:   15.40 | Length:  45.2 | Win%:  55.0 | FPS:  425.8
...
```

## Tips for Training

### Improving Performance

1. **Increase Hidden Dimensions**: Try `--hidden-dim 512` for more capacity
2. **Adjust Learning Rate**: Lower LR (`--lr 1e-4`) for more stable training
3. **Longer Training**: Use `--episodes 50000` for better convergence
4. **Batch Size**: Increase `--batch-size 512` if you have enough GPU memory

### Debugging

1. **Check Action Masking**: Ensure invalid actions are properly masked
2. **Monitor Entropy**: Should decrease over time but not too quickly
3. **Value Loss**: Should decrease steadily during training
4. **Win Rate**: Should increase over time (may plateau)

### Common Issues

**Low Win Rate**: Try increasing hidden dimensions or training longer  
**Policy Collapse**: Increase entropy coefficient or decrease learning rate  
**Slow Training**: Use GPU (CUDA) and increase batch size  
**Out of Memory**: Reduce batch size or hidden dimensions  

## Advanced Usage

### Custom Reward Shaping

Edit the `_calculate_reward` method in `CompactRegicideEnv`:

```python
def _calculate_reward(self, success: bool, message: str, action_info: Dict) -> float:
    if self.engine.status == GameStatus.WON:
        return 100.0
    elif self.engine.status == GameStatus.LOST:
        return -100.0
    
    if not success:
        return -1.0
    
    # Add custom reward shaping here
    reward = 0.1
    
    # Bonus for dealing damage
    if 'damage_dealt' in action_info:
        reward += action_info['damage_dealt'] * 0.1
    
    return reward
```

### Wandb Integration

For advanced experiment tracking, uncomment wandb in requirements and add to trainer:

```python
import wandb

wandb.init(project="regicide-mappo", config=vars(config))
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{regicide_mappo_pytorch,
  title={MAPPO Training for Regicide Card Game},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/RegiSide}
}
```

## License

See LICENSE file in the repository root.

## Acknowledgments

- MAPPO algorithm: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
- Regicide card game: Original game by Badgers from Mars
- PyTorch implementation: Based on CleanRL and RLlib examples
