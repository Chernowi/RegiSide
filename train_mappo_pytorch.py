#!/usr/bin/env python3
"""
MAPPO Training Script for Regicide Environment (PyTorch, No JAX)

This script implements Multi-Agent Proximal Policy Optimization (MAPPO) for the 
Regicide cooperative card game environment using pure PyTorch (no JAX dependencies).

Features:
- PyTorch-based actor-critic networks
- Centralized training, decentralized execution (CTDE)
- Action masking for invalid actions
- TensorBoard logging for monitoring
- Experience buffer with GAE (Generalized Advantage Estimation)
- Multi-episode parallel training
- Checkpointing and model saving

Usage:
    python train_mappo_pytorch.py --num-players 2 --episodes 10000 --log-dir runs/regicide_mappo
"""

import os
import sys
import time
import argparse
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from compact_regicide_env import CompactRegicideEnv, GameStatus


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO training"""
    # Environment settings
    num_players: int = 2
    max_steps: int = 200
    
    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 3
    
    # Training hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # PPO specific
    ppo_epochs: int = 4
    num_minibatches: int = 4
    batch_size: int = 256
    
    # Training settings
    num_episodes: int = 10000
    eval_interval: int = 100
    save_interval: int = 500
    log_interval: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for MAPPO.
    
    Actor: Outputs action logits for each agent
    Critic: Outputs centralized value estimate
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        layers = []
        layers.append(nn.Linear(obs_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
        
        self.shared_net = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, action_mask=None):
        """
        Forward pass through network.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            action_mask: Boolean mask for valid actions [batch_size, action_dim]
        
        Returns:
            action_logits: Logits for action distribution [batch_size, action_dim]
            value: Value estimate [batch_size, 1]
        """
        features = self.shared_net(obs)
        
        action_logits = self.actor(features)
        value = self.critic(features)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Set invalid actions to very large negative value
            action_logits = action_logits.masked_fill(~action_mask, -1e8)
        
        return action_logits, value
    
    def get_action(self, obs, action_mask=None, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            action_mask: Boolean mask for valid actions [batch_size, action_dim]
            deterministic: If True, take most likely action
        
        Returns:
            action: Sampled action [batch_size]
            log_prob: Log probability of action [batch_size]
            value: Value estimate [batch_size]
        """
        action_logits, value = self.forward(obs, action_mask)
        
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(self, obs, actions, action_mask=None):
        """
        Evaluate actions under current policy.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            actions: Actions to evaluate [batch_size]
            action_mask: Boolean mask for valid actions [batch_size, action_dim]
        
        Returns:
            log_probs: Log probabilities of actions [batch_size]
            values: Value estimates [batch_size]
            entropy: Policy entropy [batch_size]
        """
        action_logits, values = self.forward(obs, action_mask)
        
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class ExperienceBuffer:
    """
    Buffer for storing and processing agent experiences.
    Supports GAE (Generalized Advantage Estimation).
    """
    
    def __init__(self, config: MAPPOConfig):
        self.config = config
        self.clear()
    
    def clear(self):
        """Clear all stored experiences"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.action_masks = []
        self.dones = []
    
    def add(self, obs, action, reward, value, log_prob, action_mask, done):
        """Add a single experience"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.action_masks.append(action_mask)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, last_value):
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_value: Value estimate for final state
        
        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        values_extended = np.append(values, last_value)
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.config.gamma * values_extended[t + 1] - values[t]
                last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
            
            advantages[t] = last_gae
        
        # Compute returns
        returns = advantages + values
        
        return returns, advantages
    
    def get_batches(self, returns, advantages):
        """
        Get mini-batches for training.
        
        Args:
            returns: Computed returns
            advantages: Computed advantages
        
        Yields:
            Dictionary containing batch data
        """
        batch_size = len(self.observations)
        indices = np.arange(batch_size)
        
        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, batch_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_idx = indices[start:end]
                
                yield {
                    'observations': torch.FloatTensor(np.array(self.observations)[batch_idx]).to(self.config.device),
                    'actions': torch.LongTensor(np.array(self.actions)[batch_idx]).to(self.config.device),
                    'old_log_probs': torch.FloatTensor(np.array(self.log_probs)[batch_idx]).to(self.config.device),
                    'returns': torch.FloatTensor(returns[batch_idx]).to(self.config.device),
                    'advantages': torch.FloatTensor(advantages[batch_idx]).to(self.config.device),
                    'action_masks': torch.BoolTensor(np.array(self.action_masks)[batch_idx]).to(self.config.device),
                }


class MAPPOTrainer:
    """
    MAPPO trainer for multi-agent Regicide environment.
    """
    
    def __init__(self, config: MAPPOConfig, log_dir: str = "runs/regicide_mappo"):
        self.config = config
        self.log_dir = log_dir
        
        # Create environment
        self.env = CompactRegicideEnv(num_players=config.num_players)
        
        # Get dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        print(f"Environment: {config.num_players} players")
        print(f"Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
        print(f"Device: {config.device}")
        
        # Create networks for each agent (parameter sharing)
        self.networks = {}
        self.optimizers = {}
        
        for agent_idx in range(config.num_players):
            network = ActorCriticNetwork(
                self.obs_dim, 
                self.action_dim,
                config.hidden_dim,
                config.num_layers
            ).to(config.device)
            
            # Use parameter sharing - all agents share the same network
            if agent_idx == 0:
                self.shared_network = network
                self.optimizer = optim.Adam(
                    network.parameters(), 
                    lr=config.lr_actor
                )
            
            self.networks[agent_idx] = self.shared_network
            self.optimizers[agent_idx] = self.optimizer
        
        # Experience buffers for each agent
        self.buffers = {agent_idx: ExperienceBuffer(config) for agent_idx in range(config.num_players)}
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.win_rate = deque(maxlen=100)
        self.total_steps = 0
        self.total_episodes = 0
    
    def collect_experience(self, num_steps: int = None):
        """
        Collect experience by running episodes.
        
        Args:
            num_steps: Maximum number of steps to collect (None = full episode)
        """
        obs, info = self.env.reset()
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps or self.config.max_steps):
            # Get current player
            current_agent = self.env.current_player_idx
            
            # Get action mask
            action_mask = self.env.get_valid_action_mask()
            
            # Convert to tensors
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.config.device)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.networks[current_agent].get_action(
                    obs_tensor, mask_tensor
                )
            
            action = action.item()
            log_prob = log_prob.item()
            value = value.item()
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            done = terminated or truncated
            
            # Store experience
            self.buffers[current_agent].add(
                obs, action, reward, value, log_prob, action_mask, done
            )
            
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            obs = next_obs
            
            if done:
                break
        
        # Compute final value for bootstrapping
        if not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)
            mask_tensor = torch.BoolTensor(self.env.get_valid_action_mask()).unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                _, _, last_value = self.networks[self.env.current_player_idx].get_action(
                    obs_tensor, mask_tensor
                )
            last_value = last_value.item()
        else:
            last_value = 0.0
        
        # Store episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.win_rate.append(float(info.get('game_status', 0) == GameStatus.WON))
        self.total_episodes += 1
        
        return last_value, done, info
    
    def update_policy(self):
        """
        Update policy using PPO.
        """
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        # Update each agent's policy
        for agent_idx in range(self.config.num_players):
            buffer = self.buffers[agent_idx]
            
            if len(buffer.observations) == 0:
                continue
            
            # Compute returns and advantages
            last_value = 0.0  # We'll use actual last value from collect_experience
            returns, advantages = buffer.compute_returns_and_advantages(last_value)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Update policy with mini-batches
            for batch in buffer.get_batches(returns, advantages):
                # Evaluate actions under current policy
                log_probs, values, entropy = self.networks[agent_idx].evaluate_actions(
                    batch['observations'],
                    batch['actions'],
                    batch['action_masks']
                )
                
                # Compute PPO loss
                ratio = torch.exp(log_probs - batch['old_log_probs'])
                
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 
                                   1.0 + self.config.clip_epsilon) * batch['advantages']
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch['returns'])
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.value_loss_coef * value_loss + 
                       self.config.entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizers[agent_idx].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.networks[agent_idx].parameters(), 
                                        self.config.max_grad_norm)
                self.optimizers[agent_idx].step()
                
                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
            
            # Clear buffer
            buffer.clear()
        
        # Return average losses
        if num_updates > 0:
            return {
                'policy_loss': total_policy_loss / num_updates,
                'value_loss': total_value_loss / num_updates,
                'entropy': total_entropy / num_updates
            }
        else:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
    
    def train(self):
        """
        Main training loop.
        """
        print("\n" + "="*60)
        print("Starting MAPPO Training")
        print("="*60)
        
        start_time = time.time()
        
        for episode in range(self.config.num_episodes):
            # Collect experience
            last_value, done, info = self.collect_experience()
            
            # Update policy
            losses = self.update_policy()
            
            # Log to TensorBoard
            if episode % self.config.log_interval == 0:
                self.log_metrics(episode, losses)
            
            # Print progress
            if episode % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.total_steps / elapsed
                
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                win_pct = np.mean(self.win_rate) * 100 if self.win_rate else 0
                
                print(f"Episode {episode:5d} | "
                      f"Reward: {avg_reward:7.2f} | "
                      f"Length: {avg_length:5.1f} | "
                      f"Win%: {win_pct:5.1f} | "
                      f"FPS: {fps:6.1f}")
            
            # Evaluate
            if episode % self.config.eval_interval == 0 and episode > 0:
                self.evaluate()
            
            # Save checkpoint
            if episode % self.config.save_interval == 0 and episode > 0:
                self.save_checkpoint(episode)
        
        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)
        
        # Save final model
        self.save_checkpoint(self.config.num_episodes)
        self.writer.close()
    
    def log_metrics(self, episode: int, losses: Dict):
        """Log metrics to TensorBoard"""
        if self.episode_rewards:
            self.writer.add_scalar('train/reward', np.mean(self.episode_rewards), episode)
            self.writer.add_scalar('train/episode_length', np.mean(self.episode_lengths), episode)
            self.writer.add_scalar('train/win_rate', np.mean(self.win_rate), episode)
        
        self.writer.add_scalar('train/policy_loss', losses['policy_loss'], episode)
        self.writer.add_scalar('train/value_loss', losses['value_loss'], episode)
        self.writer.add_scalar('train/entropy', losses['entropy'], episode)
        self.writer.add_scalar('train/total_steps', self.total_steps, episode)
    
    def evaluate(self, num_episodes: int = 10):
        """
        Evaluate current policy.
        
        Args:
            num_episodes: Number of episodes to evaluate
        """
        print("\nEvaluating...")
        
        eval_rewards = []
        eval_lengths = []
        eval_wins = []
        
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.config.max_steps):
                current_agent = self.env.current_player_idx
                action_mask = self.env.get_valid_action_mask()
                
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)
                mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.config.device)
                
                with torch.no_grad():
                    action, _, _ = self.networks[current_agent].get_action(
                        obs_tensor, mask_tensor, deterministic=True
                    )
                
                obs, reward, terminated, truncated, info = self.env.step(action.item())
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_wins.append(float(info.get('game_status', 0) == GameStatus.WON))
        
        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)
        win_rate = np.mean(eval_wins) * 100
        
        print(f"Eval: Reward={avg_reward:.2f}, Length={avg_length:.1f}, Win%={win_rate:.1f}")
        
        self.writer.add_scalar('eval/reward', avg_reward, self.total_episodes)
        self.writer.add_scalar('eval/episode_length', avg_length, self.total_episodes)
        self.writer.add_scalar('eval/win_rate', win_rate, self.total_episodes)
    
    def save_checkpoint(self, episode: int):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{episode}.pt')
        
        torch.save({
            'episode': episode,
            'model_state_dict': self.shared_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.shared_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_episodes = checkpoint.get('episode', 0)
        
        print(f"Checkpoint loaded: {checkpoint_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train MAPPO on Regicide')
    
    # Environment
    parser.add_argument('--num-players', type=int, default=2, help='Number of players')
    parser.add_argument('--max-steps', type=int, default=200, help='Max steps per episode')
    
    # Training
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer size')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='runs/regicide_mappo', help='Log directory')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--eval-interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--save-interval', type=int, default=500, help='Save interval')
    
    # Other
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None, help='Load checkpoint')
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Create config
    config = MAPPOConfig(
        num_players=args.num_players,
        max_steps=args.max_steps,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        lr_actor=args.lr,
        lr_critic=args.lr,
        gamma=args.gamma,
        hidden_dim=args.hidden_dim,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )
    
    # Create trainer
    trainer = MAPPOTrainer(config, log_dir=args.log_dir)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(trainer.total_episodes)


if __name__ == "__main__":
    main()
