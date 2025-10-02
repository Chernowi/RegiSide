#!/usr/bin/env python3
"""
MAPPO Training Script for JaxMARL Regicide Environment

This script trains Multi-Agent Proximal Policy Optimization (MAPPO) agents
on the cooperative Regicide card game environment using JaxMARL.

Usage:
    python train_mappo_regicide.py --num_players 4 --total_timesteps 10000000

Requirements:
    - JAX/JaxLib 
    - JaxMARL
    - Chex, Flax, Optax
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Tuple
import json
import pickle
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import wandb

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from jaxmarl_regicide import JaxMARLRegicide
    print("✅ Successfully imported JaxMARL Regicide environment")
except ImportError as e:
    print(f"❌ Failed to import JaxMARL Regicide: {e}")
    print("Make sure jaxmarl_regicide.py is in the src/ directory")
    sys.exit(1)

# Try to import JaxMARL
try:
    from jaxmarl.algorithms.mappo import MAPPO
    from jaxmarl.environments.multi_agent_env import MultiAgentEnv
    JAXMARL_AVAILABLE = True
    print("✅ JaxMARL available - using official MAPPO implementation")
except ImportError:
    JAXMARL_AVAILABLE = False
    print("⚠️ JaxMARL not available - using custom MAPPO implementation")

# If JaxMARL is not available, provide a simplified MAPPO implementation
if not JAXMARL_AVAILABLE:
    from flax import struct
    from flax.training.train_state import TrainState
    
    class MAPPOConfig:
        """MAPPO Configuration"""
        def __init__(self, **kwargs):
            # Environment
            self.env_name = kwargs.get('env_name', 'regicide')
            self.num_envs = kwargs.get('num_envs', 32)
            self.num_steps = kwargs.get('num_steps', 128)
            
            # Training
            self.total_timesteps = kwargs.get('total_timesteps', 10_000_000)
            self.learning_rate = kwargs.get('learning_rate', 3e-4)
            self.anneal_lr = kwargs.get('anneal_lr', True)
            
            # PPO specific
            self.gamma = kwargs.get('gamma', 0.99)
            self.gae_lambda = kwargs.get('gae_lambda', 0.95)
            self.clip_eps = kwargs.get('clip_eps', 0.2)
            self.ent_coef = kwargs.get('ent_coef', 0.01)
            self.vf_coef = kwargs.get('vf_coef', 0.5)
            self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
            
            # Network architecture
            self.hidden_size = kwargs.get('hidden_size', 256)
            self.num_layers = kwargs.get('num_layers', 2)
            
            # Update parameters
            self.update_epochs = kwargs.get('update_epochs', 4)
            self.num_minibatches = kwargs.get('num_minibatches', 4)
            
            # Logging
            self.log_interval = kwargs.get('log_interval', 10)
            self.eval_interval = kwargs.get('eval_interval', 100)
            
    class ActorCritic(nn.Module):
        """Actor-Critic network for MAPPO"""
        action_dim: int
        hidden_size: int = 256
        num_layers: int = 2
        
        def setup(self):
            # Shared feature extraction
            layers = []
            for _ in range(self.num_layers):
                layers.extend([
                    nn.Dense(self.hidden_size),
                    nn.relu
                ])
            self.shared_net = nn.Sequential(layers)
            
            # Actor head
            self.actor = nn.Dense(self.action_dim)
            
            # Critic head  
            self.critic = nn.Dense(1)
        
        def __call__(self, x):
            # Shared features
            features = self.shared_net(x)
            
            # Actor logits
            logits = self.actor(features)
            
            # State value
            value = self.critic(features)
            
            return logits, value.squeeze(-1)


class MAPPOTrainer:
    """MAPPO Trainer for Regicide Environment"""
    
    def __init__(self, config: MAPPOConfig, env: JaxMARLRegicide):
        self.config = config
        self.env = env
        
        # Calculate derived parameters
        self.num_agents = len(env.agents)
        self.obs_dim = env.observation_spaces[env.agents[0]].shape[0]
        self.action_dim = env.action_spaces[env.agents[0]].n
        
        # Training parameters
        self.num_updates = config.total_timesteps // (config.num_envs * config.num_steps)
        self.minibatch_size = (config.num_envs * config.num_steps) // config.num_minibatches
        
        print(f"Training Configuration:")
        print(f"  Environment: {config.env_name}")
        print(f"  Agents: {self.num_agents}")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Parallel envs: {config.num_envs}")
        print(f"  Steps per update: {config.num_steps}")
        print(f"  Total updates: {self.num_updates}")
        print(f"  Minibatch size: {self.minibatch_size}")
        
        # Initialize networks and optimizers
        self._init_networks()
        
    def _init_networks(self):
        """Initialize actor-critic networks for all agents"""
        key = jax.random.PRNGKey(42)
        
        # Create dummy observation
        dummy_obs = jnp.zeros((1, self.obs_dim))
        
        # Initialize networks for each agent
        self.networks = {}
        self.train_states = {}
        
        for agent in self.env.agents:
            # Initialize network
            network = ActorCritic(
                action_dim=self.action_dim,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers
            )
            
            key, init_key = jax.random.split(key)
            params = network.init(init_key, dummy_obs)
            
            # Create optimizer
            if self.config.anneal_lr:
                schedule = optax.linear_schedule(
                    init_value=self.config.learning_rate,
                    end_value=0.0,
                    transition_steps=self.num_updates
                )
                optimizer = optax.chain(
                    optax.clip_by_global_norm(self.config.max_grad_norm),
                    optax.adam(schedule, eps=1e-5)
                )
            else:
                optimizer = optax.chain(
                    optax.clip_by_global_norm(self.config.max_grad_norm),
                    optax.adam(self.config.learning_rate, eps=1e-5)
                )
            
            # Create training state
            train_state_obj = train_state.TrainState.create(
                apply_fn=network.apply,
                params=params,
                tx=optimizer
            )
            
            self.networks[agent] = network
            self.train_states[agent] = train_state_obj
    
    def train(self, num_seeds: int = 1, use_wandb: bool = False):
        """Main training loop"""
        
        if use_wandb:
            wandb.init(
                project="regicide-mappo",
                config=vars(self.config),
                name=f"mappo_regicide_{self.config.num_envs}envs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        all_results = []
        
        for seed in range(num_seeds):
            print(f"\n🚀 Starting training run {seed + 1}/{num_seeds}")
            
            results = self._train_single_run(seed, use_wandb)
            all_results.append(results)
            
            if use_wandb:
                wandb.log({"run": seed, "final_return": results["returns"][-1]})
        
        if use_wandb:
            wandb.finish()
        
        return all_results
    
    def _train_single_run(self, seed: int, use_wandb: bool = False):
        """Single training run"""
        key = jax.random.PRNGKey(seed)
        
        # Initialize parallel environments
        key, *env_keys = jax.random.split(key, self.config.num_envs + 1)
        env_keys = jnp.array(env_keys)
        
        # Reset environments
        reset_fn = jax.vmap(self.env.reset)
        obs_batch, env_states = reset_fn(env_keys)
        
        # Training metrics
        returns = []
        episode_lengths = []
        win_rates = []
        policy_losses = []
        value_losses = []
        entropies = []
        
        # Training loop
        for update in range(self.num_updates):
            update_start = time.time()
            
            # Collect rollout
            key, rollout_key = jax.random.split(key)
            rollout_data, env_states = self._collect_rollout(
                rollout_key, obs_batch, env_states
            )
            
            # Update policy
            key, update_key = jax.random.split(key)
            train_metrics = self._update_policy(update_key, rollout_data)
            
            # Get new observations
            obs_batch = rollout_data["obs"][:, -1]  # Last step observations
            
            # Logging
            if update % self.config.log_interval == 0:
                update_time = time.time() - update_start
                fps = (self.config.num_envs * self.config.num_steps) / update_time
                
                # Calculate episode metrics
                episode_return = jnp.mean(rollout_data["rewards"].sum(axis=1))
                episode_length = self.config.num_steps
                
                returns.append(float(episode_return))
                episode_lengths.append(float(episode_length))
                policy_losses.append(float(train_metrics["policy_loss"]))
                value_losses.append(float(train_metrics["value_loss"]))
                entropies.append(float(train_metrics["entropy"]))
                
                print(f"Update {update}/{self.num_updates}")
                print(f"  Return: {episode_return:.2f}")
                print(f"  Policy Loss: {train_metrics['policy_loss']:.4f}")
                print(f"  Value Loss: {train_metrics['value_loss']:.4f}")
                print(f"  Entropy: {train_metrics['entropy']:.4f}")
                print(f"  FPS: {fps:.0f}")
                
                if use_wandb:
                    wandb.log({
                        "update": update,
                        "return": episode_return,
                        "episode_length": episode_length,
                        "policy_loss": train_metrics["policy_loss"],
                        "value_loss": train_metrics["value_loss"],
                        "entropy": train_metrics["entropy"],
                        "fps": fps
                    })
        
        return {
            "returns": returns,
            "episode_lengths": episode_lengths,
            "win_rates": win_rates,
            "policy_losses": policy_losses,
            "value_losses": value_losses,
            "entropies": entropies
        }
    
    def _collect_rollout(self, key, obs_batch, env_states):
        """Collect rollout data"""
        batch_size = self.config.num_envs
        num_steps = self.config.num_steps
        
        # Storage for rollout
        obs_storage = jnp.zeros((batch_size, num_steps + 1, self.obs_dim))
        action_storage = jnp.zeros((batch_size, num_steps), dtype=jnp.int32)
        reward_storage = jnp.zeros((batch_size, num_steps))
        done_storage = jnp.zeros((batch_size, num_steps), dtype=jnp.bool_)
        value_storage = jnp.zeros((batch_size, num_steps + 1))
        log_prob_storage = jnp.zeros((batch_size, num_steps))
        
        # Initialize first observations
        current_obs = obs_batch[self.env.agents[0]]  # Use first agent's obs as shared
        obs_storage = obs_storage.at[:, 0].set(current_obs)
        
        # Get initial values
        _, values = self._get_action_and_value(current_obs, key, evaluate=False)
        value_storage = value_storage.at[:, 0].set(values)
        
        # Collect steps
        for step in range(num_steps):
            key, step_key = jax.random.split(key)
            
            # Get actions and values for all environments
            actions, values = self._get_action_and_value(current_obs, step_key)
            log_probs = self._get_log_probs(current_obs, actions, step_key)
            
            # Create action dict for environment
            action_dict = {agent: actions for agent in self.env.agents}
            
            # Step environments
            step_keys = jax.random.split(step_key, batch_size)
            step_fn = jax.vmap(self.env.step_env)
            
            obs_batch, env_states, reward_batch, done_batch, _ = step_fn(
                step_keys, env_states, action_dict
            )
            
            # Store data
            action_storage = action_storage.at[:, step].set(actions)
            reward_storage = reward_storage.at[:, step].set(reward_batch[self.env.agents[0]])
            done_storage = done_storage.at[:, step].set(done_batch["__all__"])
            value_storage = value_storage.at[:, step].set(values)
            log_prob_storage = log_prob_storage.at[:, step].set(log_probs)
            
            # Update observations
            current_obs = obs_batch[self.env.agents[0]]
            obs_storage = obs_storage.at[:, step + 1].set(current_obs)
        
        # Get final values
        _, final_values = self._get_action_and_value(current_obs, key, evaluate=False)
        value_storage = value_storage.at[:, -1].set(final_values)
        
        return {
            "obs": obs_storage,
            "actions": action_storage,
            "rewards": reward_storage,
            "dones": done_storage,
            "values": value_storage,
            "log_probs": log_prob_storage
        }, env_states
    
    def _get_action_and_value(self, obs, key, evaluate=True):
        """Get actions and values from policy"""
        # Use first agent's policy (shared policy)
        agent = self.env.agents[0]
        
        logits, values = self.train_states[agent].apply_fn(
            self.train_states[agent].params, obs
        )
        
        if evaluate:
            actions = jnp.argmax(logits, axis=-1)
        else:
            actions = jax.random.categorical(key, logits, axis=-1)
        
        return actions, values
    
    def _get_log_probs(self, obs, actions, key):
        """Get log probabilities of actions"""
        agent = self.env.agents[0]
        
        logits, _ = self.train_states[agent].apply_fn(
            self.train_states[agent].params, obs
        )
        
        log_probs = jax.nn.log_softmax(logits)
        action_log_probs = jnp.take_along_axis(
            log_probs, actions[..., None], axis=-1
        ).squeeze(-1)
        
        return action_log_probs
    
    def _update_policy(self, key, rollout_data):
        """Update policy using PPO"""
        # Calculate advantages using GAE
        advantages, returns = self._calculate_gae(
            rollout_data["rewards"],
            rollout_data["values"],
            rollout_data["dones"]
        )
        
        # Flatten data for minibatch updates
        batch_size = self.config.num_envs * self.config.num_steps
        
        obs_flat = rollout_data["obs"][:, :-1].reshape(batch_size, -1)
        actions_flat = rollout_data["actions"].flatten()
        log_probs_old_flat = rollout_data["log_probs"].flatten()
        advantages_flat = advantages.flatten()
        returns_flat = returns.flatten()
        
        # Normalize advantages
        advantages_flat = (advantages_flat - jnp.mean(advantages_flat)) / (jnp.std(advantages_flat) + 1e-8)
        
        # Update for each agent (shared policy in this case)
        agent = self.env.agents[0]
        
        def ppo_update_step(train_state_obj, batch_data):
            obs_batch, actions_batch, log_probs_old_batch, advantages_batch, returns_batch = batch_data
            
            def ppo_loss(params):
                logits, values = train_state_obj.apply_fn(params, obs_batch)
                
                # Policy loss
                log_probs = jax.nn.log_softmax(logits)
                action_log_probs = jnp.take_along_axis(
                    log_probs, actions_batch[..., None], axis=-1
                ).squeeze(-1)
                
                ratio = jnp.exp(action_log_probs - log_probs_old_batch)
                surr1 = ratio * advantages_batch
                surr2 = jnp.clip(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * advantages_batch
                policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
                
                # Value loss
                value_loss = jnp.mean((values - returns_batch) ** 2)
                
                # Entropy loss
                entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1))
                
                total_loss = (
                    policy_loss + 
                    self.config.vf_coef * value_loss - 
                    self.config.ent_coef * entropy
                )
                
                return total_loss, (policy_loss, value_loss, entropy)
            
            grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
            (loss, (policy_loss, value_loss, entropy)), grads = grad_fn(train_state_obj.params)
            
            # Update parameters
            train_state_obj = train_state_obj.apply_gradients(grads=grads)
            
            return train_state_obj, {
                "policy_loss": policy_loss,
                "value_loss": value_loss, 
                "entropy": entropy,
                "total_loss": loss
            }
        
        # Perform multiple update epochs
        metrics = {}
        for epoch in range(self.config.update_epochs):
            # Shuffle data
            key, shuffle_key = jax.random.split(key)
            permutation = jax.random.permutation(shuffle_key, batch_size)
            
            obs_shuffled = obs_flat[permutation]
            actions_shuffled = actions_flat[permutation]
            log_probs_shuffled = log_probs_old_flat[permutation]
            advantages_shuffled = advantages_flat[permutation]
            returns_shuffled = returns_flat[permutation]
            
            # Update in minibatches
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                
                batch_data = (
                    obs_shuffled[start:end],
                    actions_shuffled[start:end],
                    log_probs_shuffled[start:end],
                    advantages_shuffled[start:end],
                    returns_shuffled[start:end]
                )
                
                self.train_states[agent], batch_metrics = ppo_update_step(
                    self.train_states[agent], batch_data
                )
                
                metrics = batch_metrics  # Keep last batch metrics
        
        return metrics
    
    def _calculate_gae(self, rewards, values, dones):
        """Calculate Generalized Advantage Estimation"""
        advantages = jnp.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(self.config.num_steps)):
            if t == self.config.num_steps - 1:
                next_non_terminal = 1.0 - dones[:, t]
                next_values = values[:, t + 1]
            else:
                next_non_terminal = 1.0 - dones[:, t]
                next_values = values[:, t + 1]
            
            delta = rewards[:, t] + self.config.gamma * next_values * next_non_terminal - values[:, t]
            advantages = advantages.at[:, t].set(
                delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
            )
            last_gae = advantages[:, t]
        
        returns = advantages + values[:, :-1]
        
        return advantages, returns


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train MAPPO on Regicide")
    
    # Environment args
    parser.add_argument("--num_players", type=int, default=4, help="Number of players (1-4)")
    parser.add_argument("--max_episode_steps", type=int, default=500, help="Max steps per episode")
    
    # Training args
    parser.add_argument("--total_timesteps", type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--num_steps", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    
    # PPO args
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient")
    
    # Network args
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of hidden layers")
    
    # Update args
    parser.add_argument("--update_epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument("--num_minibatches", type=int, default=4, help="Number of minibatches")
    
    # Logging args
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="regicide-mappo", help="W&B project name")
    
    # Experiment args
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of seeds to run")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save results")
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    print("🎮 MAPPO Training for Regicide Environment")
    print("=" * 50)
    
    # Create environment
    env = JaxMARLRegicide(
        num_players=args.num_players,
        max_steps=args.max_episode_steps
    )
    
    print(f"Environment: {env.name}")
    print(f"Players: {args.num_players}")
    print(f"Observation space: {env.observation_spaces[env.agents[0]].shape}")
    print(f"Action space: {env.action_spaces[env.agents[0]].n}")
    
    # Create config
    config = MAPPOConfig(
        env_name="regicide",
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval
    )
    
    # Create trainer
    trainer = MAPPOTrainer(config, env)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = save_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\nResults will be saved to: {save_dir}")
    
    # Start training
    print(f"\n🚀 Starting MAPPO training...")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Steps per rollout: {args.num_steps}")
    
    if args.use_wandb:
        print(f"Logging to W&B project: {args.wandb_project}")
    
    start_time = time.time()
    
    # Train
    results = trainer.train(
        num_seeds=args.num_seeds,
        use_wandb=args.use_wandb
    )
    
    training_time = time.time() - start_time
    
    # Save results
    results_path = save_dir / "results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✅ Training completed!")
    print(f"Training time: {training_time/3600:.2f} hours")
    print(f"Results saved to: {results_path}")
    
    # Print final statistics
    if results:
        final_returns = [run["returns"][-1] if run["returns"] else 0 for run in results]
        print(f"\nFinal Results ({args.num_seeds} seeds):")
        print(f"  Mean return: {np.mean(final_returns):.2f} ± {np.std(final_returns):.2f}")
        print(f"  Best return: {np.max(final_returns):.2f}")
        print(f"  Worst return: {np.min(final_returns):.2f}")


if __name__ == "__main__":
    main()