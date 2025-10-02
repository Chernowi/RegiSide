"""
MAPPO Runner for Regicide Training

Implements Multi-Agent Proximal Policy Optimization (MAPPO) algorithm
using PyTorch for training on the Regicide environment.

This implementation is based on the standalone PyTorch MAPPO trainer,
integrated into the framework structure.
"""

import os
import sys
import time
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from .base_runner import BaseRunner
from compact_regicide_env import CompactRegicideEnv, GameStatus, CompactCard


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO training"""
    # Environment settings
    num_players: int = 2
    max_steps: int = 10000  # Large default since episodes end naturally

    # Visualization settings
    visualize_games: bool = True
    visualization_interval: int = 10

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


class MAPPORunner(BaseRunner):
    """
    MAPPO Runner for cooperative multi-agent training on Regicide.

    This implementation integrates the standalone PyTorch MAPPO trainer
    into the framework structure, providing centralized training with
    decentralized execution, GAE advantage estimation, and PPO clipping.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize MAPPO runner"""
        super().__init__(config)

        # Extract MAPPO-specific configuration
        self.mappo_config = config.get('mappo', {})
        self.env_config = config.get('env', {})

        # Create MAPPO config from framework config
        self.map_config = self._create_mappo_config()

        # Initialize training components
        self.env = None
        self.shared_network = None
        self.optimizer = None
        self.buffers = {}

        # Training statistics (matching the standalone implementation)
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.win_rate = deque(maxlen=100)
        self.enemies_defeated = deque(maxlen=100)
        self.game_progress = deque(maxlen=100)
        self.jacks_defeated = deque(maxlen=100)
        self.queens_defeated = deque(maxlen=100)
        self.kings_defeated = deque(maxlen=100)

        self.logger.info("Initialized MAPPO Runner with PyTorch backend")

    def _create_mappo_config(self) -> MAPPOConfig:
        """Create MAPPOConfig from framework configuration"""
        config = MAPPOConfig()

        # Environment settings
        config.num_players = self.env_config.get('num_players', 2)
        max_steps = self.env_config.get('max_episode_steps')
        config.max_steps = max_steps if max_steps is not None else 10000  # Large number if no limit

        # Network architecture
        network_config = self.mappo_config.get('network', {})
        config.hidden_dim = network_config.get('hidden_dims', [256])[0]  # Use first hidden dim
        config.num_layers = len(network_config.get('hidden_dims', [256]))

        # Training hyperparameters
        config.lr_actor = self.mappo_config.get('learning_rate', 3e-4)
        config.lr_critic = self.mappo_config.get('learning_rate', 1e-3)  # Same as actor for now
        config.gamma = self.mappo_config.get('gamma', 0.99)
        config.gae_lambda = self.mappo_config.get('gae_lambda', 0.95)
        config.clip_epsilon = self.mappo_config.get('clip_eps', 0.2)
        config.value_loss_coef = self.mappo_config.get('vf_coef', 0.5)
        config.entropy_coef = self.mappo_config.get('ent_coef', 0.01)
        config.max_grad_norm = self.mappo_config.get('max_grad_norm', 0.5)

        # PPO specific
        config.ppo_epochs = self.mappo_config.get('update_epochs', 4)
        config.num_minibatches = self.mappo_config.get('num_minibatches', 4)
        config.batch_size = self.mappo_config.get('num_steps', 256)  # Use num_steps as batch size

        # Training settings
        config.num_episodes = self.mappo_config.get('total_timesteps', 10000) // config.max_steps  # Convert timesteps to episodes
        config.eval_interval = self.config.get('logging', {}).get('eval_interval', 100)
        config.save_interval = self.config.get('logging', {}).get('save_interval', 500)
        config.log_interval = self.config.get('logging', {}).get('log_interval', 10)

        # Visualization settings
        eval_config = self.config.get('evaluation', {})
        config.visualize_games = eval_config.get('visualize_games', True)
        config.visualization_interval = eval_config.get('visualization_interval', 10)

        # Device
        device_config = self.config.get('device', 'auto')
        if device_config == 'auto':
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            config.device = device_config

        return config

    def setup(self):
        """Setup environment, networks, and optimizers"""
        # Create environment
        enemy_defeat_only = self.env_config.get('behavior', {}).get('enemy_defeat_only', False)
        self.env = CompactRegicideEnv(
            num_players=self.map_config.num_players,
            enemy_defeat_only=enemy_defeat_only
        )

        # Get dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.logger.info(f"Environment: {self.map_config.num_players} players")
        self.logger.info(f"Enemy defeat only mode: {enemy_defeat_only}")
        self.logger.info(f"Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
        self.logger.info(f"Device: {self.map_config.device}")

        # Create shared network for all agents (parameter sharing)
        self.shared_network = ActorCriticNetwork(
            self.obs_dim,
            self.action_dim,
            self.map_config.hidden_dim,
            self.map_config.num_layers
        ).to(self.map_config.device)

        # Create single optimizer for shared parameters
        self.optimizer = optim.Adam(
            self.shared_network.parameters(),
            lr=self.map_config.lr_actor
        )

        # Create experience buffers for each agent
        self.buffers = {agent_idx: ExperienceBuffer(self.map_config)
                       for agent_idx in range(self.map_config.num_players)}

        # Set random seed
        seed = self.config.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        self.logger.info("Setup complete")

    def collect_experience(self, num_steps: int = None):
        """
        Collect experience by running episodes.

        Args:
            num_steps: Maximum number of steps to collect (None = full episode)

        Returns:
            last_value: Value estimate for final state
            done: Whether episode ended
            info: Episode information
        """
        obs, info = self.env.reset()

        episode_reward = 0
        episode_length = 0

        for step in range(num_steps or self.map_config.max_steps):
            # Get current player
            current_agent = self.env.current_player_idx

            # Get action mask
            action_mask = self.env.get_valid_action_mask()

            # Convert to tensors
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.map_config.device)
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.map_config.device)

            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.shared_network.get_action(
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
            self.global_step += 1

            obs = next_obs

            if done:
                break

        # Compute final value for bootstrapping
        if not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.map_config.device)
            mask_tensor = torch.BoolTensor(self.env.get_valid_action_mask()).unsqueeze(0).to(self.map_config.device)

            with torch.no_grad():
                _, _, last_value = self.shared_network.get_action(
                    obs_tensor, mask_tensor
                )
            last_value = last_value.item()
        else:
            last_value = 0.0

        # Store episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.win_rate.append(float(info.get('game_status') == "WON"))
        self.enemies_defeated.append(12 - info.get('enemies_remaining', 12))
        self.game_progress.append((12 - info.get('enemies_remaining', 12)) / 12.0)
        
        jacks = 4 - sum(1 for cid in self.env.engine.castle_deck if CompactCard(cid).rank_idx == 10)
        queens = 4 - sum(1 for cid in self.env.engine.castle_deck if CompactCard(cid).rank_idx == 11)
        kings = 4 - sum(1 for cid in self.env.engine.castle_deck if CompactCard(cid).rank_idx == 12)
        self.jacks_defeated.append(jacks)
        self.queens_defeated.append(queens)
        self.kings_defeated.append(kings)
        
        self.episode_count += 1

        return last_value, done, info

    def update_policy(self):
        """
        Update policy using PPO.

        Returns:
            losses: Dictionary of loss values
        """
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        # Update policy for each agent (though they share parameters)
        for agent_idx in range(self.map_config.num_players):
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
                log_probs, values, entropy = self.shared_network.evaluate_actions(
                    batch['observations'],
                    batch['actions'],
                    batch['action_masks']
                )

                # Compute PPO loss
                ratio = torch.exp(log_probs - batch['old_log_probs'])

                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(ratio, 1.0 - self.map_config.clip_epsilon,
                                   1.0 + self.map_config.clip_epsilon) * batch['advantages']

                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch['returns'])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss +
                       self.map_config.value_loss_coef * value_loss +
                       self.map_config.entropy_coef * entropy_loss)

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.shared_network.parameters(),
                                        self.map_config.max_grad_norm)
                self.optimizer.step()

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

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Dictionary containing final training metrics
        """
        self.logger.info("Starting MAPPO training...")

        start_time = time.time()

        # Determine target timesteps from config (support underscores in yaml integer literals)
        raw_target = self.mappo_config.get('total_timesteps', None)
        if isinstance(raw_target, str):
            # Remove underscores if YAML loaded them as strings
            raw_target = int(raw_target.replace('_', ''))
        target_timesteps = int(raw_target) if raw_target is not None else None

        # Main loop: continue collecting experience and updating until we reach target_timesteps
        while True:
            # Stop condition
            if target_timesteps is not None and self.global_step >= target_timesteps:
                break

            # Collect experience (one episode or up to max steps)
            last_value, done, info = self.collect_experience()

            # Update policy
            losses = self.update_policy()

            # Update counter
            self.update_count += 1

            # Log metrics (use update_count as step indicator)
            if self.update_count % self.map_config.log_interval == 0:
                self._log_training_metrics(self.update_count, losses)

            # Periodic evaluation
            if self.should_evaluate():
                eval_metrics = self.evaluate(visualize=True)
                self.log_metrics(eval_metrics, self.episode_count)

            # Save checkpoint
            if self.should_save_checkpoint():
                self.save_checkpoint()

        # Final evaluation
        final_eval_metrics = self.evaluate(visualize=True)

        # Compute final metrics
        final_metrics = {
            'total_episodes': self.episode_count,
            'total_steps': self.global_step,
            'final_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'final_win_rate': np.mean(self.win_rate) if self.win_rate else 0,
            'final_game_progress': np.mean(self.game_progress) if self.game_progress else 0,
            'final_enemies_defeated': np.mean(self.enemies_defeated) if self.enemies_defeated else 0,
        }
        final_metrics.update({f"eval_{k}": v for k, v in final_eval_metrics.items()})

        training_time = time.time() - start_time
        final_metrics['training_time'] = training_time

        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        return final_metrics

    def _log_training_metrics(self, episode: int, losses: Dict):
        """Log training metrics to TensorBoard and console"""
        metrics = {}

        if self.episode_rewards:
            metrics['train/reward'] = np.mean(self.episode_rewards)
            metrics['train/episode_length'] = np.mean(self.episode_lengths)
            metrics['train/win_rate'] = np.mean(self.win_rate)

            # Progression metrics
            metrics['train/enemies_defeated'] = np.mean(self.enemies_defeated)
            metrics['train/game_progress'] = np.mean(self.game_progress)
            metrics['train/jacks_defeated'] = np.mean(self.jacks_defeated)
            metrics['train/queens_defeated'] = np.mean(self.queens_defeated)
            metrics['train/kings_defeated'] = np.mean(self.kings_defeated)

        metrics['train/policy_loss'] = losses['policy_loss']
        metrics['train/value_loss'] = losses['value_loss']
        metrics['train/entropy'] = losses['entropy']
        metrics['train/total_steps'] = self.global_step

        self.log_metrics(metrics, episode)

    def _get_card_name(self, card_id: int) -> str:
        """Convert card ID to human-readable name"""
        if card_id == 255:
            return "Empty"
        return str(CompactCard(card_id))

    def _get_player_hand_description(self, player_idx: int) -> str:
        """Get description of a player's hand"""
        hand = self.env.engine.hands[player_idx]
        card_names = [self._get_card_name(card_id) for card_id in hand if card_id != 255]
        return f"[{', '.join(card_names)}]" if card_names else "[empty]"

    def _decode_action_description(self, action: int, player_idx: int) -> str:
        """Convert action ID to human-readable description for the narrative."""
        hand = self.env.engine.hands[player_idx]
        env = self.env
        
        try:
            if action == env.ACTION_YIELD:
                return "yields their turn."

            elif env.ACTION_SINGLE_START <= action < env.ACTION_ACE_COMBO_START:
                slot = action - env.ACTION_SINGLE_START
                card_name = self._get_card_name(hand[slot])
                # Check if it's a Jester
                if CompactCard(hand[slot]).is_jester:
                    return f"plays the Jester, cancelling the enemy's immunity!"
                return f"attacks with a single card: {card_name}."

            elif env.ACTION_ACE_COMBO_START <= action < env.ACTION_PAIR_COMBO_START:
                linear_idx = action - env.ACTION_ACE_COMBO_START
                ace_slot = linear_idx // (env.MAX_HAND_SIZE - 1)
                other_offset = linear_idx % (env.MAX_HAND_SIZE - 1)
                other_slot = other_offset if other_offset < ace_slot else other_offset + 1
                ace_card = self._get_card_name(hand[ace_slot])
                other_card = self._get_card_name(hand[other_slot])
                return f"attacks with an Ace combo: {ace_card} and {other_card}."

            elif env.ACTION_PAIR_COMBO_START <= action < env.ACTION_TRIPLE_COMBO_START:
                idx = action - env.ACTION_PAIR_COMBO_START
                slots = self.env.engine._slot_pairs[idx]
                cards = ", ".join([self._get_card_name(hand[s]) for s in slots])
                return f"attacks with a pair: {cards}."

            elif env.ACTION_TRIPLE_COMBO_START <= action < env.ACTION_QUAD_COMBO_START:
                idx = action - env.ACTION_TRIPLE_COMBO_START
                slots = self.env.engine._slot_triples[idx]
                cards = ", ".join([self._get_card_name(hand[s]) for s in slots])
                return f"attacks with a triple: {cards}."

            elif env.ACTION_QUAD_COMBO_START <= action < env.ACTION_DEFEND_START:
                idx = action - env.ACTION_QUAD_COMBO_START
                slots = self.env.engine._slot_quads[idx]
                cards = ", ".join([self._get_card_name(hand[s]) for s in slots])
                return f"attacks with a quadruple combo: {cards}."

            elif env.ACTION_DEFEND_START <= action < env.ACTION_JESTER_CHOICE_START:
                # This case is handled in _get_action_outcome_summary for better context
                return "prepares to defend..."

            elif env.ACTION_JESTER_CHOICE_START <= action < env.TOTAL_ACTIONS:
                player_choice = action - env.ACTION_JESTER_CHOICE_START
                return f"chooses Player {player_choice + 1} to take the next turn."

        except (IndexError, KeyError):
            return f"performs an invalid or out-of-context action ({action})."

        return f"performs unknown action {action}."
    
    def _get_action_outcome_summary(self, info: Dict, old_state: Dict) -> List[str]:
        """Generates narrative lines based on the outcome of an action."""
        lines = []
        new_state = self.env.engine.get_state_dict()
        
        # Damage and Shield
        damage_dealt = info['action_result'].get('damage_dealt', 0)
        if damage_dealt > 0:
            old_hp = old_state['current_enemy_health']
            new_hp = new_state['current_enemy_health']
            lines.append(f"  ⚔️ Dealt {damage_dealt} damage! Enemy HP: {old_hp} -> {max(0, new_hp)}")
        
        shield_added = new_state['current_enemy_shield'] - old_state['current_enemy_shield']
        if shield_added > 0:
            lines.append(f"  ✨ Spades power adds {shield_added} shield. Total shield is now {new_state['current_enemy_shield']}.")

        # Enemy defeated
        if 'defeated_enemy_id' in info['action_result']:
            defeated_name = self._get_card_name(info['action_result']['defeated_enemy_id'])
            lines.append(f"  💥 The {defeated_name} is defeated!")
            if damage_dealt == old_state['current_enemy_health']:
                lines.append("  🌟 Perfect strike! The defeated enemy is returned to the Tavern Deck.")
        
        # Enemy counter-attack and defense
        if new_state['status'] == GameStatus.AWAITING_DEFENSE:
            incoming_damage = new_state['damage_to_defend']
            lines.append(f"  The enemy survives and counter-attacks for {old_state['current_enemy_attack']} (base) - {new_state['current_enemy_shield']} (shield) = {incoming_damage} damage.")
        
        if old_state['status'] == GameStatus.AWAITING_DEFENSE and new_state['status'] == GameStatus.IN_PROGRESS:
            # old_state may not include defend_player_idx (engine stores it as an attribute),
            # so read defensively from the state dict first and fall back to the engine attribute.
            defended_damage = old_state.get('damage_to_defend', getattr(self.env.engine, 'damage_to_defend', 0))
            defender_idx = old_state.get('defend_player_idx', getattr(self.env.engine, 'defend_player_idx', -1))
            if defender_idx is None or defender_idx < 0:
                lines.append(f"  🛡️ A player successfully defends against {defended_damage} damage.")
            else:
                lines.append(f"  🛡️ Player {defender_idx+1} successfully defends against {defended_damage} damage.")

        return lines

    def _log_evaluation_narrative(self, episode_num: int, narrative: str):
        """Save evaluation narrative to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_narrative_{timestamp}_ep{episode_num}.txt"
        
        runs_dir = Path("runs")
        narratives_dir = runs_dir / "narratives"
        narratives_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = narratives_dir / filename

        with open(filepath, 'w') as f:
            f.write(narrative)

        self.logger.info(f"Saved evaluation narrative to {filepath}")

    def evaluate(self, num_episodes: Optional[int] = None, visualize: bool = False) -> Dict[str, Any]:
        """
        Evaluate current policy with improved narrative generation.
        """
        if num_episodes is None:
            num_episodes = self.config.get('evaluation', {}).get('num_episodes', 10)

        self.logger.info(f"Evaluating policy on {num_episodes} episodes...")

        eval_rewards, eval_lengths, eval_wins, eval_enemies_defeated, eval_progress = [], [], [], [], []

        for episode_idx in range(num_episodes):
            obs, info = self.env.reset()
            should_visualize = visualize and episode_idx == 0
            narrative_lines = []

            if should_visualize:
                narrative_lines.append(f"=== Evaluation Episode {self.episode_count + episode_idx} ===")
                narrative_lines.append("The adventurers gather to face the corrupted royals.\n")

            episode_reward, episode_length = 0, 0
            
            for step in range(self.map_config.max_steps):
                current_agent = self.env.current_player_idx
                old_state = self.env.engine.get_state_dict()
                
                if should_visualize:
                    enemy_name = self._get_card_name(old_state['current_enemy_id'])
                    hand_desc = self._get_player_hand_description(current_agent)
                    narrative_lines.append(f"--- Turn {step + 1}: Player {current_agent + 1}'s move ---")
                    narrative_lines.append(f"Facing: {enemy_name} (HP: {old_state['current_enemy_health']}, Shield: {old_state['current_enemy_shield']})")
                    narrative_lines.append(f"Hand: {hand_desc}")

                action_mask = self.env.get_valid_action_mask()
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.map_config.device)
                mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.map_config.device)

                with torch.no_grad():
                    action, _, _ = self.shared_network.get_action(obs_tensor, mask_tensor, deterministic=True)
                
                action_item = action.item()

                if should_visualize:
                    action_desc = self._decode_action_description(action_item, current_agent)
                    narrative_lines.append(f"Action: Player {current_agent + 1} {action_desc}")
                
                obs, reward, terminated, truncated, info = self.env.step(action_item)

                if should_visualize:
                    outcome_lines = self._get_action_outcome_summary(info, old_state)
                    narrative_lines.extend(outcome_lines)
                    
                    new_enemy_id = self.env.engine.current_enemy_id
                    if old_state['current_enemy_id'] != new_enemy_id and new_enemy_id != 255:
                        new_enemy_name = self._get_card_name(new_enemy_id)
                        new_enemy_health = self.env.engine.current_enemy_health
                        narrative_lines.append(f"\nA new foe appears: The {new_enemy_name} (HP: {new_enemy_health})!")
                    
                    narrative_lines.append("") # Blank line for readability

                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    if should_visualize:
                        if info.get('game_status') == "WON":
                            narrative_lines.append("🎉 VICTORY! The last king has fallen and the realm is saved!")
                        elif info.get('game_status') == "LOST":
                            narrative_lines.append("💀 DEFEAT! The adventurers have been overwhelmed by the darkness.")
                    break

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_wins.append(float(info.get('game_status') == "WON"))
            enemies_defeated = 12 - info.get('enemies_remaining', 12)
            eval_enemies_defeated.append(enemies_defeated)
            eval_progress.append(enemies_defeated / 12.0)
            
            if should_visualize and narrative_lines:
                narrative = "\n".join(narrative_lines)
                self._log_evaluation_narrative(self.episode_count + episode_idx, narrative)

        eval_metrics = {
            'eval/reward': np.mean(eval_rewards),
            'eval/episode_length': np.mean(eval_lengths),
            'eval/win_rate': np.mean(eval_wins),
            'eval/enemies_defeated': np.mean(eval_enemies_defeated),
            'eval/game_progress': np.mean(eval_progress),
        }

        self.logger.info(f"Evaluation: Reward={eval_metrics['eval/reward']:.2f}, "
                        f"Win%={eval_metrics['eval/win_rate'] * 100:.1f}")

        return eval_metrics


    def _save_checkpoint_impl(self, checkpoint_data: Dict[str, Any], filepath: Path):
        """Save model checkpoint"""
        checkpoint = {
            **checkpoint_data,
            'model_state_dict': self.shared_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'map_config': self.map_config,
        }

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)

        self.shared_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore training state
        self.global_step = checkpoint.get('global_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.update_count = checkpoint.get('update_count', 0)
        self.best_metric = checkpoint.get('best_metric')

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")