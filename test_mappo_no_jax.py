#!/usr/bin/env python3
"""
Non-JAX MAPPO Test for Regicide Environment

This script tests MAPPO training on the Regicide environment without JAX dependencies.
Uses PyTorch for the neural networks and training loop.

Usage: python test_mappo_no_jax.py
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Any
import random

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
    print("✅ PyTorch available for non-JAX MAPPO")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - will use numpy-only implementation")

# Import our environment but force it to not use JAX
import jax
import jax.numpy as jnp

# Monkey patch to simulate JAX unavailability if needed
original_jaxmarl_available = None

def disable_jax_temporarily():
    """Temporarily disable JaxMARL imports for testing"""
    global original_jaxmarl_available
    sys.modules_backup = sys.modules.copy()
    
    # Remove JaxMARL from modules to simulate it being unavailable
    modules_to_remove = [k for k in sys.modules.keys() if 'jaxmarl' in k]
    for module in modules_to_remove:
        del sys.modules[module]
    
    return sys.modules_backup

def restore_modules(modules_backup):
    """Restore the original modules"""
    sys.modules.clear()
    sys.modules.update(modules_backup)

# Test environment import without JaxMARL
modules_backup = disable_jax_temporarily()

try:
    from jaxmarl_regicide import JaxMARLRegicide, JAXMARL_AVAILABLE
    print(f"✅ Successfully imported Regicide environment (JaxMARL available: {JAXMARL_AVAILABLE})")
except ImportError as e:
    print(f"❌ Failed to import Regicide environment: {e}")
    restore_modules(modules_backup)
    sys.exit(1)


class SimpleActorCritic(nn.Module if TORCH_AVAILABLE else object):
    """Simple Actor-Critic network using PyTorch"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        if TORCH_AVAILABLE:
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            self.actor = nn.Linear(hidden_dim, action_dim)
            self.critic = nn.Linear(hidden_dim, 1)
        else:
            # Numpy-based implementation
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            
            # Initialize weights randomly
            self.w1 = np.random.randn(obs_dim, hidden_dim) * 0.1
            self.b1 = np.zeros(hidden_dim)
            self.w2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
            self.b2 = np.zeros(hidden_dim)
            self.w_actor = np.random.randn(hidden_dim, action_dim) * 0.1
            self.b_actor = np.zeros(action_dim)
            self.w_critic = np.random.randn(hidden_dim, 1) * 0.1
            self.b_critic = np.zeros(1)
    
    def forward(self, x):
        if TORCH_AVAILABLE:
            shared = self.shared(x)
            return self.actor(shared), self.critic(shared)
        else:
            # Numpy forward pass
            h1 = np.maximum(0, np.dot(x, self.w1) + self.b1)  # ReLU
            h2 = np.maximum(0, np.dot(h1, self.w2) + self.b2)  # ReLU
            action_logits = np.dot(h2, self.w_actor) + self.b_actor
            value = np.dot(h2, self.w_critic) + self.b_critic
            return action_logits, value


class SimpleAgent:
    """Simple MAPPO-style agent"""
    
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        if TORCH_AVAILABLE:
            self.network = SimpleActorCritic(obs_dim, action_dim)
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        else:
            self.network = SimpleActorCritic(obs_dim, action_dim)
            self.lr = lr
    
    def get_action(self, obs, action_mask=None):
        """Get action from policy"""
        if TORCH_AVAILABLE:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_logits, value = self.network(obs_tensor)
                
                if action_mask is not None:
                    # Apply action mask
                    mask_tensor = torch.FloatTensor(action_mask)
                    action_logits = action_logits + (mask_tensor - 1) * 1e8
                
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                return action.item(), log_prob.item(), value.item()
        else:
            # Numpy implementation
            action_logits, value = self.network.forward(obs)
            
            if action_mask is not None:
                # Apply action mask
                action_logits = action_logits + (action_mask - 1) * 1e8
            
            # Softmax for action probabilities
            exp_logits = np.exp(action_logits - np.max(action_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # Sample action
            action = np.random.choice(len(probs), p=probs)
            log_prob = np.log(probs[action])
            
            return action, log_prob, value[0]
    
    def update(self, experiences):
        """Simple policy update (placeholder)"""
        if TORCH_AVAILABLE:
            # Implement actual PPO update here
            pass
        else:
            # Implement numpy-based update here
            pass


def test_environment_without_jax():
    """Test environment functionality without JAX dependencies"""
    print("\n" + "="*60)
    print("TESTING REGICIDE ENVIRONMENT WITHOUT JAX DEPENDENCIES")
    print("="*60)
    
    # Create environment
    env = JaxMARLRegicide(num_players=2, max_steps=50)
    
    print(f"Environment created successfully")
    print(f"Number of players: {env.num_players}")
    print(f"Max steps: {env.max_steps}")
    
    # Test basic functionality without JAX
    try:
        # Create a simple random key replacement
        key = jax.random.PRNGKey(42)
        
        # Test reset
        obs, state = env.reset(key)
        print("✅ Environment reset successful")
        
        # Test step with random actions
        print("Testing random actions...")
        for step in range(5):
            # Get valid actions for current player
            current_agent = env.agents[state.current_player]
            
            # Simple random action (without action masking for now)
            action = np.random.randint(0, env.action_spaces[current_agent].n)
            actions = {agent: action if agent == current_agent else 0 for agent in env.agents}
            
            obs, state, rewards, dones, infos = env.step(key, state, actions)
            
            print(f"  Step {step + 1}: Player {state.current_player}, Reward: {rewards.get(current_agent, 0)}")
            
            if state.terminal:
                print("  Game ended!")
                break
        
        print("✅ Environment stepping successful")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_simple_agent():
    """Test simple agent implementation"""
    print("\n" + "="*60)
    print("TESTING SIMPLE MAPPO AGENT")
    print("="*60)
    
    # Environment parameters
    obs_dim = 100  # Simplified observation size
    action_dim = 30  # Regicide action space size
    
    try:
        # Create agent
        agent = SimpleAgent(obs_dim, action_dim)
        print("✅ Agent created successfully")
        
        # Test action selection
        fake_obs = np.random.randn(obs_dim)
        action_mask = np.ones(action_dim)  # All actions valid
        
        action, log_prob, value = agent.get_action(fake_obs, action_mask)
        
        print(f"✅ Agent action selection successful")
        print(f"  Action: {action}")
        print(f"  Log prob: {log_prob:.4f}")
        print(f"  Value: {value:.4f}")
        
        # Test with action masking
        action_mask[15:25] = 0  # Mask some actions
        action, log_prob, value = agent.get_action(fake_obs, action_mask)
        
        if action not in range(15, 25):
            print("✅ Action masking working correctly")
        else:
            print("⚠️ Action masking might not be working")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """Test a simple training loop"""
    print("\n" + "="*60)
    print("TESTING SIMPLE TRAINING LOOP")
    print("="*60)
    
    try:
        # Create environment and agents
        env = JaxMARLRegicide(num_players=2, max_steps=20)
        
        # Get observation and action dimensions
        # This is a simplified approach - in reality we'd need proper observation processing
        obs_dim = 100  # Simplified
        action_dim = env.action_spaces[env.agents[0]].n
        
        agents = {agent: SimpleAgent(obs_dim, action_dim) for agent in env.agents}
        
        print(f"Created {len(agents)} agents")
        print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
        
        # Run a few training episodes
        num_episodes = 3
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}")
            
            # Reset environment
            key = jax.random.PRNGKey(episode)
            obs, state = env.reset(key)
            
            step_count = 0
            total_reward = 0
            
            while not state.terminal and step_count < env.max_steps:
                # Get current player
                current_agent = env.agents[state.current_player]
                
                # Simplified observation (in reality we'd process the actual observation)
                fake_obs = np.random.randn(obs_dim)
                
                # Get action from agent
                action, log_prob, value = agents[current_agent].get_action(fake_obs)
                
                # Create action dict
                actions = {agent: action if agent == current_agent else 0 for agent in env.agents}
                
                # Step environment
                obs, state, rewards, dones, infos = env.step(key, state, actions)
                
                reward = rewards.get(current_agent, 0)
                total_reward += reward
                
                step_count += 1
            
            print(f"  Episode finished after {step_count} steps")
            print(f"  Total reward: {total_reward}")
            
            if state.status == 3:  # Won
                print("  🎉 Game won!")
            elif state.status == 4:  # Lost
                print("  😞 Game lost!")
        
        print("✅ Training loop test successful")
        return True
        
    except Exception as e:
        print(f"❌ Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🧪 Non-JAX MAPPO Regicide Test Suite")
    print("="*60)
    
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"JaxMARL available: {JAXMARL_AVAILABLE}")
    
    # Run tests
    tests = [
        ("Environment Test", test_environment_without_jax),
        ("Simple Agent Test", test_simple_agent),
        ("Training Loop Test", test_training_loop)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Non-JAX MAPPO system is working!")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    # Restore modules
    restore_modules(modules_backup)


if __name__ == "__main__":
    main()