#!/usr/bin/env python3
"""
Quick test script for MAPPO PyTorch training setup.

This script verifies that:
1. PyTorch is installed correctly
2. The environment can be created
3. The network can process observations
4. Training loop basics work

Usage: python test_mappo_pytorch_setup.py
"""

import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from compact_regicide_env import CompactRegicideEnv, GameStatus


def test_pytorch_available():
    """Test PyTorch availability"""
    print("Testing PyTorch availability...")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print("  ✅ PyTorch is available\n")


def test_environment_creation():
    """Test environment creation"""
    print("Testing environment creation...")
    env = CompactRegicideEnv(num_players=2)
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print("  ✅ Environment created successfully\n")
    return env


def test_environment_reset():
    """Test environment reset"""
    print("Testing environment reset...")
    env = CompactRegicideEnv(num_players=2)
    obs, info = env.reset(seed=42)
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation dtype: {obs.dtype}")
    print(f"  Info keys: {list(info.keys())}")
    print(f"  Valid actions: {len(info['valid_actions'])}")
    print("  ✅ Environment reset successful\n")
    return env, obs, info


def test_action_masking():
    """Test action masking"""
    print("Testing action masking...")
    env = CompactRegicideEnv(num_players=2)
    obs, info = env.reset(seed=42)
    
    action_mask = env.get_valid_action_mask()
    print(f"  Action mask shape: {action_mask.shape}")
    print(f"  Action mask dtype: {action_mask.dtype}")
    print(f"  Valid actions: {np.sum(action_mask)}/{len(action_mask)}")
    print(f"  Valid action indices: {np.where(action_mask)[0][:5].tolist()}...")
    print("  ✅ Action masking working\n")


def test_network_creation():
    """Test network creation and forward pass"""
    print("Testing network creation...")
    
    # Import from training script
    sys.path.append(os.path.dirname(__file__))
    from train_mappo_pytorch import ActorCriticNetwork
    
    obs_dim = 48
    action_dim = 30
    hidden_dim = 128
    
    network = ActorCriticNetwork(obs_dim, action_dim, hidden_dim)
    print(f"  Network created with {sum(p.numel() for p in network.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    obs_tensor = torch.randn(batch_size, obs_dim)
    action_mask = torch.ones(batch_size, action_dim, dtype=torch.bool)
    
    action_logits, values = network(obs_tensor, action_mask)
    
    print(f"  Action logits shape: {action_logits.shape}")
    print(f"  Values shape: {values.shape}")
    print("  ✅ Network forward pass successful\n")


def test_action_sampling():
    """Test action sampling with masking"""
    print("Testing action sampling...")
    
    from train_mappo_pytorch import ActorCriticNetwork
    
    obs_dim = 48
    action_dim = 30
    
    network = ActorCriticNetwork(obs_dim, action_dim)
    
    # Create observation and mask
    obs = torch.randn(1, obs_dim)
    action_mask = torch.zeros(1, action_dim, dtype=torch.bool)
    action_mask[0, [0, 5, 10, 15, 20]] = True  # Only 5 actions valid
    
    # Sample action
    action, log_prob, value = network.get_action(obs, action_mask)
    
    print(f"  Sampled action: {action.item()}")
    print(f"  Log probability: {log_prob.item():.4f}")
    print(f"  Value estimate: {value.item():.4f}")
    print(f"  Action is valid: {action.item() in [0, 5, 10, 15, 20]}")
    
    # Test deterministic mode
    action_det, _, _ = network.get_action(obs, action_mask, deterministic=True)
    print(f"  Deterministic action: {action_det.item()}")
    print("  ✅ Action sampling working\n")


def test_simple_episode():
    """Test running a simple episode"""
    print("Testing simple episode...")
    
    env = CompactRegicideEnv(num_players=2)
    obs, info = env.reset(seed=42)
    
    episode_reward = 0
    episode_length = 0
    
    for step in range(50):
        # Get valid actions
        valid_actions = info['valid_actions']
        
        if len(valid_actions) == 0:
            print("  ⚠️ No valid actions available")
            break
        
        # Random action from valid actions
        action = np.random.choice(valid_actions)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_length += 1
        
        if terminated or truncated:
            break
    
    print(f"  Episode length: {episode_length}")
    print(f"  Episode reward: {episode_reward:.2f}")
    print(f"  Game status: {GameStatus(info['game_status']).name}")
    print("  ✅ Episode completed\n")


def test_experience_buffer():
    """Test experience buffer"""
    print("Testing experience buffer...")
    
    from train_mappo_pytorch import ExperienceBuffer, MAPPOConfig
    
    config = MAPPOConfig(num_players=2)
    buffer = ExperienceBuffer(config)
    
    # Add some fake experiences
    for i in range(10):
        obs = np.random.randn(48)
        action = np.random.randint(0, 30)
        reward = np.random.randn()
        value = np.random.randn()
        log_prob = np.random.randn()
        action_mask = np.random.rand(30) > 0.5
        done = (i == 9)
        
        buffer.add(obs, action, reward, value, log_prob, action_mask, done)
    
    print(f"  Buffer size: {len(buffer.observations)}")
    
    # Compute returns
    returns, advantages = buffer.compute_returns_and_advantages(last_value=0.0)
    
    print(f"  Returns shape: {returns.shape}")
    print(f"  Advantages shape: {advantages.shape}")
    print(f"  Mean return: {returns.mean():.4f}")
    print(f"  Mean advantage: {advantages.mean():.4f}")
    print("  ✅ Experience buffer working\n")


def test_tensorboard_import():
    """Test TensorBoard import"""
    print("Testing TensorBoard import...")
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("  ✅ TensorBoard available\n")
    except ImportError:
        print("  ⚠️ TensorBoard not available - install with: pip install tensorboard\n")


def main():
    """Run all tests"""
    print("="*60)
    print("MAPPO PyTorch Setup Test")
    print("="*60 + "\n")
    
    tests = [
        ("PyTorch Available", test_pytorch_available),
        ("Environment Creation", test_environment_creation),
        ("Environment Reset", test_environment_reset),
        ("Action Masking", test_action_masking),
        ("Network Creation", test_network_creation),
        ("Action Sampling", test_action_sampling),
        ("Simple Episode", test_simple_episode),
        ("Experience Buffer", test_experience_buffer),
        ("TensorBoard Import", test_tensorboard_import),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test_name} failed: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✅ All tests passed! You're ready to train MAPPO!")
        print("\nRun training with:")
        print("  python train_mappo_pytorch.py")
        print("\nMonitor with TensorBoard:")
        print("  tensorboard --logdir runs/")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before training.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
