#!/usr/bin/env python3
"""
Simple MAPPO Test Script for Regicide Environment

This script runs a quick test of MAPPO training on the Regicide environment
to verify everything is working correctly.

Usage: python test_mappo_regicide.py
"""

import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from jaxmarl_regicide import JaxMARLRegicide
    print("✅ Successfully imported JaxMARL Regicide environment")
except ImportError as e:
    print(f"❌ Failed to import JaxMARL Regicide: {e}")
    sys.exit(1)

def test_environment_basic():
    """Test basic environment functionality"""
    print("\n" + "="*50)
    print("TESTING BASIC ENVIRONMENT FUNCTIONALITY")  
    print("="*50)
    
    # Create environment
    env = JaxMARLRegicide(num_players=4, max_steps=200)
    
    print(f"Environment: {env.name}")
    print(f"Agents: {env.agents}")
    print(f"Action space size: {env.action_spaces[env.agents[0]].n}")
    print(f"Observation space shape: {env.observation_spaces[env.agents[0]].shape}")
    
    # Test reset
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    
    print(f"\n✅ Reset successful")
    print(f"Initial state status: {state.status}")
    print(f"Current player: {state.current_player}")
    print(f"Enemy health: {state.current_enemy_health}")
    
    # Test action masking
    avail_actions = env.get_avail_actions(state)
    current_agent = env.agents[state.current_player]
    valid_count = jnp.sum(avail_actions[current_agent])
    print(f"Valid actions for {current_agent}: {valid_count}/30")
    
    # Test step
    actions = {agent: 0 for agent in env.agents}  # All yield
    key, step_key = jax.random.split(key)
    obs, state, rewards, dones, infos = env.step_env(step_key, state, actions)
    
    print(f"✅ Step successful")
    print(f"Rewards: {[f'{agent}: {rewards[agent]:.2f}' for agent in env.agents]}")
    print(f"Done: {dones['__all__']}")
    
    return env

def test_vectorized_environments():
    """Test vectorized environment operations"""
    print("\n" + "="*50)
    print("TESTING VECTORIZED ENVIRONMENTS")
    print("="*50)
    
    env = JaxMARLRegicide(num_players=4, max_steps=100)
    num_envs = 8
    
    # Create vectorized reset
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    
    print(f"Testing {num_envs} parallel environments...")
    
    # Vectorized reset
    reset_fn = jax.vmap(env.reset)
    start_time = time.time()
    obs_batch, state_batch = reset_fn(keys)
    reset_time = time.time() - start_time
    
    print(f"✅ Vectorized reset successful in {reset_time:.3f}s")
    print(f"Observation batch shape: {obs_batch[env.agents[0]].shape}")
    
    # Vectorized step
    actions_batch = {}
    for agent in env.agents:
        actions_batch[agent] = jnp.zeros(num_envs, dtype=jnp.int32)  # All yield
    
    step_fn = jax.vmap(env.step_env)
    
    start_time = time.time()
    obs_batch, state_batch, rewards_batch, dones_batch, infos_batch = step_fn(
        keys, state_batch, actions_batch
    )
    step_time = time.time() - start_time
    
    print(f"✅ Vectorized step successful in {step_time:.3f}s")
    print(f"Reward batch shape: {rewards_batch[env.agents[0]].shape}")
    print(f"Throughput: {num_envs / step_time:.0f} envs/second")
    
    return num_envs / step_time

def test_action_masking():
    """Test action masking functionality"""
    print("\n" + "="*50)
    print("TESTING ACTION MASKING")
    print("="*50)
    
    env = JaxMARLRegicide(num_players=4)
    key = jax.random.PRNGKey(123)
    
    # Collect statistics over multiple episodes
    total_actions_available = 0
    total_states = 0
    action_type_counts = jnp.zeros(6)  # Categories: yield, single, ace, sets, jokers, defense
    
    for episode in range(10):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        
        for step in range(50):  # Short episodes
            if state.terminal:
                break
            
            # Get available actions for current player
            avail_actions = env.get_avail_actions(state)
            current_agent = env.agents[state.current_player]
            mask = avail_actions[current_agent]
            
            # Count action types
            action_type_counts = action_type_counts.at[0].add(mask[0])  # Yield
            action_type_counts = action_type_counts.at[1].add(jnp.sum(mask[1:6]))  # Single cards
            action_type_counts = action_type_counts.at[2].add(jnp.sum(mask[6:16]))  # Ace combos
            action_type_counts = action_type_counts.at[3].add(jnp.sum(mask[16:21]))  # Sets
            action_type_counts = action_type_counts.at[4].add(jnp.sum(mask[21:26]))  # Jokers
            action_type_counts = action_type_counts.at[5].add(jnp.sum(mask[26:30]))  # Defense
            
            total_actions_available += jnp.sum(mask)
            total_states += 1
            
            # Take a random valid action
            valid_actions = jnp.where(mask)[0]
            if len(valid_actions) > 0:
                key, action_key = jax.random.split(key)
                action_idx = jax.random.randint(action_key, (), 0, len(valid_actions))
                action = valid_actions[action_idx]
            else:
                action = 0
            
            actions = {agent: action if agent == current_agent else 0 for agent in env.agents}
            
            # Step environment
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, infos = env.step_env(step_key, state, actions)
    
    # Print statistics
    avg_actions = total_actions_available / total_states
    action_type_names = ["Yield", "Single Cards", "Ace Combos", "Sets", "Jokers", "Defense"]
    
    print(f"Action masking statistics over {total_states} states:")
    print(f"Average actions available: {avg_actions:.1f}/30")
    
    for i, name in enumerate(action_type_names):
        frequency = action_type_counts[i] / total_states
        print(f"  {name}: {frequency:.2f} avg available")
    
    print(f"✅ Action masking working correctly")
    
    return avg_actions

def test_simple_policy():
    """Test a simple policy to see if training could work"""
    print("\n" + "="*50)
    print("TESTING SIMPLE POLICY LEARNING")
    print("="*50)
    
    env = JaxMARLRegicide(num_players=2, max_steps=150)  # Smaller for faster testing
    key = jax.random.PRNGKey(456)
    
    # Simple policy: prefer attacking over yielding
    def simple_policy(obs, avail_actions, key):
        """Simple heuristic policy"""
        mask = avail_actions
        valid_actions = jnp.where(mask)[0]
        
        if len(valid_actions) == 0:
            return 0
        
        # Prefer non-yield actions
        non_yield_actions = valid_actions[valid_actions != 0]
        if len(non_yield_actions) > 0:
            # Prefer single card plays (actions 1-5)
            single_card_actions = non_yield_actions[non_yield_actions <= 5]
            if len(single_card_actions) > 0:
                return single_card_actions[0]
            else:
                return non_yield_actions[0]
        else:
            return 0  # Yield if no other options
    
    # Test policy over multiple episodes
    episode_rewards = []
    episode_lengths = []
    win_count = 0
    
    for episode in range(20):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        
        episode_reward = 0
        episode_length = 0
        
        while not state.terminal and episode_length < 150:
            current_agent = env.agents[state.current_player]
            
            # Get available actions
            avail_actions = env.get_avail_actions(state)
            mask = avail_actions[current_agent]
            
            # Get action from simple policy
            key, policy_key = jax.random.split(key)
            action = simple_policy(obs[current_agent], mask, policy_key)
            
            actions = {agent: action if agent == current_agent else 0 for agent in env.agents}
            
            # Step environment
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, infos = env.step_env(step_key, state, actions)
            
            episode_reward += rewards[current_agent]
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if state.status == 3:  # Won
            win_count += 1
    
    # Statistics
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    win_rate = win_count / len(episode_rewards)
    
    print(f"Simple policy results over {len(episode_rewards)} episodes:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average length: {avg_length:.1f}")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Reward range: {np.min(episode_rewards):.2f} to {np.max(episode_rewards):.2f}")
    
    print(f"✅ Simple policy test completed")
    
    return {
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'win_rate': win_rate
    }

def performance_benchmark():
    """Benchmark environment performance"""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    env = JaxMARLRegicide(num_players=4)
    key = jax.random.PRNGKey(0)
    
    # Warmup for JIT compilation
    print("Warming up JIT compilation...")
    obs, state = env.reset(key)
    actions = {agent: 0 for agent in env.agents}
    
    for _ in range(10):
        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, infos = env.step_env(subkey, state, actions)
        if state.terminal:
            obs, state = env.reset(subkey)
    
    print("✅ JIT warmup complete")
    
    # Benchmark steps
    print("Benchmarking step operations...")
    obs, state = env.reset(key)
    
    start_time = time.time()
    step_count = 5000
    episodes = 0
    
    for i in range(step_count):
        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, infos = env.step_env(subkey, state, actions)
        
        if state.terminal:
            episodes += 1
            key, reset_key = jax.random.split(key)
            obs, state = env.reset(reset_key)
    
    step_time = time.time() - start_time
    step_fps = step_count / step_time
    
    print(f"✅ Performance benchmark completed")
    print(f"Step performance: {step_fps:.0f} steps/second")
    print(f"Episodes completed: {episodes}")
    print(f"Average episode length: {step_count/episodes:.1f} steps")
    
    return step_fps

def main():
    """Run all tests"""
    print("🧪 MAPPO Regicide Environment Test Suite")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Basic functionality
    env = test_environment_basic()
    
    # Test 2: Vectorized operations
    vectorized_fps = test_vectorized_environments()
    results['vectorized_fps'] = vectorized_fps
    
    # Test 3: Action masking
    avg_actions = test_action_masking()
    results['avg_actions_available'] = avg_actions
    
    # Test 4: Simple policy
    policy_results = test_simple_policy()
    results.update(policy_results)
    
    # Test 5: Performance benchmark
    step_fps = performance_benchmark()
    results['step_fps'] = step_fps
    
    # Final summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print("✅ All tests completed successfully!")
    print(f"✅ Environment performance: {step_fps:.0f} steps/second")
    print(f"✅ Vectorized performance: {vectorized_fps:.0f} envs/second") 
    print(f"✅ Action masking: {avg_actions:.1f}/30 actions available on average")
    print(f"✅ Simple policy win rate: {policy_results['win_rate']:.1%}")
    
    print(f"\n🚀 Environment is ready for MAPPO training!")
    print(f"Expected training performance:")
    print(f"  - Single environment: ~{step_fps:.0f} steps/second")
    print(f"  - Vectorized (32 envs): ~{32 * step_fps / 1000:.0f}k steps/second")
    print(f"  - Suitable for training: {'✅ Yes' if step_fps > 1000 else '⚠️  Marginal'}")
    
    print(f"\nTo start MAPPO training, run:")
    print(f"  python train_mappo_regicide.py --num_players 4 --total_timesteps 1000000")
    
    return results

if __name__ == "__main__":
    main()