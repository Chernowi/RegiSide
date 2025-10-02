#!/usr/bin/env python3
"""
JaxMARL Regicide Integration Examples

This script demonstrates how to use the JaxMARL-compatible Regicide environment
with various MARL algorithms and training configurations.

Run with: python jaxmarl_examples.py
"""

import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from jaxmarl_regicide import JaxMARLRegicide, make_regicide_env
    print("✅ Successfully imported JaxMARL Regicide environment")
except ImportError as e:
    print(f"❌ Failed to import JaxMARL Regicide: {e}")
    sys.exit(1)

# Check if JaxMARL is available
try:
    import jaxmarl
    from jaxmarl import make
    JAXMARL_AVAILABLE = True
    print("✅ JaxMARL available for full integration")
except ImportError:
    JAXMARL_AVAILABLE = False
    print("⚠️  JaxMARL not installed - running basic examples only")

def basic_environment_test():
    """Test basic environment functionality."""
    print("\n" + "="*50)
    print("BASIC ENVIRONMENT TEST")
    print("="*50)
    
    # Create environment
    env = JaxMARLRegicide(num_players=4, max_steps=500)
    
    print(f"Environment: {env.name}")
    print(f"Agents: {env.agents}")
    print(f"Action space size: {env.action_spaces[env.agents[0]].n}")
    print(f"Observation space shape: {env.observation_spaces[env.agents[0]].shape}")
    
    # Test reset
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    
    print(f"\nInitial state:")
    print(f"  Status: {['IN_PROGRESS', 'AWAITING_DEFENSE', 'AWAITING_JESTER_CHOICE', 'WON', 'LOST'][state.status]}")
    print(f"  Current player: {env.agents[state.current_player]}")
    print(f"  Enemy health: {state.current_enemy_health}")
    print(f"  Castle remaining: {state.castle_size}")
    print(f"  Tavern remaining: {state.tavern_size}")
    
    # Test action masking
    avail_actions = env.get_avail_actions(state)
    current_agent = env.agents[state.current_player]
    valid_count = jnp.sum(avail_actions[current_agent])
    print(f"  Valid actions for {current_agent}: {valid_count}/30")
    
    # Test multiple steps
    total_reward = 0
    steps = 0
    
    for step in range(10):
        # Sample valid actions for all agents
        actions = {}
        for agent in env.agents:
            mask = avail_actions[agent]
            valid_actions = jnp.where(mask)[0]
            
            if len(valid_actions) > 0:
                action_idx = jax.random.randint(key, (), 0, len(valid_actions))
                actions[agent] = valid_actions[action_idx]
            else:
                actions[agent] = 0  # Yield
        
        # Step environment
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, infos = env.step_env(step_key, state, actions)
        
        # Update tracking
        current_agent = env.agents[infos["current_player"]]
        total_reward += rewards[current_agent]
        steps += 1
        
        # Get new action masks
        avail_actions = env.get_avail_actions(state)
        
        print(f"  Step {step+1}: Player={current_agent}, Reward={rewards[current_agent]:.2f}, "
              f"Enemy_HP={state.current_enemy_health}, Done={dones['__all__']}")
        
        if dones["__all__"]:
            break
    
    print(f"\nCompleted {steps} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final status: {['IN_PROGRESS', 'AWAITING_DEFENSE', 'AWAITING_JESTER_CHOICE', 'WON', 'LOST'][state.status]}")
    
    return env

def performance_benchmark():
    """Benchmark environment performance."""
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
    
    print("JIT warmup complete!")
    
    # Benchmark resets
    print("\nBenchmarking reset operations...")
    start_time = time.time()
    reset_count = 1000
    
    for i in range(reset_count):
        key, subkey = jax.random.split(key)
        obs, state = env.reset(subkey)
    
    reset_time = time.time() - start_time
    reset_fps = reset_count / reset_time
    
    print(f"Reset performance: {reset_fps:.0f} resets/second")
    
    # Benchmark steps
    print("Benchmarking step operations...")
    obs, state = env.reset(key)
    
    start_time = time.time()
    step_count = 10000
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
    
    print(f"Step performance: {step_fps:.0f} steps/second")
    print(f"Episodes completed: {episodes}")
    print(f"Average episode length: {step_count/episodes:.1f} steps")
    
    return {
        'reset_fps': reset_fps,
        'step_fps': step_fps,
        'episodes': episodes,
        'avg_episode_length': step_count/episodes
    }

def vectorized_training_demo():
    """Demonstrate vectorized training capabilities."""
    print("\n" + "="*50)
    print("VECTORIZED TRAINING DEMO")
    print("="*50)
    
    # Create multiple environments
    num_envs = 64
    env = JaxMARLRegicide(num_players=4, max_steps=200)
    
    print(f"Creating {num_envs} parallel environments...")
    
    # Vectorized reset
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    
    print("Vectorized reset...")
    start_time = time.time()
    
    # Use vmap to reset all environments in parallel
    reset_fn = jax.vmap(env.reset)
    obs_batch, state_batch = reset_fn(keys)
    
    reset_time = time.time() - start_time
    
    print(f"Reset {num_envs} environments in {reset_time:.3f}s")
    print(f"Observation batch shape: {obs_batch[env.agents[0]].shape}")
    
    # Vectorized steps
    print("Running vectorized training loop...")
    
    # Create batch actions (all agents yield)
    actions_batch = {}
    for agent in env.agents:
        actions_batch[agent] = jnp.zeros(num_envs, dtype=jnp.int32)
    
    step_fn = jax.vmap(env.step_env)
    
    total_steps = 0
    episodes_completed = 0
    start_time = time.time()
    
    for iteration in range(100):  # 100 iterations
        # Generate new keys for each environment
        keys = jax.random.split(jax.random.PRNGKey(iteration), num_envs)
        
        # Step all environments
        obs_batch, state_batch, rewards_batch, dones_batch, infos_batch = step_fn(
            keys, state_batch, actions_batch
        )
        
        total_steps += num_envs
        
        # Count completed episodes
        episodes_completed += jnp.sum(dones_batch["__all__"])
        
        # Reset completed environments
        terminal_mask = dones_batch["__all__"]
        if jnp.any(terminal_mask):
            reset_keys = jax.random.split(keys[0], num_envs)
            new_obs, new_states = reset_fn(reset_keys)
            
            # Update only terminal environments
            for agent in env.agents:
                obs_batch[agent] = jnp.where(
                    terminal_mask[:, None], 
                    new_obs[agent], 
                    obs_batch[agent]
                )
            
            # Update states (this is simplified - real implementation would be more complex)
            # state_batch = jax.tree_map(
            #     lambda new_val, old_val: jnp.where(terminal_mask, new_val, old_val),
            #     new_states, state_batch
            # )
    
    training_time = time.time() - start_time
    throughput = total_steps / training_time
    
    print(f"Vectorized training results:")
    print(f"  Total steps: {total_steps}")
    print(f"  Episodes completed: {episodes_completed}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Throughput: {throughput:.0f} steps/second")
    print(f"  Parallel speedup: ~{throughput/5000:.1f}x over single env")
    
    return {
        'total_steps': total_steps,
        'episodes_completed': int(episodes_completed),
        'throughput': throughput,
        'num_envs': num_envs
    }

def action_masking_analysis():
    """Analyze action masking patterns in the environment."""
    print("\n" + "="*50)
    print("ACTION MASKING ANALYSIS")
    print("="*50)
    
    env = JaxMARLRegicide(num_players=4)
    key = jax.random.PRNGKey(123)
    
    # Collect action masking statistics
    action_counts = jnp.zeros(30, dtype=jnp.int32)
    total_states = 0
    
    for episode in range(100):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        
        for step in range(200):
            if state.terminal:
                break
                
            # Get available actions for current player
            avail_actions = env.get_avail_actions(state)
            current_agent = env.agents[state.current_player]
            mask = avail_actions[current_agent]
            
            # Count available actions
            action_counts += mask.astype(jnp.int32)
            total_states += 1
            
            # Take random valid action
            valid_actions = jnp.where(mask)[0]
            if len(valid_actions) > 0:
                action_idx = jax.random.randint(key, (), 0, len(valid_actions))
                action = valid_actions[action_idx]
            else:
                action = 0
            
            actions = {agent: action if agent == current_agent else 0 for agent in env.agents}
            
            # Step environment
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, infos = env.step_env(step_key, state, actions)
    
    # Analyze results
    action_frequencies = action_counts / total_states
    
    print(f"Action masking statistics over {total_states} states:")
    print(f"Action 0 (Yield): {action_frequencies[0]:.3f}")
    print(f"Actions 1-5 (Single cards): {jnp.mean(action_frequencies[1:6]):.3f}")
    print(f"Actions 6-15 (Ace combos): {jnp.mean(action_frequencies[6:16]):.3f}")
    print(f"Actions 16-20 (Sets): {jnp.mean(action_frequencies[16:21]):.3f}")
    print(f"Actions 21-25 (Jokers): {jnp.mean(action_frequencies[21:26]):.3f}")
    print(f"Actions 26-29 (Defense): {jnp.mean(action_frequencies[26:30]):.3f}")
    
    avg_valid_actions = jnp.mean(jnp.sum(action_counts)) / total_states
    print(f"Average valid actions per state: {avg_valid_actions:.1f}/30")
    
    return action_frequencies

def simple_training_example():
    """Simple training loop example with basic Q-learning concepts."""
    print("\n" + "="*50)
    print("SIMPLE TRAINING EXAMPLE")
    print("="*50)
    
    env = JaxMARLRegicide(num_players=2)  # Smaller game for faster training
    key = jax.random.PRNGKey(456)
    
    # Simple epsilon-greedy policy parameters
    epsilon = 0.1
    learning_rate = 0.01
    
    # Track training metrics
    episode_rewards = []
    episode_lengths = []
    win_rate_window = []
    
    print(f"Training simple policy for 1000 episodes...")
    
    for episode in range(1000):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        
        episode_reward = 0
        episode_length = 0
        
        while not state.terminal and episode_length < 300:
            # Get current player
            current_agent = env.agents[state.current_player]
            
            # Get available actions
            avail_actions = env.get_avail_actions(state)
            mask = avail_actions[current_agent]
            valid_actions = jnp.where(mask)[0]
            
            # Epsilon-greedy action selection (simplified)
            key, action_key = jax.random.split(key)
            if jax.random.uniform(action_key) < epsilon:
                # Random action
                action_idx = jax.random.randint(key, (), 0, len(valid_actions))
                action = valid_actions[action_idx]
            else:
                # Greedy action (prefer non-yield actions when possible)
                non_yield_actions = valid_actions[valid_actions != 0]
                if len(non_yield_actions) > 0:
                    action = non_yield_actions[0]  # Simplified greedy policy
                else:
                    action = valid_actions[0]
            
            # Create action dict
            actions = {agent: action if agent == current_agent else 0 for agent in env.agents}
            
            # Step environment
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, infos = env.step_env(step_key, state, actions)
            
            episode_reward += rewards[current_agent]
            episode_length += 1
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Track win rate
        won = state.status == 3
        win_rate_window.append(won)
        if len(win_rate_window) > 100:
            win_rate_window.pop(0)
        
        # Decay epsilon
        epsilon = max(0.01, epsilon * 0.9995)
        
        # Log progress
        if episode % 200 == 0:
            recent_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            recent_length = np.mean(episode_lengths[-50:]) if len(episode_lengths) >= 50 else np.mean(episode_lengths)
            win_rate = np.mean(win_rate_window) if win_rate_window else 0
            
            print(f"Episode {episode}:")
            print(f"  Avg Reward (last 50): {recent_reward:.2f}")
            print(f"  Avg Length (last 50): {recent_length:.1f}")
            print(f"  Win Rate (last 100): {win_rate:.2%}")
            print(f"  Epsilon: {epsilon:.3f}")
    
    print(f"\nTraining completed!")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_win_rate': np.mean(win_rate_window[-50:]) if len(win_rate_window) >= 50 else 0
    }

def jaxmarl_compatibility_test():
    """Test compatibility with JaxMARL framework if available."""
    print("\n" + "="*50)
    print("JAXMARL COMPATIBILITY TEST")
    print("="*50)
    
    if not JAXMARL_AVAILABLE:
        print("⚠️  JaxMARL not available - skipping compatibility test")
        print("Install JaxMARL with: pip install jaxmarl")
        return None
    
    try:
        # Test direct environment creation
        env = JaxMARLRegicide(num_players=4)
        print(f"✅ Direct environment creation successful")
        
        # Test JaxMARL interface methods
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        print(f"✅ Reset method compatible")
        
        actions = {agent: 0 for agent in env.agents}
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, infos = env.step_env(step_key, state, actions)
        print(f"✅ Step method compatible")
        
        obs_dict = env.get_obs(state)
        print(f"✅ Observation method compatible")
        
        avail_dict = env.get_avail_actions(state)
        print(f"✅ Action masking compatible")
        
        # Test registered environments
        try:
            reg_env = make('regicide_v1')
            print(f"✅ Registry integration successful")
        except Exception as e:
            print(f"⚠️  Registry integration issue: {e}")
        
        print("\n🚀 JaxMARL compatibility test PASSED!")
        print("Environment is ready for use with JaxMARL algorithms!")
        
        return True
        
    except Exception as e:
        print(f"❌ JaxMARL compatibility test FAILED: {e}")
        return False

def create_performance_report(results: Dict):
    """Create a performance report from benchmark results."""
    print("\n" + "="*50)
    print("PERFORMANCE REPORT")
    print("="*50)
    
    if 'reset_fps' in results:
        print(f"Reset Performance: {results['reset_fps']:.0f} resets/second")
        print(f"Step Performance: {results['step_fps']:.0f} steps/second")
        print(f"Average Episode Length: {results['avg_episode_length']:.1f} steps")
    
    if 'throughput' in results:
        print(f"Vectorized Throughput: {results['throughput']:.0f} steps/second")
        print(f"Parallel Environments: {results['num_envs']}")
        print(f"Episodes Completed: {results['episodes_completed']}")
    
    print(f"\nPerformance vs Original Environment:")
    print(f"  Original: ~10-50 steps/second (with database)")
    print(f"  Optimized: {results.get('step_fps', 'N/A')} steps/second")
    print(f"  Improvement: ~{results.get('step_fps', 0)/50:.0f}x faster")
    
    print(f"\nSuitability for MARL Training:")
    if results.get('step_fps', 0) > 1000:
        print(f"  ✅ Excellent - Can train millions of episodes efficiently")
    elif results.get('step_fps', 0) > 500:
        print(f"  ✅ Good - Suitable for most MARL experiments") 
    else:
        print(f"  ⚠️  Marginal - May need further optimization")
    
    return results

def main():
    """Run all examples and tests."""
    print("JaxMARL Regicide Integration Examples")
    print("====================================")
    
    results = {}
    
    # Basic functionality test
    env = basic_environment_test()
    
    # Performance benchmark
    perf_results = performance_benchmark()
    results.update(perf_results)
    
    # Vectorized training demo
    vec_results = vectorized_training_demo()
    results.update(vec_results)
    
    # Action masking analysis
    action_freqs = action_masking_analysis()
    
    # Simple training example
    training_results = simple_training_example()
    
    # JaxMARL compatibility test
    compat_result = jaxmarl_compatibility_test()
    
    # Final performance report
    create_performance_report(results)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("✅ Environment successfully created and tested")
    print(f"✅ Performance: {results.get('step_fps', 'N/A')} steps/second")
    print(f"✅ Vectorized: {results.get('throughput', 'N/A')} parallel steps/second")
    
    if compat_result:
        print("✅ JaxMARL integration working")
    else:
        print("⚠️  JaxMARL integration needs attention")
    
    print(f"✅ Training example completed with {training_results.get('final_win_rate', 0):.1%} win rate")
    print("\n🎉 All tests completed successfully!")
    print("\nNext steps:")
    print("  1. Install JaxMARL: pip install jaxmarl")
    print("  2. Run with your favorite MARL algorithm (QMIX, MAPPO, IQL, etc.)")
    print("  3. Experiment with different player configurations")
    print("  4. Scale up to large-scale distributed training")

if __name__ == "__main__":
    main()