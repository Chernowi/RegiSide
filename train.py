#!/usr/bin/env python3
"""
Main Training Script for Regicide MARL

This script loads a YAML configuration file and trains a multi-agent
reinforcement learning algorithm on the Regicide environment.

Usage:
    python train.py --config configs/mappo_default.yaml
    python train.py --config configs/mappo_default.yaml --seed 123
    python train.py --config configs/qmix_config.yaml --run-name my_experiment
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.runners import BaseRunner, MAPPORunner


# Registry of available algorithms
ALGORITHM_REGISTRY = {
    'mappo': MAPPORunner,
    # Add more algorithms here as they're implemented
    # 'qmix': QMIXRunner,
    # 'iql': IQLRunner,
    # 'vdn': VDNRunner,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment configuration if specified
    if 'env_config' in config:
        env_config_path = Path(config['env_config'])
        if not env_config_path.exists():
            raise FileNotFoundError(f"Environment configuration file not found: {env_config_path}")
        
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
        
        # Merge environment config into main config under 'env' key
        config['env'] = env_config
        
        # Remove the env_config reference
        del config['env_config']
    
    return config


def override_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Override configuration with command-line arguments.
    
    Args:
        config: Base configuration dictionary
        args: Command-line arguments
    
    Returns:
        Updated configuration dictionary
    """
    # Override seed
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Override run name
    if args.run_name is not None:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['run_name'] = args.run_name
    
    # Override device
    if args.device is not None:
        config['device'] = args.device
    
    # Override number of environments
    if args.num_envs is not None:
        if 'env' not in config:
            config['env'] = {}
        config['env']['num_envs'] = args.num_envs
    
    # Override total timesteps
    if args.timesteps is not None:
        algorithm = config.get('algorithm', 'mappo')
        if algorithm not in config:
            config[algorithm] = {}
        config[algorithm]['total_timesteps'] = args.timesteps
    
    # Disable tensorboard if requested
    if args.no_tensorboard:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['use_tensorboard'] = False
    
    # Enable wandb if requested
    if args.wandb:
        if 'advanced' not in config:
            config['advanced'] = {}
        config['advanced']['use_wandb'] = True
        
        if args.wandb_project:
            config['advanced']['wandb_project'] = args.wandb_project
    
    return config


def create_runner(config: Dict[str, Any]) -> BaseRunner:
    """
    Create appropriate runner based on algorithm in config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Initialized runner instance
    """
    algorithm = config.get('algorithm', '').lower()
    
    if not algorithm:
        raise ValueError("No algorithm specified in configuration")
    
    if algorithm not in ALGORITHM_REGISTRY:
        available = ', '.join(ALGORITHM_REGISTRY.keys())
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. "
            f"Available algorithms: {available}"
        )
    
    runner_class = ALGORITHM_REGISTRY[algorithm]
    return runner_class(config)


def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train MARL algorithms on Regicide environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default MAPPO configuration
  python train.py --config configs/mappo_default.yaml
  
  # Override seed and run name
  python train.py --config configs/mappo_default.yaml --seed 123 --run-name experiment_1
  
  # Train with more environments
  python train.py --config configs/mappo_default.yaml --num-envs 128
  
  # Quick test run with fewer timesteps
  python train.py --config configs/mappo_default.yaml --timesteps 1000000
  
  # Use wandb for logging
  python train.py --config configs/mappo_default.yaml --wandb --wandb-project my-project
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Name for this training run (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['auto', 'cpu', 'cuda', 'gpu'],
        help='Device to use for training (overrides config)'
    )
    
    parser.add_argument(
        '--num-envs',
        type=int,
        default=None,
        help='Number of parallel environments (overrides config)'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Total training timesteps (overrides config)'
    )
    
    parser.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='Disable tensorboard logging'
    )
    
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='W&B project name (requires --wandb)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Setup
    setup_logging()
    logger = logging.getLogger('train')
    
    # Parse arguments
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("Regicide MARL Training")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Override with command-line arguments
        config = override_config(config, args)
        
        # Display configuration summary
        algorithm = config.get('algorithm', 'unknown').upper()
        logger.info(f"Algorithm: {algorithm}")
        logger.info(f"Seed: {config.get('seed', 'not set')}")
        
        if 'logging' in config and 'run_name' in config['logging']:
            logger.info(f"Run name: {config['logging']['run_name']}")
        
        # Create runner
        logger.info(f"Creating {algorithm} runner...")
        runner = create_runner(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            runner.load_checkpoint(args.resume)
        
        final_metrics = runner.run()
        
        # Display final results
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        logger.info("Final Metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
