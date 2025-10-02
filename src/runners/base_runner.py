"""
Base Runner Class for MARL Training

Abstract base class that defines the interface for all algorithm runners.
Each specific algorithm (MAPPO, QMIX, IQL, etc.) should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging
import os
import time
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class BaseRunner(ABC):
    """
    Abstract base class for algorithm runners.
    
    Provides common functionality for:
    - Configuration management
    - Logging (console + tensorboard)
    - Checkpointing
    - Training loop orchestration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the runner.
        
        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config
        self.algorithm = config.get('algorithm', 'unknown')
        
        # Setup logging
        self._setup_logging()
        
        # Setup directories
        self._setup_directories()
        
        # Initialize tensorboard writer
        self.writer: Optional[SummaryWriter] = None
        if config.get('logging', {}).get('use_tensorboard', True):
            self._setup_tensorboard()
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.update_count = 0
        self.start_time = None
        self.best_metric = None
        
        # Metrics storage
        self.metrics_buffer = []
        
        self.logger.info(f"Initialized {self.algorithm.upper()} Runner")
        self.logger.info(f"Config: {self.config.get('logging', {}).get('experiment_name', 'unknown')}")
    
    def _setup_logging(self):
        """Setup console logging"""
        log_level = self.config.get('logging', {}).get('log_level', 'INFO')
        
        # Create logger
        self.logger = logging.getLogger(f"{self.algorithm}_runner")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        # Prevent messages from propagating to the root logger (avoid duplicate prints)
        self.logger.propagate = False
    
    def _setup_directories(self):
        """Create necessary directories for checkpoints and logs"""
        # Checkpoint directory
        checkpoint_dir = self.config.get('checkpoint', {}).get('save_dir', 'checkpoints')
        self.checkpoint_dir = Path(checkpoint_dir) / self._get_run_name()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _setup_tensorboard(self):
        """Setup tensorboard writer"""
        tensorboard_dir = self.config.get('logging', {}).get('tensorboard_dir', 'runs')
        log_dir = Path(tensorboard_dir) / self._get_run_name()
        
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.logger.info(f"Tensorboard log directory: {log_dir}")
    
    def _get_run_name(self) -> str:
        """Generate a unique run name"""
        run_name = self.config.get('logging', {}).get('run_name')
        
        if run_name is None:
            # Auto-generate run name
            experiment_name = self.config.get('logging', {}).get('experiment_name', 'experiment')
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            run_name = f"{experiment_name}_{timestamp}"
        
        return run_name
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to console and tensorboard.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Global step number (uses self.global_step if None)
        """
        if step is None:
            step = self.global_step
        
        # Log to tensorboard
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    self.writer.add_scalar(key, value, step)
        
        # Log to console (if verbose)
        if self.config.get('logging', {}).get('verbose', True):
            log_interval = self.config.get('logging', {}).get('log_interval', 10)
            if self.update_count % log_interval == 0:
                metrics_str = ' | '.join([f"{k}: {v:.4f}" if isinstance(v, (float, np.floating)) 
                                         else f"{k}: {v}" 
                                         for k, v in metrics.items()])
                self.logger.info(f"Step {step} | {metrics_str}")
    
    def save_checkpoint(self, filepath: Optional[str] = None, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint (auto-generated if None)
            is_best: Whether this is the best model so far
        """
        if filepath is None:
            filename = f"checkpoint_step_{self.global_step}.pt"
            filepath = self.checkpoint_dir / filename
        
        checkpoint_data = self._get_checkpoint_data()
        
        # Delegate to algorithm-specific implementation
        self._save_checkpoint_impl(checkpoint_data, filepath)
        
        self.logger.info(f"Saved checkpoint to {filepath}")
        
        # Save best model separately
        if is_best and self.config.get('checkpoint', {}).get('save_best', True):
            best_path = self.checkpoint_dir / "best_model.pt"
            self._save_checkpoint_impl(checkpoint_data, best_path)
            self.logger.info(f"Saved best model to {best_path}")
    
    def _get_checkpoint_data(self) -> Dict[str, Any]:
        """Get common checkpoint data"""
        return {
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'best_metric': self.best_metric,
            'config': self.config,
        }
    
    @abstractmethod
    def _save_checkpoint_impl(self, checkpoint_data: Dict[str, Any], filepath: Path):
        """
        Algorithm-specific checkpoint saving.
        
        Args:
            checkpoint_data: Common checkpoint data
            filepath: Path to save checkpoint
        """
        pass
    
    @abstractmethod
    def setup(self):
        """
        Setup the runner: initialize environment, agent, optimizer, etc.
        This is called before training starts.
        """
        pass
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Dictionary containing final training metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: Optional[int] = None, visualize: bool = False) -> Dict[str, Any]:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of episodes to evaluate (uses config if None)
            visualize: Whether to generate visualizations during evaluation
        
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def run(self) -> Dict[str, Any]:
        """
        Main entry point: setup, train, and cleanup.
        
        Returns:
            Final training metrics
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting {self.algorithm.upper()} Training")
        self.logger.info("=" * 80)
        
        self.start_time = time.time()
        
        try:
            # Setup
            self.logger.info("Setting up environment and agent...")
            self.setup()
            
            # Train
            self.logger.info("Starting training...")
            final_metrics = self.train()
            
            # Final evaluation
            if self.config.get('evaluation', {}).get('enabled', True):
                self.logger.info("Running final evaluation...")
                eval_metrics = self.evaluate(visualize=True)
                final_metrics.update({f"final_{k}": v for k, v in eval_metrics.items()})
            
            # Log total training time
            total_time = time.time() - self.start_time
            self.logger.info(f"Training completed in {total_time:.2f} seconds")
            final_metrics['total_training_time'] = total_time
            
            return final_metrics
            
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            self.save_checkpoint(is_best=False)
            raise
        
        except Exception as e:
            self.logger.error(f"Error during training: {e}", exc_info=True)
            raise
        
        finally:
            # Cleanup
            if self.writer is not None:
                self.writer.close()
            self.logger.info("Cleaned up resources")
    
    def should_save_checkpoint(self) -> bool:
        """Check if we should save a checkpoint at this step"""
        save_interval = self.config.get('logging', {}).get('save_interval', 100)
        return self.update_count % save_interval == 0
    
    def should_evaluate(self) -> bool:
        """Check if we should run evaluation at this step"""
        eval_interval = self.config.get('logging', {}).get('eval_interval', 50)
        return self.update_count % eval_interval == 0
    
    def update_best_metric(self, current_metric: float) -> bool:
        """
        Update best metric and return True if this is a new best.
        
        Args:
            current_metric: Current metric value
        
        Returns:
            True if this is a new best metric
        """
        metric_name = self.config.get('checkpoint', {}).get('metric', 'episode_return')
        mode = self.config.get('checkpoint', {}).get('mode', 'max')
        
        if self.best_metric is None:
            self.best_metric = current_metric
            return True
        
        if mode == 'max':
            is_better = current_metric > self.best_metric
        else:
            is_better = current_metric < self.best_metric
        
        if is_better:
            self.best_metric = current_metric
            self.logger.info(f"New best {metric_name}: {current_metric:.4f}")
        
        return is_better
