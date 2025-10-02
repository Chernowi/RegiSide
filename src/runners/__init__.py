"""
Runners package for MARL training.

Contains base runner class and algorithm-specific implementations.
"""

from .base_runner import BaseRunner
from .mappo_runner import MAPPORunner

__all__ = ['BaseRunner', 'MAPPORunner']
