"""
Base Task Class — Abstract interface for all cleaning tasks.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from models import Observation, Action


class BaseTask(ABC):
    """Abstract base class for data cleaning tasks."""

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.step_count = 0
        self.data = []
        self.errors_remaining = 0

    @abstractmethod
    def reset(self) -> Observation:
        """Initialize task with seeded data and errors. Return initial observation."""
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Execute action on current data.
        
        Returns:
            observation: Current state after action
            reward: Score for this action (0.0-1.0)
            done: True if all errors fixed
            info: Metadata (error message if action failed, etc.)
        """
        pass

    def close(self):
        """Cleanup resources (if any)."""
        pass

    @abstractmethod
    def get_hint(self) -> str:
        """Return task-specific hint for the agent."""
        pass

    def _calculate_reward(self, errors_before: int, errors_after: int, 
                          action_valid: bool = True, is_correct_fix: bool = True) -> float:
        """Simple reward: +1.0 if fixed an error, 0.0 if no-op, -0.5 if false positive."""
        if not action_valid:
            return -0.5
        if errors_before > errors_after:
            return 1.0 if is_correct_fix else -0.5
        return 0.0
