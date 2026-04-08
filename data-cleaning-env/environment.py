"""
DataClean OpenEnv — Reinforcement Learning environment for data cleaning tasks.
"""
from typing import Tuple, Optional
from models import Observation, Action, Reward
from tasks.task1_format import FormatFixTask
from tasks.task2_imputation import ImputationTask
from tasks.task3_pipeline import PipelineTask


class DataCleanEnv:
    """Main RL environment orchestrator for data cleaning tasks."""

    def __init__(self, task_name: str):
        if task_name not in ("format_fix", "imputation", "pipeline"):
            raise ValueError(f"Unknown task: {task_name}")
        
        self.task_name = task_name
        
        # Load appropriate task
        if task_name == "format_fix":
            self.task = FormatFixTask()
        elif task_name == "imputation":
            self.task = ImputationTask()
        else:  # pipeline
            self.task = PipelineTask()
        
        self.obs: Optional[Observation] = None

    def reset(self) -> Observation:
        """Initialize environment and return initial observation."""
        self.obs = self.task.reset()
        return self.obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Execute action and return (observation, reward, done, info).
        
        Args:
            action: Action object with operation, target_rows, target_column, etc.
        
        Returns:
            observation: Updated state after action
            reward: Score for this action
            done: True if task is complete
            info: Metadata (errors, explanations)
        """
        obs, reward, done, info = self.task.step(action)
        self.obs = obs
        return obs, reward, done, info

    def state(self) -> dict:
        """Return current environment state as dict."""
        if self.obs is None:
            return {"error": "Environment not initialized. Call reset() first."}
        
        return {
            "task_name": self.obs.task_name,
            "step": self.obs.step,
            "current_data": self.obs.current_data,
            "errors_remaining": self.obs.errors_remaining,
            "last_action_result": self.obs.last_action_result,
            "hint": self.obs.hint,
        }

    def close(self):
        """Cleanup resources."""
        self.task.close()
