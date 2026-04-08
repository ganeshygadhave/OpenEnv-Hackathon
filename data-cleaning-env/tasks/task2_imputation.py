"""
Task 2: Missing Value Imputation
Fill missing values using department-level group means.
"""
from typing import Tuple, Optional
import statistics
import copy
from models import Observation, Action
from tasks.base import BaseTask


class ImputationTask(BaseTask):
    """Fill missing (None) salary values using department-level group means."""

    def __init__(self):
        super().__init__("imputation")
        self.original_data = []

    def reset(self) -> Observation:
        """Initialize dataset with missing values."""
        self.step_count = 0
        self.data = [
            {"id": 1, "name": "Alice", "department": "Sales", "salary": 75000},
            {"id": 2, "name": "Bob", "department": "Sales", "salary": None},
            {"id": 3, "name": "Charlie", "department": "Engineering", "salary": 95000},
            {"id": 4, "name": "Diana", "department": "Engineering", "salary": None},
            {"id": 5, "name": "Eve", "department": "HR", "salary": 65000},
            {"id": 6, "name": "Frank", "department": "Sales", "salary": 80000},
        ]
        self.original_data = copy.deepcopy(self.data)
        self.errors_remaining = 2  # 2 missing values
        
        return Observation(
            task_name=self.task_name,
            step=self.step_count,
            current_data=self.data,
            errors_remaining=self.errors_remaining,
            hint=self.get_hint(),
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Execute action and return observation + reward."""
        self.step_count = getattr(self, 'step_count', -1) + 1
        errors_before = self.errors_remaining
        info = {}
        last_action_result = ""
        is_correct_fix = True

        try:
            if action.operation == "impute":
                for row_idx in action.target_rows:
                    if row_idx < 0 or row_idx >= len(self.data):
                        raise ValueError(f"Row index {row_idx} out of bounds")
                    
                    row = self.data[row_idx]
                    col = action.target_column
                    
                    if col not in row:
                        raise ValueError(f"Column '{col}' not found")
                    
                    if row[col] is not None:
                        is_correct_fix = False  # False positive
                    
                    # Validate imputed value
                    if action.new_value is None:
                        raise ValueError("new_value cannot be None")
                    
                    try:
                        imputed_val = float(action.new_value)
                        if imputed_val < 0:
                            raise ValueError("Imputed value must be non-negative")
                        
                        # Check if close to department mean
                        dept = row.get("department")
                        if dept:
                            dept_mean = self._get_department_mean(dept)
                            if dept_mean and abs(imputed_val - dept_mean) > dept_mean * 0.5:
                                # Allow it but note it's far from mean
                                pass
                        
                        row[col] = imputed_val
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Invalid imputation value: {action.new_value}")
                
                self.errors_remaining = self._count_errors()
                last_action_result = f"Imputed {len(action.target_rows)} value(s). Errors remaining: {self.errors_remaining}"
            
            elif action.operation == "done":
                if self.errors_remaining == 0:
                    last_action_result = "Task completed successfully!"
                else:
                    raise ValueError(f"Cannot finish: {self.errors_remaining} errors remain")
            else:
                raise ValueError(f"Operation '{action.operation}' not allowed in this task")

        except Exception as e:
            last_action_result = f"Error: {str(e)}"
            info["error"] = str(e)
            reward = -0.5
            done = False
            return (
                Observation(
                    task_name=self.task_name,
                    step=self.step_count,
                    current_data=self.data,
                    errors_remaining=self.errors_remaining,
                    last_action_result=last_action_result,
                    hint=self.get_hint(),
                ),
                reward,
                done,
                info,
            )

        reward = self._calculate_reward(errors_before, self.errors_remaining, True, is_correct_fix)
        done = self.errors_remaining == 0

        return (
            Observation(
                task_name=self.task_name,
                step=self.step_count,
                current_data=self.data,
                errors_remaining=self.errors_remaining,
                last_action_result=last_action_result,
                hint=self.get_hint(),
            ),
            reward,
            done,
            info,
        )

    def get_hint(self) -> str:
        return "Use department-level group means to impute missing salary values"

    def _get_department_mean(self, department: str) -> Optional[float]:
        """Calculate mean salary for a department (excluding None values)."""
        values = [
            row["salary"] for row in self.data
            if row.get("department") == department and row.get("salary") is not None
        ]
        if values:
            return statistics.mean(values)
        return None

    def _count_errors(self) -> int:
        """Count remaining missing values."""
        count = 0
        for row in self.data:
            if row.get("salary") is None:
                count += 1
        return count
