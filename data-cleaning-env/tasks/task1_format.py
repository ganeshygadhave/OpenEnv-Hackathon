"""
Task 1: Format Fixing
Fix date and salary formats to standard form.
"""
from typing import Tuple
from datetime import datetime
import re
import copy
from models import Observation, Action
from tasks.base import BaseTask


class FormatFixTask(BaseTask):
    """Fix date (→ YYYY-MM-DD) and salary (→ integer) formatting errors."""

    def __init__(self):
        super().__init__("format_fix")
        self.original_data = []

    def reset(self) -> Observation:
        """Initialize dataset with format errors."""
        self.step_count = 0
        self.data = [
            {"id": 1, "name": "Alice", "join_date": "03-15-2019", "salary": "$75,000.00"},
            {"id": 2, "name": "Bob", "join_date": "2020/07/22", "salary": "82500"},
            {"id": 3, "name": "Charlie", "join_date": "2021-11-10", "salary": "$95,000.50"},
            {"id": 4, "name": "Diana", "join_date": "15/11/2018", "salary": "68,500"},
        ]
        self.original_data = copy.deepcopy(self.data)
        self.errors_remaining = 4  # rows 0,1,2,3 all have format issues
        
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
            if action.operation == "fix_format":
                for row_idx in action.target_rows:
                    if row_idx < 0 or row_idx >= len(self.data):
                        raise ValueError(f"Row index {row_idx} out of bounds")
                    
                    row = self.data[row_idx]
                    col = action.target_column
                    
                    if col not in row:
                        raise ValueError(f"Column '{col}' not found")
                    
                    old_val = row[col]
                    new_val = action.new_value
                    
                    # Validate fix
                    if col == "join_date":
                        if not self._is_valid_date(new_val):
                            raise ValueError(f"Invalid date format: {new_val}")
                        if self._is_valid_date(str(old_val)):
                            is_correct_fix = False  # Already correct, false positive
                        row[col] = new_val
                    elif col == "salary":
                        try:
                            salary_int = int(new_val)
                            if salary_int <= 0:
                                raise ValueError("Salary must be positive")
                            if str(old_val).isdigit():
                                is_correct_fix = False  # Already correct
                            row[col] = salary_int
                        except (ValueError, TypeError) as e:
                            raise ValueError(f"Invalid salary: {new_val}")
                    else:
                        raise ValueError(f"Cannot fix column '{col}'")
                
                # Recount errors
                self.errors_remaining = self._count_errors()
                last_action_result = f"Fixed {len(action.target_rows)} row(s). Errors remaining: {self.errors_remaining}"
            
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
        return "Fix date formats to YYYY-MM-DD and salary to plain integers (no $ or commas)"

    def _is_valid_date(self, val: str) -> bool:
        """Check if value is already in YYYY-MM-DD format."""
        if not isinstance(val, str):
            return False
        try:
            datetime.strptime(val, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def _count_errors(self) -> int:
        """Count remaining format errors."""
        count = 0
        for row in self.data:
            # Check date format
            if "join_date" in row:
                if not self._is_valid_date(str(row["join_date"])):
                    count += 1
            # Check salary format
            if "salary" in row:
                val = row["salary"]
                if not (isinstance(val, int) or (isinstance(val, float) and val.is_integer())):
                    count += 1
        return count
