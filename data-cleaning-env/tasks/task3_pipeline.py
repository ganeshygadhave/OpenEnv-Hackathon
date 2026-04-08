"""
Task 3: Multi-step Pipeline Cleaning
Clean data in correct pipeline order with poison cell trap.
"""
from typing import Tuple, Optional
import copy
from models import Observation, Action
from tasks.base import BaseTask


class PipelineTask(BaseTask):
    """Multi-step cleaning: format → imputation → dedup → anomaly detection (in order)."""

    def __init__(self):
        super().__init__("pipeline")
        self.original_data = []
        self.operations_done = set()  # Track which operations have been done

    def reset(self) -> Observation:
        """Initialize dataset with multiple error types."""
        self.step_count = 0
        self.operations_done = set()
        self.data = [
            {"id": 1, "name": "Alice", "department": "Sales", "join_date": "03-15-2019", "salary": 75000},
            {"id": 2, "name": "Bob", "department": "Sales", "join_date": "2020/07/22", "salary": None},
            {"id": 3, "name": "Charlie", "department": "Engineering", "join_date": "2021-11-10", "salary": 95000},
            {"id": 4, "name": "Charlie", "department": "Engineering", "join_date": "2021-11-10", "salary": 95000},  # duplicate
            {"id": 5, "name": "Diana", "department": "HR", "join_date": "2018/15/11", "salary": 999999},  # poison cell (impossible date)
        ]
        self.original_data = copy.deepcopy(self.data)
        self.errors_remaining = 5  # format + impute + dedup + anomaly
        
        return Observation(
            task_name=self.task_name,
            step=self.step_count,
            current_data=self.data,
            errors_remaining=self.errors_remaining,
            hint=self.get_hint(),
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Execute action in correct pipeline order."""
        self.step_count = getattr(self, 'step_count', -1) + 1
        errors_before = self.errors_remaining
        info = {}
        last_action_result = ""
        is_correct_fix = True
        
        # Pipeline order enforcement
        allowed_ops = {
            "fix_format": {"fix_format"},
            "impute": {"fix_format", "impute"},
            "remove_duplicate": {"fix_format", "impute", "remove_duplicate"},
            "flag_anomaly": {"fix_format", "impute", "remove_duplicate", "flag_anomaly"},
            "done": None,  # Can call done anytime
        }

        try:
            op = action.operation
            
            # Check pipeline order
            if op != "done" and allowed_ops[op]:
                if not allowed_ops[op].issubset(self.operations_done):
                    missing = allowed_ops[op] - self.operations_done
                    raise ValueError(
                        f"Operation '{op}' requires prior operations: {', '.join(sorted(missing))}"
                    )
                self.operations_done.add(op)
            
            if op == "fix_format":
                for row_idx in action.target_rows:
                    if row_idx < 0 or row_idx >= len(self.data):
                        raise ValueError(f"Row index {row_idx} out of bounds")
                    
                    row = self.data[row_idx]
                    col = action.target_column
                    
                    if col == "join_date":
                        new_val = action.new_value
                        if not self._is_valid_date(new_val):
                            raise ValueError(f"Invalid date format: {new_val}")
                        row[col] = new_val
                    else:
                        raise ValueError(f"Can only fix 'join_date' in this task")
            
            elif op == "impute":
                for row_idx in action.target_rows:
                    if row_idx < 0 or row_idx >= len(self.data):
                        raise ValueError(f"Row index {row_idx} out of bounds")
                    
                    row = self.data[row_idx]
                    col = action.target_column
                    
                    if row[col] is None:
                        row[col] = float(action.new_value)
                    else:
                        is_correct_fix = False
            
            elif op == "remove_duplicate":
                for row_idx in sorted(action.target_rows, reverse=True):
                    if row_idx < 0 or row_idx >= len(self.data):
                        raise ValueError(f"Row index {row_idx} out of bounds")
                    # Check if it's actually a duplicate
                    if not self._is_duplicate(row_idx):
                        is_correct_fix = False
                    self.data.pop(row_idx)
            
            elif op == "flag_anomaly":
                # In this simplified version, flagging is just a no-op validation
                for row_idx in action.target_rows:
                    if row_idx < 0 or row_idx >= len(self.data):
                        raise ValueError(f"Row index {row_idx} out of bounds")
                    if not self._is_anomaly(row_idx):
                        is_correct_fix = False
            
            elif op == "done":
                if self._count_errors() == 0:
                    last_action_result = "Task completed successfully!"
                else:
                    raise ValueError(f"Cannot finish: {self._count_errors()} errors remain")
            else:
                raise ValueError(f"Unknown operation: '{op}'")
            
            self.errors_remaining = self._count_errors()
            if op != "done":
                last_action_result = f"{op} executed. Errors remaining: {self.errors_remaining}"

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
        return "Fix in order: format -> impute -> remove_duplicate -> flag_anomaly. Beware poison cells!"

    def _is_valid_date(self, val: str) -> bool:
        """Check if value is in YYYY-MM-DD format."""
        from datetime import datetime
        if not isinstance(val, str):
            return False
        try:
            datetime.strptime(val, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def _is_duplicate(self, row_idx: int) -> bool:
        """Check if row is an exact duplicate of another."""
        if row_idx < 0 or row_idx >= len(self.data):
            return False
        row = self.data[row_idx]
        for i, other in enumerate(self.data):
            if i != row_idx and row == other:
                return True
        return False

    def _is_anomaly(self, row_idx: int) -> bool:
        """Check if row has anomalous values (e.g., impossible salary)."""
        if row_idx < 0 or row_idx >= len(self.data):
            return False
        row = self.data[row_idx]
        # Salary > 500k is anomaly
        if row.get("salary") and row["salary"] > 500000:
            return True
        return False

    def _count_errors(self) -> int:
        """Count all remaining errors."""
        count = 0
        for row in self.data:
            # Format errors
            if isinstance(row.get("join_date"), str):
                if not self._is_valid_date(row["join_date"]):
                    count += 1
            # Missing values
            if row.get("salary") is None:
                count += 1
        
        # Duplicates (count once per set)
        seen = set()
        for i, row in enumerate(self.data):
            row_tuple = tuple(sorted(row.items()))
            if row_tuple in seen:
                count += 1
            seen.add(row_tuple)
        
        # Anomalies
        for i in range(len(self.data)):
            if self._is_anomaly(i):
                count += 1
        
        return max(0, count)
