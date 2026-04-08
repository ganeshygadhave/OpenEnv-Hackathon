# models.py
from typing import Optional
from pydantic import BaseModel

class Observation(BaseModel):
    task_name: str
    step: int
    current_data: list
    errors_remaining: int
    last_action_result: Optional[str] = None
    hint: Optional[str] = None

class Action(BaseModel):
    operation: str
    target_rows: list
    target_column: str
    new_value: Optional[str] = None
    reason: Optional[str] = None

class Reward(BaseModel):
    value: float
    correctness: float
    order_bonus: float
    efficiency_bonus: float
    penalty: float
    breakdown: dict