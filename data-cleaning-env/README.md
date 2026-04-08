# DataClean OpenEnv

**Real-world Data Cleaning Environment for AI Agents**

A reinforcement learning environment where AI agents learn to clean enterprise datasets by fixing format errors, imputing missing values, removing duplicates, and detecting anomalies.

## 🎯 Overview

DataClean OpenEnv simulates real-world data quality challenges that engineers face daily. Instead of toy problems, agents interact with realistic datasets containing multiple error types across three escalating tasks:

1. **Format Fixing (Easy)**: Fix date and salary formats
2. **Missing Value Imputation (Medium)**: Fill missing values using statistical methods
3. **Multi-step Pipeline (Hard)**: Execute cleaning steps in strict order with poison cell traps

The environment is fully compliant with the OpenEnv specification and designed for benchmarking LLM-based and RL agents.

---

## 📋 Environment Specification

### Task: Format Fixing (Easy)

**Objective**: Fix malformed dates and salaries in 4 employee records.

**Errors to Fix**: 4 format errors (rows 0-3)
- Row 0: join_date "03-15-2019" -> "2019-03-15", salary "$75,000.00" -> 75000
- Row 1: join_date "2020/07/22" -> "2020-07-22", salary "82500" (correct)
- Row 2: join_date "2021-11-10" (correct), salary "$95,000.50" -> 95000
- Row 3: join_date "15/11/2018" -> "2018-11-15", salary "68,500" -> 68500

**Operations**: `fix_format`, `done`

**Reward**: +1.0 for fixing error, -0.5 for false positive, 0.0 for no-op

---

### Task: Missing Value Imputation (Medium)

**Objective**: Fill 2 missing salary values using department-level means.

**Errors to Fix**: 2 missing values (rows 1, 3)
- Row 1 (Bob, Sales): None -> ~77,500 (Sales mean)
- Row 3 (Diana, Engineering): None -> ~95,000 (Engineering mean)

**Operations**: `impute`, `done`

**Reward**: +1.0 for correctly imputing, -0.5 for false positive, 0.0 for no-op

---

### Task: Multi-step Pipeline (Hard)

**Objective**: Execute 4 cleaning steps in strict order.

**Operations** (must be done in order):
1. `fix_format` — Fix date formats
2. `impute` — Fill missing salaries
3. `remove_duplicate` — Remove exact duplicate rows
4. `flag_anomaly` — Identify anomalous values
5. `done` — Complete episode

**Reward**: +1.0 per correct step, -0.5 for violations/false positives, 0.0 for no-op

---

## 🛠 Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Python API

```python
from environment import DataCleanEnv
from models import Action

env = DataCleanEnv(task_name="format_fix")
obs = env.reset()

action = Action(
    operation="fix_format",
    target_rows=[0],
    target_column="join_date",
    new_value="2019-03-15"
)

obs, reward, done, info = env.step(action)
env.close()
```

### REST API

```bash
python server.py
# http://localhost:7860
```

### Baseline Inference

```bash
export HF_TOKEN="your-token"
python inference.py
```

---

## 📊 Spaces

### Observation Space
**Typed Pydantic model** with fields:
- `task_name`: str - Task identifier
- `step`: int - Current step number
- `current_data`: list - Dataset rows
- `errors_remaining`: int - Count of remaining errors
- `last_action_result`: Optional[str] - Feedback from previous action
- `hint`: Optional[str] - Task guidance

### Action Space
**Typed Pydantic model** with fields:
- `operation`: str - Operation type (fix_format, impute, etc)
- `target_rows`: list - Row indices to modify
- `target_column`: str - Column to fix
- `new_value`: Optional[str] - Replacement value
- `reason`: Optional[str] - Explanation

### Reward Space
**Typed Pydantic model** with fields:
- `value`: float - Overall reward [0.0-1.0]
- `correctness`: float - Correctness bonus
- `order_bonus`: float - Bonus for correct order
- `efficiency_bonus`: float - Efficiency bonus
- `penalty`: float - Penalties for errors
- `breakdown`: dict - Detailed metrics

---

## 🐳 Docker Deployment

The included Dockerfile provides containerized deployment:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "server.py"]
```

Build and run locally:

```bash
docker build -t dataclean-openenv .
docker run -p 7860:7860 dataclean-openenv
```

---

## 📈 Baseline Scores

Example reproducible scores using Qwen/Qwen2.5-72B-Instruct:

| Task | Avg Reward | Success Rate |
|------|-----------|--------------|
| format_fix | 0.95 | 95% |
| imputation | 0.88 | 88% |
| pipeline | 0.72 | 72% |

---

## 📝 License

Open source for hackathon evaluation.
