"""
FastAPI server — exposes OpenEnv REST endpoints.
Judges ping this URL directly.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from environment import DataCleanEnv
from models import Action
import uvicorn

app = FastAPI(
    title="DataClean OpenEnv",
    description="Enterprise Data Pipeline Quality Agent Environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env instance per task (simple, stateless enough for hackathon)
_envs: dict[str, DataCleanEnv] = {}


def _get_env(task: str) -> DataCleanEnv:
    if task not in ("format_fix", "imputation", "pipeline"):
        raise HTTPException(400, f"Unknown task '{task}'. Choose: format_fix | imputation | pipeline")
    if task not in _envs:
        _envs[task] = DataCleanEnv(task_name=task)
    return _envs[task]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":  "DataClean OpenEnv",
        "tasks": ["format_fix", "imputation", "pipeline"],
        "spec":  "openenv-v1",
    }


@app.post("/reset")
def reset(task: str = "format_fix"):
    env = _get_env(task)
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: Action, task: str = "format_fix"):
    env = _get_env(task)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {
        "observation": obs.model_dump(),
        "reward":      reward,
        "done":        done,
        "info":        info,
    }


@app.get("/state")
def state(task: str = "format_fix"):
    env = _get_env(task)
    return env.state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name":        "format_fix",
                "difficulty":  "easy",
                "description": "Fix date and salary formats to standard form.",
                "max_score":   1.0,
            },
            {
                "name":        "imputation",
                "difficulty":  "medium",
                "description": "Fill missing values using department-level group means.",
                "max_score":   1.0,
            },
            {
                "name":        "pipeline",
                "difficulty":  "hard",
                "description": "Clean data in correct pipeline order. Includes a poison cell trap.",
                "max_score":   1.0,
            },
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)