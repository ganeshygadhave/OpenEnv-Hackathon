"""
Inference Script — DataClean OpenEnv
=====================================
Runs an LLM agent against all 3 tasks and emits [START][STEP][END] logs.

MANDATORY ENV VARS:
  API_BASE_URL  — LLM endpoint (default: HF router)
  MODEL_NAME    — model identifier
  HF_TOKEN      — your Hugging Face API key
"""
import os, json, textwrap
from typing import Optional
from openai import OpenAI
from environment import DataCleanEnv
from models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN")     or os.getenv("API_KEY", "")
BENCHMARK    = "dataclean-openenv"
MAX_STEPS    = 10
TEMPERATURE  = 0.2
MAX_TOKENS   = 512

TASKS = ["format_fix", "imputation", "pipeline"]

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ── Logging helpers ────────────────────────────────────────────────────────────

def log_start(task: str, model: str):
    print("[START] task={} env={} model={}".format(task, BENCHMARK, model), flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(
        "[STEP] step={} action={} reward={:.2f} done={} error={}".format(
            step, action, reward, str(done).lower(), err
        ),
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list[float]):
    r_str = ",".join("{:.2f}".format(r) for r in rewards)
    print(
        "[END] success={} steps={} score={:.2f} rewards={}".format(
            str(success).lower(), steps, score, r_str
        ),
        flush=True,
    )


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a data cleaning agent interacting with the DataClean OpenEnv environment.

On each turn you receive the current dataset state and must return a JSON action.

Available operations:
- fix_format:       Fix date (→ YYYY-MM-DD) or salary (→ plain integer) formats
- impute:           Fill a missing (None) value with a reasonable number
- remove_duplicate: Remove an exact duplicate row
- flag_anomaly:     Flag a row with an impossible/extreme value
- done:             End the episode when all errors are fixed

Your response must be ONLY a valid JSON object like:
{
  "operation": "fix_format",
  "target_rows": [2],
  "target_column": "join_date",
  "new_value": "2019-03-15",
  "reason": "date was in DD-MM-YYYY format"
}

Rules:
- Only fix real errors — false positives are penalized
- For Task 3 (pipeline): fix in order: fix_format → impute → remove_duplicate → flag_anomaly
- Not all high values are anomalies — reason carefully
- Call done when finished
""").strip()


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_task(task_name: str) -> float:
    env     = DataCleanEnv(task_name=task_name)
    obs     = env.reset()
    rewards = []
    step    = 0
    done    = False
    score   = 0.0
    last_error: Optional[str] = None

    log_start(task_name, MODEL_NAME)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Task: {task_name}\n"
            f"Hint: {obs.hint}\n"
            f"Current data:\n{json.dumps(obs.current_data, indent=2)}\n"
            f"Errors remaining: {obs.errors_remaining}\n"
            "Return your action as JSON."
        )},
    ]

    while not done and step < MAX_STEPS:
        step += 1

        try:
            response = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = messages,
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON action
            # Strip markdown fences if present
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            action_dict = json.loads(raw)
            action      = Action(**action_dict)
            action_str  = action.operation

        except Exception as e:
            last_error = str(e)
            log_step(step, "parse_error", 0.0, False, last_error)
            rewards.append(0.0)
            continue

        # Step environment
        try:
            obs, reward, done, info = env.step(action)
            last_error = None
        except Exception as e:
            last_error = str(e)
            reward     = 0.0
            done       = True

        rewards.append(reward)
        log_step(step, action_str, reward, done, last_error)

        if not done:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": (
                f"Result: {obs.last_action_result}\n"
                f"Current data:\n{json.dumps(obs.current_data, indent=2)}\n"
                f"Errors remaining: {obs.errors_remaining}\n"
                "Return your next action as JSON."
            )})

    score   = sum(rewards) / max(len(rewards), 1)
    success = score >= 0.5

    env.close()
    log_end(success, step, score, rewards)
    return score


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total = 0.0
    for task in TASKS:
        s = run_task(task)
        total += s
    avg = total / len(TASKS)
    print(f"\nOverall average score: {avg:.2f}", flush=True)