import os
import uuid
from typing import List

import requests
from openai import OpenAI

from tasks import EpisodeRecord, grade_task1, grade_task2, grade_task3

ENV_URL = os.environ.get("ENV_URL", "https://Vijay745-traffic-signal-env.hf.space")
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY = os.environ["API_KEY"]

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

BENCHMARK = "traffic_env"


def clamp_score(value: float) -> float:
    return max(0.001, min(0.999, round(float(value), 4)))


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.3f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success, steps, score, rewards: List[float]):
    rewards_str = ",".join(f"{r:.3f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def heuristic_decide(obs):
    ns = obs.get("north_queue", 0) + obs.get("south_queue", 0)
    ew = obs.get("east_queue", 0) + obs.get("west_queue", 0)
    task_id = obs.get("task_id", "task1")
    emergency = obs.get("emergency_direction")
    current_phase = obs.get("current_phase", "NS_GREEN")

    if emergency in ("N", "S"):
        return "NS_GREEN"
    if emergency in ("E", "W"):
        return "EW_GREEN"

    if task_id == "task1":
        return "NS_GREEN" if ns >= ew - 2 else "EW_GREEN"

    if task_id == "task2":
        return "NS_GREEN" if ns >= ew else "EW_GREEN"

    if task_id == "task3":
        if ns > ew + 2:
            return "NS_GREEN"
        if ew > ns + 2:
            return "EW_GREEN"
        return current_phase

    return "NS_GREEN" if ns >= ew else "EW_GREEN"


def llm_decide(obs):
    heuristic = heuristic_decide(obs)

    prompt = f"""
You are controlling a traffic signal at a 4-way intersection.

Return exactly one action:
NS_GREEN
or
EW_GREEN

Current observation:
{obs}

Heuristic suggestion: {heuristic}

Rules:
- Prioritize the direction of any active emergency vehicle.
- Reduce total queueing and waiting time.
- Reply with exactly one token: NS_GREEN or EW_GREEN
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip().upper()

        if "EW_GREEN" in raw:
            return "EW_GREEN"
        if "NS_GREEN" in raw:
            return "NS_GREEN"
    except Exception:
        pass

    return heuristic


def compute_fallback_score(task_id: str, record: EpisodeRecord) -> float:
    if task_id == "task1":
        return clamp_score(grade_task1(record))
    if task_id == "task2":
        return clamp_score(
            grade_task2(
                record,
                record.response_times,
                record.emergency_cleared,
                max(record.total_emergencies, 1),
            )
        )
    return clamp_score(
        grade_task3(
            record,
            record.response_times,
            record.emergency_cleared,
            max(record.total_emergencies, 1),
        )
    )


def run_task(task_id):
    record = EpisodeRecord(task_id)
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.001

    log_start(task_id, BENCHMARK, MODEL_NAME)

    try:
        r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        data = r.json()

        session_id = data.get("session_id", str(uuid.uuid4()))
        headers = {"x-session-id": session_id}

        obs = data["observation"]
        done = False

        while not done:
            steps_taken += 1
            phase = llm_decide(obs)

            try:
                r = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": {"phase": phase, "task_id": task_id}},
                    headers=headers,
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
                obs = data["observation"]
                done = bool(data.get("done", False) or obs.get("done", False))

                record.update(type("O", (), obs)())

                reward = round(float(data.get("reward", obs.get("reward", 0.0))), 4)

                if done and "score" in data:
                    score = clamp_score(data["score"])

            except Exception as e:
                reward = -0.1
                done = True
                log_step(steps_taken, phase, reward, True, str(e))
                rewards.append(reward)
                break

            rewards.append(reward)
            log_step(steps_taken, phase, reward, done)

        if score <= 0.001:
            score = compute_fallback_score(task_id, record)

        score = clamp_score(score)
        success = score > 0.3

    except Exception:
        score = 0.001
        success = False

    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    for task in ["task1", "task2", "task3"]:
        run_task(task)
