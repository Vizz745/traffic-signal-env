import os
import requests
import uuid
from typing import List, Optional
from openai import OpenAI

from tasks import EpisodeRecord, grade_task1, grade_task2, grade_task3

# ---------------- CONFIG ----------------
ENV_URL = os.environ.get("ENV_URL", "https://Vijay745-traffic-signal-env.hf.space")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

BENCHMARK = "traffic_env"

# ---------------- LOGGING ----------------
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True
    )

def log_end(success, steps, score, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True
    )

# ---------------- DECISION ----------------
def heuristic_decide(obs):
    ns = obs["north_queue"] + obs["south_queue"]
    ew = obs["east_queue"] + obs["west_queue"]
    return "NS_GREEN" if ns >= ew else "EW_GREEN"

def llm_decide(obs):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": str(obs)}],
            max_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.upper()
        if "EW" in raw:
            return "EW_GREEN"
        elif "NS" in raw:
            return "NS_GREEN"
    except:
        pass
    return heuristic_decide(obs)

# ---------------- EPISODE ----------------
def run_task(task_id):

    record = EpisodeRecord(task_id)
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task_id, BENCHMARK, MODEL_NAME)

    try:
        r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
        data = r.json()

        session_id = data.get("session_id", str(uuid.uuid4()))
        headers = {"x-session-id": session_id}

        obs = data["observation"]
        done = False

        while not done:
            steps_taken += 1
            record.update(type("O", (), obs)())

            phase = llm_decide(obs)

            try:
                r = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": {"phase": phase, "task_id": task_id}},
                    headers=headers,
                )
                data = r.json()
                obs = data["observation"]
                done = data.get("done", False) or obs.get("done", False)

                # 🔥 REAL REWARD
                reward = (obs.get("throughput", 0) * 0.5) - (obs.get("total_wait_time", 0) * 0.05)
                reward = max(-1.0, min(1.0, reward))

            except Exception as e:
                reward = 0.0
                done = True
                log_step(steps_taken, phase, reward, True, str(e))
                rewards.append(reward)
                break

            rewards.append(reward)
            log_step(steps_taken, phase, reward, done)

        # GRADING
        if task_id == "task1":
            score = grade_task1(record)
        elif task_id == "task2":
            score = grade_task2(record, [], 0, 0)
        else:
            score = grade_task3(record, [], 0, 0)

        score = max(0.01, min(0.99, score))
        success = score > 0.3

    finally:
        log_end(success, steps_taken, score, rewards)

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    for task in ["task1", "task2", "task3"]:
        run_task(task)