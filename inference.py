"""
inference.py — Baseline LLM agent for Traffic Signal Control Environment.
"""

import os
import json
import argparse
import requests
import uuid
from openai import OpenAI

from tasks import EpisodeRecord, grade_task1, grade_task2, grade_task3

# ---------------- GLOBAL STATE ----------------
LAST_EMERGENCY = None

# ---------------- CONFIG ----------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL      = os.environ.get("ENV_URL", "https://Vijay745-traffic-signal-env.hf.space")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are a traffic signal controller at a 4-way intersection.
Each step output EXACTLY one of these two strings, nothing else:
  NS_GREEN
  EW_GREEN

Always prioritize emergency vehicles.
"""

# ---------------- HEURISTIC ----------------
def heuristic_decide(obs):
    global LAST_EMERGENCY

    # 🚑 Emergency handling with memory
    if obs.get("emergency_direction"):
        LAST_EMERGENCY = obs["emergency_direction"]
        return "NS_GREEN" if LAST_EMERGENCY in ["N", "S"] else "EW_GREEN"

    if LAST_EMERGENCY:
        if obs["phase_duration"] < 4:
            return "NS_GREEN" if LAST_EMERGENCY in ["N", "S"] else "EW_GREEN"
        else:
            LAST_EMERGENCY = None

    # Normal logic
    ns = obs["north_queue"] + obs["south_queue"] + obs.get("predicted_north", 0) + obs.get("predicted_south", 0)
    ew = obs["east_queue"] + obs["west_queue"] + obs.get("predicted_east", 0) + obs.get("predicted_west", 0)

    if obs["phase_duration"] < 3:
        return obs["current_phase"]

    return "NS_GREEN" if ns >= ew else "EW_GREEN"

# ---------------- LLM ----------------
def llm_decide(obs):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(obs)},
            ],
            max_tokens=10,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip().upper()

        if "EW" in raw:
            return "EW_GREEN"
        elif "NS" in raw:
            return "NS_GREEN"
        else:
            return heuristic_decide(obs)

    except Exception:
        return heuristic_decide(obs)

# ---------------- MAIN ----------------
def run_task(task_id):
    record = EpisodeRecord(task_id=task_id)

    response_steps = []
    cleared = 0
    total_emg = 0
    emg_active = False
    current_emg_steps = 0

    # RESET
    r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    data = r.json()

    session_id = data.get("session_id", str(uuid.uuid4()))
    headers = {"x-session-id": session_id}

    obs = data["observation"]
    done = data.get("done", False) or obs.get("done", False)

    while not done:
        record.update(type("O", (), obs)())

        # Emergency tracking
        if obs.get("emergency_direction"):
            if not emg_active:
                emg_active = True
                total_emg += 1
                current_emg_steps = 0
            current_emg_steps += 1
        elif emg_active:
            emg_active = False
            cleared += 1
            response_steps.append(current_emg_steps)

        # 🔥 HYBRID DECISION (FIXED)
        llm_phase = llm_decide(obs)
        heuristic_phase = heuristic_decide(obs)

        if obs.get("emergency_direction"):
            phase = heuristic_phase
        elif llm_phase == heuristic_phase:
            phase = llm_phase
        else:
            phase = heuristic_phase

        # STEP
        r = requests.post(
            f"{ENV_URL}/step",
            json={"action": {"phase": phase, "task_id": task_id}},
            headers=headers,
        )

        data = r.json()
        obs = data["observation"]
        done = data.get("done", False) or obs.get("done", False)

    # GRADING
    if task_id == "task1":
        return grade_task1(record)
    elif task_id == "task2":
        return grade_task2(record, response_steps, cleared, total_emg)
    else:
        return grade_task3(record, response_steps, cleared, total_emg)

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["task1", "task2", "task3"], default="task1")
    args = parser.parse_args()

    score = run_task(args.task)

    print("[START]")
    print(json.dumps({
        "task": args.task,
        "score": score,
        "env_url": ENV_URL,
        "model": MODEL_NAME,
    }))
    print("[END]")