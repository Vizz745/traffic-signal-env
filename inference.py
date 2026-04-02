"""
inference.py — Baseline LLM agent for Traffic Signal Control Environment.
Required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN

Usage:
    python inference.py --task task1
    python inference.py --task task2
    python inference.py --task task3
"""
import os
import sys
import json
import argparse
import requests
from openai import OpenAI
import uuid

from tasks import EpisodeRecord, grade_task1, grade_task2, grade_task3

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL      = os.environ.get("ENV_URL", "https://Vijay745-traffic-signal-env.hf.space")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are a traffic signal controller at a 4-way intersection.
Each step output EXACTLY one of these two strings, nothing else:
  NS_GREEN
  EW_GREEN

NS_GREEN = North and South arms get green light.
EW_GREEN = East and West arms get green light.
Vehicles clear at 3 per green arm per step.
Always prioritize clearing emergency vehicles immediately."""


def llm_decide(obs: dict) -> str:
    user_msg = f"""Intersection state:
- North queue: {obs['north_queue']} vehicles (predicted next: {obs.get('predicted_north', 0)})
- South queue: {obs['south_queue']} vehicles (predicted next: {obs.get('predicted_south', 0)})
- East queue:  {obs['east_queue']} vehicles (predicted next: {obs.get('predicted_east', 0)})
- West queue:  {obs['west_queue']} vehicles (predicted next: {obs.get('predicted_west', 0)})
- Current phase: {obs['current_phase']} (active {obs['phase_duration']} steps)
- Emergency vehicle: {obs.get('emergency_direction') or 'None'}
- Hint: {obs['hint']}

Your decision (NS_GREEN or EW_GREEN):"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip().upper()
        return "EW_GREEN" if "EW" in raw else "NS_GREEN"
    except Exception:
        # Fallback heuristic
        ns = obs['north_queue'] + obs['south_queue']
        ew = obs['east_queue']  + obs['west_queue']
        return "NS_GREEN" if ns >= ew else "EW_GREEN"


def run_task(task_id: str) -> float:
    record = EpisodeRecord(task_id=task_id)
    response_steps = []
    cleared = 0
    total_emg = 0
    emg_active = False
    current_emg_steps = 0

    # Reset and capture session ID
    r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    data = r.json()
    session_id = data.get("session_id", str(uuid.uuid4()))
    headers = {"x-session-id": session_id}
    obs = data["observation"]
    done = data.get("done", False) or obs.get("done", False)

    while not done:
        record.update(type("O", (), obs)())

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

        phase = llm_decide(obs)
        r = requests.post(
            f"{ENV_URL}/step",
            json={"action": {"phase": phase, "task_id": task_id}},
            headers=headers,
        )
        data = r.json()
        obs = data["observation"]
        done = data.get("done", False) or obs.get("done", False)

    if task_id == "task1":
        return grade_task1(record)
    elif task_id == "task2":
        return grade_task2(record, response_steps, cleared, total_emg)
    else:
        return grade_task3(record, response_steps, cleared, total_emg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="task1",
        choices=["task1", "task2", "task3"],
        help="Which task to run"
    )
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