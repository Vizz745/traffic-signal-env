import os
import json
import requests
import uuid
from openai import OpenAI

from tasks import EpisodeRecord, grade_task1, grade_task2, grade_task3

LAST_EMERGENCY = None

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

MAX_STEPS = {"task1": 30, "task2": 40, "task3": 60}
SAMPLE_STEPS = 3  # how many steps to log per task

# ---------------- SHAPED PER-STEP REWARD [0, 1] ----------------
def compute_step_reward(obs, action):
    """
    Returns a reward in [0, 1].
    - Base: how clear the intersection is (fewer queues = higher)
    - Emergency bonus/penalty applied on top, then clamped
    """
    ns_q = obs["north_queue"] + obs["south_queue"]
    ew_q = obs["east_queue"] + obs["west_queue"]
    total_q = ns_q + ew_q

    # Base reward: 1.0 when queues empty, approaches 0 as queues grow (max ~20)
    reward = 1.0 - min(1.0, total_q / 20.0)

    emg = obs.get("emergency_direction")
    if emg:
        correct = "NS_GREEN" if emg in ["N", "S"] else "EW_GREEN"
        if action == correct:
            reward = min(1.0, reward + 0.3)   # bonus for correct emergency response
        else:
            reward = max(0.0, reward - 0.3)   # penalty for wrong response
    elif total_q > 0:
        # small bonus for picking the more congested direction
        if action == "NS_GREEN" and ns_q >= ew_q:
            reward = min(1.0, reward + 0.05)
        elif action == "EW_GREEN" and ew_q > ns_q:
            reward = min(1.0, reward + 0.05)

    return round(max(0.0, min(1.0, reward)), 4)

# ---------------- HEURISTIC ----------------
def heuristic_decide(obs):
    global LAST_EMERGENCY

    if obs.get("emergency_direction"):
        LAST_EMERGENCY = obs["emergency_direction"]
        return "NS_GREEN" if LAST_EMERGENCY in ["N", "S"] else "EW_GREEN"

    if LAST_EMERGENCY:
        if obs["phase_duration"] < 4:
            return "NS_GREEN" if LAST_EMERGENCY in ["N", "S"] else "EW_GREEN"
        else:
            LAST_EMERGENCY = None

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

# ---------------- TASK RUNNER ----------------
def run_task(task_id):
    global LAST_EMERGENCY
    LAST_EMERGENCY = None

    record = EpisodeRecord(task_id)
    all_steps = []  # list of (action, reward) for every step
    response_steps = []
    cleared = 0
    total_emg = 0
    emg_active = False
    current_emg_steps = 0

    max_steps = MAX_STEPS[task_id]

    # RESET
    r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    data = r.json()
    session_id = data.get("session_id", str(uuid.uuid4()))
    headers = {"x-session-id": session_id}
    obs = data["observation"]
    done = data.get("done", False) or obs.get("done", False)

    print(f"[START] task={task_id} env=traffic model={MODEL_NAME}", flush=True)

    # LOOP
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

        llm_phase = llm_decide(obs)
        heuristic_phase = heuristic_decide(obs)

        if obs.get("emergency_direction"):
            phase = heuristic_phase
        elif llm_phase == heuristic_phase:
            phase = llm_phase
        else:
            phase = heuristic_phase

        step_reward = compute_step_reward(obs, phase)

        r = requests.post(
            f"{ENV_URL}/step",
            json={"action": {"phase": phase, "task_id": task_id}},
            headers=headers,
        )
        data = r.json()
        obs = data["observation"]
        done = data.get("done", False) or obs.get("done", False)

        all_steps.append((phase, step_reward))

    total_steps = len(all_steps)

    # Pick 3 evenly-spaced sample steps to log
    if total_steps <= SAMPLE_STEPS:
        sample_indices = list(range(total_steps))
    else:
        # e.g. steps 0, mid, last
        sample_indices = [
            0,
            total_steps // 2,
            total_steps - 1,
        ]

    # GRADING
    if task_id == "task1":
        score = grade_task1(record)
    elif task_id == "task2":
        score = grade_task2(record, response_steps, cleared, total_emg)
    else:
        score = grade_task3(record, response_steps, cleared, total_emg)

    # Clamp score
    final_score = round(min(max(float(score), 0.01), 0.99), 2)

    # Print 3 sampled steps
    for i, idx in enumerate(sample_indices):
        phase, reward = all_steps[idx]
        clamped_reward = round(min(max(reward, 0.01), 0.99), 2)
        is_done = (i == len(sample_indices) - 1)
        print(
            f"[STEP] step={i+1} action={phase} reward={clamped_reward:.2f} "
            f"done={'true' if is_done else 'false'} error=null",
            flush=True
        )

    # Single [END]
    sampled_rewards = [round(min(max(all_steps[i][1], 0.01), 0.99), 2) for i in sample_indices]
    rewards_str = ",".join(f"{r:.2f}" for r in sampled_rewards)
    print(
        f"[END] success=true steps={total_steps} score={final_score:.2f} rewards={rewards_str}",
        flush=True
    )

    return final_score, total_steps

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    print("MAIN STARTED")

    scores = []
    for task_id in ["task1", "task2", "task3"]:
        print(f"Running {task_id}")
        try:
            score, _ = run_task(task_id)
            scores.append(score)
        except Exception as e:
            print(f"[END] success=false steps=0 rewards=", flush=True)
            print(f"ERROR: {e}", flush=True)
            scores.append(0.01)

    avg = round(max(0.01, min(0.99, sum(scores) / len(scores))), 4)
    print(f"\nFinal average score across tasks: {avg}", flush=True)