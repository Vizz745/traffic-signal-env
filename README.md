---
title: Traffic Signal Control Environment
emoji: 🚦
colorFrom: green
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - traffic
---

# Traffic Signal Control Environment

An OpenEnv-compatible reinforcement learning environment for training and evaluating LLM agents on **emergency-aware urban traffic signal control**.

Inspired by real-world intelligent transportation systems, this environment challenges an agent to manage a 4-way intersection under varying traffic loads and emergency vehicle scenarios — balancing throughput, fairness, and emergency response simultaneously.

## Motivation

Urban traffic signal control is a genuine real-world problem. Poorly timed signals cause congestion, increase emissions, and critically — delay emergency vehicles. Traditional fixed-timer systems are inefficient. Rule-based systems are brittle. This environment provides a standardized benchmark for evaluating how well LLM agents can reason about dynamic, multi-objective control problems in a realistic setting.

This work is inspired by our minor project: **Hybrid Traffic Prediction & RL Control for Emergency-Aware Urban Mobility**, which combined LSTM-based traffic prediction with Dueling Double DQN for intelligent signal management. This OpenEnv environment adapts that domain for LLM agent evaluation — replacing the neural controller with a text-based decision interface and adding programmatic graders for standardized benchmarking.

## Environment Overview

The agent controls a single 4-way intersection. At each step it observes the full intersection state and must decide which axis gets the green light. Vehicles arrive stochastically based on task-specific rates and are cleared at a fixed service rate when their arm is green.
```
         North
          ↑↓
West ←→ [  ] ←→ East  
          ↑↓
         South
```

Each step the agent outputs one of two actions:
- `NS_GREEN` — North and South arms get green light
- `EW_GREEN` — East and West arms get green light

## Action Space

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| phase | string | `NS_GREEN`, `EW_GREEN` | Which axis gets green light |
| task_id | string | `task1`, `task2`, `task3` | Which task scenario to run |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| step | int | Current step number |
| current_phase | string | Active signal phase |
| phase_duration | int | Steps current phase has been active |
| north_queue | int | Waiting vehicles on North arm |
| south_queue | int | Waiting vehicles on South arm |
| east_queue | int | Waiting vehicles on East arm |
| west_queue | int | Waiting vehicles on West arm |
| predicted_north | float | Predicted N queue next step (weighted moving average) |
| predicted_south | float | Predicted S queue next step |
| predicted_east | float | Predicted E queue next step |
| predicted_west | float | Predicted W queue next step |
| emergency_direction | string | Direction of active emergency vehicle, or null |
| emergency_steps_remaining | int | Steps until emergency vehicle clears, or null |
| total_wait_time | float | Cumulative vehicle waiting time this episode |
| throughput | int | Total vehicles cleared this episode |
| task_id | string | Current task identifier |
| hint | string | Plain-English situation summary for the agent |

## Tasks

### Task 1 — Easy: Rush Hour Timing
**Objective:** Minimize total vehicle waiting time during peak hours.

- 30 steps per episode
- Heavy North/South traffic (arrival rate 0.7–0.8 per step)
- Light East/West traffic (arrival rate 0.2 per step)
- No emergency vehicles
- **Grader:** `score = clamp(1 - total_wait / baseline_wait, 0, 1)`
- **Baseline:** 420.0 total wait time for naive 50/50 fixed timer

A good agent learns to strongly favour the NS phase and only switch when EW queues grow too large.

### Task 2 — Medium: Emergency Clearance
**Objective:** Balance traffic flow while prioritizing emergency vehicles.

- 40 steps per episode
- Balanced traffic on all arms (arrival rate 0.4 per step)
- Emergency vehicles spawn randomly (12% probability per step)
- Emergency vehicles appear from random directions
- **Grader:** `0.6 × emergency_score + 0.4 × traffic_score`
  - Emergency score: fraction cleared × speed bonus (faster = higher)
  - Traffic score: throughput vs expected baseline

A good agent learns to immediately switch phase when an emergency vehicle appears, even at the cost of traffic efficiency.

### Task 3 — Hard: Multi-Constraint Optimization
**Objective:** Juggle rush hour load, emergencies, fairness, and yellow light penalties simultaneously.

- 60 steps per episode
- Heavy traffic on all arms (arrival rate 0.6–0.8 per step)
- Emergency vehicles spawn randomly (15% probability per step)
- **Yellow light penalty:** Switching phases freezes the intersection for 2 steps (no vehicles move) — penalizes indecisive switching
- **Fairness constraint:** Penalizes any arm with max queue > 3× the average
- **Grader:** `0.4 × emergency_score + 0.3 × throughput_score + 0.3 × fairness_score`

A good agent must plan phase switches carefully — switching too often wastes 2 steps to yellow light, but switching too rarely causes starvation on some arms.

## Reward Function

Reward is provided at **every step** (not just episode end), giving dense signal throughout the trajectory:

| Component | Value | Condition |
|-----------|-------|-----------|
| Throughput bonus | +0.1 per vehicle | Per vehicle cleared this step |
| Queue penalty | -0.02 per vehicle | Per waiting vehicle across all arms |
| Emergency clear | +0.5 | Emergency arm is currently green |
| Emergency block | -0.4 | Emergency arm is currently red |
| Phase switch | -0.05 | Agent switched phase this step |
| Starvation | -0.15 | Any arm exceeds 15 vehicles |

## Predicted Queue Feature

Each observation includes a **weighted moving average prediction** of next-step queue lengths for all 4 arms, computed from the last 5 steps of history. Recent steps are weighted more heavily than older ones.

This mirrors the LSTM-based traffic prediction from our original research project — giving the agent forward-looking information to make proactive rather than reactive decisions.

## Baseline Scores

Baseline agent uses a simple greedy heuristic: always serve the heavier traffic axis, switch to emergency arm immediately when emergency detected.

| Task | Score | Notes |
|------|-------|-------|
| task1 | 0.9238 | Heuristic handles heavy NS traffic well |
| task2 | 0.8200 | Emergency timing is partially lucky |
| task3 | 0.8957 | Yellow light penalty limits switching frequency |
| **mean** | **0.8798** | |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment, returns initial observation |
| `/reset?task_id=task2` | POST | Reset with specific task |
| `/step` | POST | Execute action, returns observation + reward + done |
| `/state` | GET | Get current episode state |
| `/health` | GET | Health check |
| `/web` | GET | Interactive browser UI |

## Interactive Web UI

Visit `/web` to interact with the environment directly in your browser:
- Select task difficulty
- Click NS_GREEN / EW_GREEN to manually control signals
- Watch queue bars update in real time
- See emergency vehicles appear and clear
- Auto Step mode runs the heuristic agent automatically

**Live demo:** https://Vijay745-traffic-signal-env.hf.space/web

## Setup & Usage

### Connect to live Space
```python
import requests

# Reset environment
obs = requests.post("https://Vijay745-traffic-signal-env.hf.space/reset?task_id=task1").json()

# Step
result = requests.post(
    "https://Vijay745-traffic-signal-env.hf.space/step",
    json={"action": {"phase": "NS_GREEN", "task_id": "task1"}}
).json()
```

### Run locally
```bash
pip install openenv-core
git clone https://huggingface.co/spaces/Vijay745/traffic-signal-env
cd traffic-signal-env
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker
```bash
docker build -t traffic-signal-env .
docker run -p 7860:7860 traffic-signal-env
```

### Run baseline inference
```bash
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

## Project Structure
```
traffic_signal_env/
├── models.py                          # Pydantic Action + Observation types
├── client.py                          # TrafficSignalEnv client
├── tasks.py                           # Task graders (grade_task1/2/3)
├── inference.py                       # Baseline LLM inference script
├── openenv.yaml                       # OpenEnv manifest
├── Dockerfile                         # HuggingFace Spaces container
├── README.md                          
└── server/
    ├── app.py                         # FastAPI server
    ├── traffic_signal_env_environment.py  # Core simulator
    ├── requirements.txt               
    └── Dockerfile                     # Local development container
```

## Technical Details

### Simulator
Pure Python implementation — no external traffic simulators required. Runs in under 1ms per step. Vehicle arrivals modelled as Bernoulli trials per arm per step. Service rate: 3 vehicles per green arm per step.

### Observation Design
The `hint` field provides plain-English context specifically designed for LLM agents — describing urgency, queue imbalances, and phase timing in natural language rather than raw numbers alone.

The predicted queue fields give agents forward-looking information, enabling proactive signal timing rather than purely reactive control.

### Why this domain?
Traffic signal control is a genuine real-world problem affecting millions of people daily. Emergency vehicle response time directly impacts survival outcomes. This environment models both — making it immediately valuable for evaluating agents that must balance competing objectives under time pressure.

## License
MIT