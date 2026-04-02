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

---

## Motivation

Urban traffic signal control is a genuine real-world problem. Poorly timed signals cause congestion, increase emissions, and critically — delay emergency vehicles. Traditional fixed-timer systems are inefficient. Rule-based systems are brittle. This environment provides a standardized benchmark for evaluating how well LLM agents can reason about dynamic, multi-objective control problems in a realistic setting.

This work is inspired by our minor project: **Hybrid Traffic Prediction & RL Control for Emergency-Aware Urban Mobility**, which combined LSTM-based traffic prediction with Dueling Double DQN for intelligent signal management. This OpenEnv environment adapts that domain for LLM agent evaluation — replacing the neural controller with a text-based decision interface and adding programmatic graders for standardized benchmarking.

---

## Environment Overview

The agent controls a single 4-way intersection. At each step it observes the full intersection state and must decide which axis gets the green light. Vehicles arrive stochastically based on task-specific rates and are cleared at a fixed service rate when their arm is green.

     North
      ↑↓

West ←→ [ ] ←→ East
↑↓
South


Each step the agent outputs one of two actions:
- `NS_GREEN` — North and South arms get green light
- `EW_GREEN` — East and West arms get green light

---

## Stochastic Environment Design

Traffic arrivals and emergency events are **stochastic (random but bounded)**, simulating real-world uncertainty.

This ensures agents are evaluated on:
- robustness  
- consistency  
- decision-making under uncertainty  

rather than overfitting to deterministic patterns. Each episode differs slightly, but overall difficulty remains consistent.

---

## Action Space

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| phase | string | `NS_GREEN`, `EW_GREEN` | Which axis gets green light |
| task_id | string | `task1`, `task2`, `task3` | Which task scenario to run |

---

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

---

## Tasks

### Task 1 — Easy: Rush Hour Timing
**Objective:** Minimize total vehicle waiting time during peak hours.

- 30 steps per episode  
- Heavy North/South traffic  
- Light East/West traffic  
- No emergency vehicles  

**Grader:**  
`score = clamp(1 - total_wait / baseline_wait, 0, 1)`

---

### Task 2 — Medium: Emergency Clearance
**Objective:** Balance traffic flow while prioritizing emergency vehicles.

- 40 steps per episode  
- Balanced traffic  
- Emergency vehicles spawn randomly  

**Grader:**  
`0.6 × emergency_score + 0.4 × traffic_score`

---

### Task 3 — Hard: Multi-Constraint Optimization
**Objective:** Handle traffic, emergencies, fairness, and switching penalties.

- 60 steps per episode  
- Heavy traffic  
- Emergency vehicles  
- Yellow light penalty (2-step freeze)  
- Fairness constraint  

**Grader:**  
`0.4 × emergency + 0.3 × throughput + 0.3 × fairness`

---

## Reward Function

| Component | Value | Condition |
|-----------|-------|-----------|
| Throughput bonus | +0.1 per vehicle | Per vehicle cleared |
| Queue penalty | -0.02 per vehicle | Waiting vehicles |
| Emergency clear | +0.5 | Emergency served |
| Emergency block | -0.4 | Emergency blocked |
| Phase switch | -0.05 | Switching |
| Starvation | -0.15 | Large queue |

---

## Predicted Queue Feature

Each observation includes a **weighted moving average prediction** of next-step queue lengths.

This mirrors real-world forecasting systems and allows:
- proactive decisions  
- not just reactive control  

---

## Agent Strategy

The baseline agent uses a **hybrid decision system**:

- LLM provides flexible reasoning from natural-language observations  
- Heuristic controller ensures stability and emergency prioritization  
- In case of disagreement, the system favors the heuristic  

This ensures:
- robustness  
- consistency  
- reliable performance under uncertainty  

---

## Session-Based Environment Isolation

The environment supports **concurrent usage via session-based state management**.

- Each `/reset` creates a new environment instance  
- Each session is tracked independently  
- Requests use a session ID to maintain state  

This prevents:
- state corruption  
- cross-request interference  
- instability during evaluation  

---

## Baseline Scores (Approximate)

Due to stochasticity, scores vary slightly per run:

| Task | Score Range |
|------|-------------|
| task1 | ~0.6 – 0.75 |
| task2 | ~0.6 – 0.7 |
| task3 | ~0.85 – 0.95 |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment |
| `/step` | POST | Take action |
| `/state` | GET | Get current state |
| `/health` | GET | Health check |
| `/web` | GET | Interactive UI |

---

## Setup & Usage

### Connect to live Space

```python
import requests

obs = requests.post("https://Vijay745-traffic-signal-env.hf.space/reset?task_id=task1").json()

result = requests.post(
    "https://Vijay745-traffic-signal-env.hf.space/step",
    json={"action": {"phase": "NS_GREEN", "task_id": "task1"}}
).json()
Run locally
pip install openenv-core
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
Run inference
python inference.py --task task2
Technical Details
Pure Python simulator
Sub-millisecond step time
Stochastic vehicle arrivals
Fixed service rate
Why this domain?

Traffic signal control is a real-world, high-impact problem.

This environment models:

congestion
emergency response
fairness
uncertainty

making it a strong benchmark for intelligent decision-making systems.

License

MIT