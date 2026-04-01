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

An OpenEnv-compatible environment for training and evaluating LLM agents
on urban traffic signal control with emergency vehicle priority.

## Environment Description

The agent controls a single 4-way intersection. Each step it observes
queue lengths on all 4 arms and must decide which axis gets the green light:
`NS_GREEN` (North+South) or `EW_GREEN` (East+West).

## Action Space

| Field | Type | Values |
|-------|------|--------|
| phase | string | `NS_GREEN` or `EW_GREEN` |
| task_id | string | `task1`, `task2`, `task3` |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| north_queue | int | Waiting vehicles on North arm |
| south_queue | int | Waiting vehicles on South arm |
| east_queue | int | Waiting vehicles on East arm |
| west_queue | int | Waiting vehicles on West arm |
| current_phase | string | Active signal phase |
| phase_duration | int | Steps current phase has been active |
| emergency_direction | string | Direction of emergency vehicle, or null |
| total_wait_time | float | Cumulative vehicle waiting time |
| throughput | int | Total vehicles cleared this episode |
| hint | string | Plain-English situation summary |

## Tasks

| Task | Difficulty | Steps | Emergency | Description |
|------|-----------|-------|-----------|-------------|
| task1 | Easy | 30 | None | Rush hour, heavy N/S traffic |
| task2 | Medium | 40 | 12%/step | Balanced load + random emergencies |
| task3 | Hard | 60 | 15%/step | Rush hour all arms + fairness constraint |

## Reward Function

- `+0.1` per vehicle cleared
- `-0.02` per waiting vehicle per step
- `+0.5` when emergency vehicle's path is green
- `-0.4` when emergency vehicle's path is blocked
- `-0.05` per phase switch (discourages jitter)
- `-0.15` per arm exceeding 15 vehicles (starvation penalty)

## Baseline Scores (heuristic agent)

| Task | Score |
|------|-------|
| task1 | 0.9548 |
| task2 | 0.6800 |
| task3 | 0.8913 |
| mean | 0.8420 |

## Setup
```bash
pip install openenv-core
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Run Baseline
```bash
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

## Docker
```bash
docker build -t traffic-signal-env .
docker run -p 8000:8000 traffic-signal-env
```