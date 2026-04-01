import random
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TrafficSignalAction, TrafficSignalObservation
except ImportError:
    from models import TrafficSignalAction, TrafficSignalObservation


# ── Task configuration ────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "task1": {
        "max_steps": 30,
        "arrival_rates": {"N": 0.7, "S": 0.8, "E": 0.2, "W": 0.2},
        "emergency_prob": 0.0,
        "service_rate": 3,
    },
    "task2": {
        "max_steps": 40,
        "arrival_rates": {"N": 0.4, "S": 0.4, "E": 0.4, "W": 0.4},
        "emergency_prob": 0.12,
        "service_rate": 3,
    },
    "task3": {
        "max_steps": 60,
        "arrival_rates": {"N": 0.8, "S": 0.7, "E": 0.6, "W": 0.6},
        "emergency_prob": 0.15,
        "service_rate": 3,
    },
}

BASELINE_WAIT = {"task1": 420.0, "task2": 600.0, "task3": 1800.0}


# ── Internal simulation state ─────────────────────────────────────────────────

@dataclass
class IntersectionState:
    queues: dict = field(default_factory=lambda: {"N": 0, "S": 0, "E": 0, "W": 0})
    phase: str = "NS_GREEN"
    phase_duration: int = 0
    step: int = 0
    total_wait: float = 0.0
    throughput: int = 0
    emergency_dir: Optional[str] = None
    emergency_steps: int = 0
    prev_phase: str = "NS_GREEN"
    episode_id: str = field(default_factory=lambda: str(uuid4()))


# ── Environment ───────────────────────────────────────────────────────────────

class TrafficSignalEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._sim = IntersectionState()
        self._task_id = "task1"
        self._cfg = TASK_CONFIGS["task1"]
        self._rng = random.Random()

    # ── Required OpenEnv API ──────────────────────────────────────────────────

    def reset(self, task_id: str = "task1") -> TrafficSignalObservation:
       self._task_id = task_id
       self._cfg = TASK_CONFIGS.get(task_id, TASK_CONFIGS["task1"])
       self._sim = IntersectionState()
       return self._observe(reward=0.0, done=False)

    def step(self, action: TrafficSignalAction) -> TrafficSignalObservation:
        s = self._sim
        cfg = self._cfg

        # Update task if agent changed it
        if action.task_id != self._task_id:
            self._task_id = action.task_id
            self._cfg = TASK_CONFIGS[action.task_id]
            cfg = self._cfg

        # 1. Apply phase decision
        s.prev_phase = s.phase
        if action.phase != s.phase:
            s.phase = action.phase
            s.phase_duration = 0
        else:
            s.phase_duration += 1

        # 2. Spawn vehicles (3 Bernoulli trials per arm per step)
        for arm, rate in cfg["arrival_rates"].items():
            arrivals = sum(1 for _ in range(3) if self._rng.random() < rate / 3)
            s.queues[arm] += arrivals

        # 3. Possibly spawn emergency vehicle
        if (
            cfg["emergency_prob"] > 0
            and s.emergency_dir is None
            and self._rng.random() < cfg["emergency_prob"]
        ):
            s.emergency_dir = self._rng.choice(["N", "S", "E", "W"])
            s.emergency_steps = self._rng.randint(5, 10)
            s.queues[s.emergency_dir] = max(s.queues[s.emergency_dir], 1)

        # 4. Move vehicles on green arms
        green_arms = ["N", "S"] if s.phase == "NS_GREEN" else ["E", "W"]
        vehicles_moved = 0
        for arm in green_arms:
            moved = min(s.queues[arm], cfg["service_rate"])
            s.queues[arm] -= moved
            vehicles_moved += moved
            s.throughput += moved

        # 5. Accumulate waiting time
        s.total_wait += sum(s.queues.values())
        s.step += 1

        # 6. Tick emergency countdown
        if s.emergency_dir is not None:
            if s.emergency_dir in green_arms:
                s.emergency_steps -= 1
                if s.emergency_steps <= 0:
                    s.emergency_dir = None
                    s.emergency_steps = 0

        # 7. Compute reward and return
        reward = self._reward(vehicles_moved, s, action.phase != s.prev_phase)
        done = s.step >= cfg["max_steps"]
        return self._observe(reward=reward, done=done)

    @property
    def state(self) -> State:
        return State(
            episode_id=self._sim.episode_id,
            step_count=self._sim.step,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _reward(self, vehicles_moved: int, s: IntersectionState, switched: bool) -> float:
        r = 0.0
        r += 0.1 * vehicles_moved
        r -= 0.02 * sum(s.queues.values())

        if s.emergency_dir is not None:
            green_arms = ["N", "S"] if s.phase == "NS_GREEN" else ["E", "W"]
            r += 0.5 if s.emergency_dir in green_arms else -0.4

        if switched:
            r -= 0.05

        for q in s.queues.values():
            if q > 15:
                r -= 0.15

        return round(r, 4)

    def _observe(self, reward: float, done: bool) -> TrafficSignalObservation:
        s = self._sim
        return TrafficSignalObservation(
            done=done,
            reward=reward,
            step=s.step,
            current_phase=s.phase,
            phase_duration=s.phase_duration,
            north_queue=s.queues["N"],
            south_queue=s.queues["S"],
            east_queue=s.queues["E"],
            west_queue=s.queues["W"],
            emergency_direction=s.emergency_dir,
            emergency_steps_remaining=s.emergency_steps if s.emergency_dir else None,
            total_wait_time=round(s.total_wait, 2),
            throughput=s.throughput,
            task_id=self._task_id,
            hint=self._hint(s),
        )

    def _hint(self, s: IntersectionState) -> str:
        parts = []
        if s.emergency_dir:
            parts.append(
                f"URGENT: Emergency vehicle from {s.emergency_dir}! "
                f"Clear within {s.emergency_steps} steps."
            )
        heaviest = max(s.queues, key=s.queues.get)
        if s.queues[heaviest] > 8:
            parts.append(f"Heavy backlog on {heaviest} ({s.queues[heaviest]} vehicles).")
        if s.phase_duration > 8:
            parts.append(f"Phase {s.phase} active {s.phase_duration} steps — consider switching.")
        return " ".join(parts) if parts else "Intersection operating normally."