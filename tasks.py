# tasks.py
from dataclasses import dataclass, field
from typing import List

BASELINE_WAIT = {"task1": 420.0, "task2": 600.0, "task3": 1800.0}


@dataclass
class EpisodeRecord:
    """Accumulates stats across a full episode for end-of-episode grading."""
    task_id: str

    # ---- CORE METRICS ----
    total_wait_time: float = 0.0
    throughput: int = 0
    steps: int = 0

    # ---- QUEUE TRACKING ----
    max_queue_seen: dict = field(default_factory=lambda: {"N": 0, "S": 0, "E": 0, "W": 0})

    # ---- EMERGENCY TRACKING ----
    emergency_cleared: int = 0
    total_emergencies: int = 0
    response_times: List[int] = field(default_factory=list)

    # ---- FAIRNESS ----
    fairness_index: float = 0.5

    def update(self, obs):
        # ---- STEP ----
        self.steps += 1

        # ---- ACCUMULATE (FIXED) ----
        self.total_wait_time += getattr(obs, "total_wait_time", 0)
        self.throughput += getattr(obs, "throughput", 0)

        # ---- QUEUE ----
        for arm, q in [
            ("N", getattr(obs, "north_queue", 0)),
            ("S", getattr(obs, "south_queue", 0)),
            ("E", getattr(obs, "east_queue", 0)),
            ("W", getattr(obs, "west_queue", 0)),
        ]:
            if q > self.max_queue_seen[arm]:
                self.max_queue_seen[arm] = q

        # ---- EMERGENCY ----
        if getattr(obs, "emergency_direction", None):
            self.total_emergencies += 1

        if getattr(obs, "emergency_steps_remaining", None) == 0:
            self.emergency_cleared += 1
            self.response_times.append(1)  # simple approximation

    @property
    def avg_response_time(self):
        if not self.response_times:
            return 10.0
        return sum(self.response_times) / len(self.response_times)


# ---------------- HELPERS ----------------
def _clamp(value: float) -> float:
    """Strictly between 0 and 1 — validator rejects exactly 0.0 or 1.0."""
    return round(max(0.001, min(0.999, value)), 4)


# ---------------- TASK 1 ----------------
def grade_task1(record: EpisodeRecord) -> float:
    steps = max(record.steps, 1)
    avg_wait = record.total_wait_time / steps  # per-step average, not cumulative
    # avg_wait ~0-50 in practice; 1/(1+x) keeps it in (0,1)
    score = 1.0 / (1.0 + avg_wait / 15.0)
    return _clamp(score)


# ---------------- TASK 2 ----------------
def grade_task2(
    record: EpisodeRecord,
    response_steps: List[int],
    cleared: int,
    total: int,
) -> float:
    """
    Medium: emergency + throughput.
    """
    if total == 0:
        emg_score = 0.995
    else:
        cleared_ratio = cleared / total
        avg_resp = sum(response_steps) / len(response_steps) if response_steps else record.avg_response_time
        speed = max(0.001, 1.0 - (avg_resp - 3) / 7.0)
        emg_score = 0.7 * cleared_ratio + 0.3 * speed

    baseline_tp = 60
    tp_score = min(0.999, record.throughput / baseline_tp)

    return _clamp(0.6 * emg_score + 0.4 * tp_score)


# ---------------- TASK 3 ----------------
def grade_task3(
    record: EpisodeRecord,
    response_steps: List[int],
    cleared: int,
    total: int,
) -> float:
    """
    Hard: emergency + throughput + fairness.
    """
    if total == 0:
        emg_score = 0.995
    else:
        cleared_ratio = cleared / total
        avg_resp = sum(response_steps) / len(response_steps) if response_steps else record.avg_response_time
        speed = max(0.001, 1.0 - (avg_resp - 3) / 7.0)
        emg_score = 0.7 * cleared_ratio + 0.3 * speed

    baseline_tp = 180
    tp_score = min(0.999, record.throughput / baseline_tp)

    max_q = max(record.max_queue_seen.values())
    avg_q = sum(record.max_queue_seen.values()) / 4 or 1
    fairness_score = max(0.001, 1.0 - (max_q / avg_q - 3) / 5.0)

    final = 0.4 * emg_score + 0.3 * tp_score + 0.3 * fairness_score
    return _clamp(final)