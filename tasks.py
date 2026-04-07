# tasks.py
from dataclasses import dataclass, field
from typing import List

BASELINE_WAIT = {"task1": 420.0, "task2": 600.0, "task3": 1800.0}


@dataclass
class EpisodeRecord:
    """Accumulates stats across a full episode for end-of-episode grading."""
    task_id: str
    total_wait: float = 0.0
    throughput: int = 0
    steps: int = 0
    max_queue_seen: dict = field(default_factory=lambda: {"N": 0, "S": 0, "E": 0, "W": 0})

    def update(self, obs):
        self.total_wait = obs.total_wait_time
        self.throughput = obs.throughput
        self.steps = obs.step
        for arm, q in [
            ("N", obs.north_queue), ("S", obs.south_queue),
            ("E", obs.east_queue),  ("W", obs.west_queue),
        ]:
            if q > self.max_queue_seen[arm]:
                self.max_queue_seen[arm] = q


def _clamp(value: float) -> float:
    """Strictly between 0 and 1 — validator rejects exactly 0.0 or 1.0."""
    return round(max(0.001, min(0.999, value)), 4)


def grade_task1(record: EpisodeRecord) -> float:
    """
    Easy: minimize total waiting time vs a naive fixed-timer baseline.
    Score = clamp(1 - total_wait / baseline, 0, 1)
    """
    baseline = BASELINE_WAIT["task1"]
    score = 1.0 - (record.total_wait / baseline)
    return _clamp(score)


def grade_task2(
    record: EpisodeRecord,
    response_steps: List[int],
    cleared: int,
    total: int,
) -> float:
    """
    Medium: 60% emergency clearance + 40% traffic throughput.
    """
    if total == 0:
        emg_score = 0.999
    else:
        cleared_ratio = cleared / total
        avg_resp = sum(response_steps) / len(response_steps) if response_steps else 10.0
        speed = max(0.001, 1.0 - (avg_resp - 3) / 7.0)
        emg_score = 0.7 * cleared_ratio + 0.3 * speed

    baseline_tp = 60
    tp_score = min(0.999, record.throughput / baseline_tp)

    return _clamp(0.6 * emg_score + 0.4 * tp_score)


def grade_task3(
    record: EpisodeRecord,
    response_steps: List[int],
    cleared: int,
    total: int,
) -> float:
    """
    Hard: 40% emergency + 30% throughput + 30% fairness.
    Fairness penalizes any arm with max queue > 3x average.
    """
    if total == 0:
        emg_score = 0.999
    else:
        cleared_ratio = cleared / total
        avg_resp = sum(response_steps) / len(response_steps) if response_steps else 10.0
        speed = max(0.001, 1.0 - (avg_resp - 3) / 7.0)
        emg_score = 0.7 * cleared_ratio + 0.3 * speed

    baseline_tp = 180
    tp_score = min(0.999, record.throughput / baseline_tp)

    max_q = max(record.max_queue_seen.values())
    avg_q = sum(record.max_queue_seen.values()) / 4 or 1
    fairness_score = max(0.001, 1.0 - (max_q / avg_q - 3) / 5.0)

    final = 0.4 * emg_score + 0.3 * tp_score + 0.3 * fairness_score
    return _clamp(final)