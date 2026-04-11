from dataclasses import dataclass, field
from typing import List


@dataclass
class EpisodeRecord:
    """Accumulates stats across a full episode for end-of-episode grading."""
    task_id: str

    total_wait_time: float = 0.0
    throughput: int = 0
    steps: int = 0

    max_queue_seen: dict = field(
        default_factory=lambda: {"N": 0, "S": 0, "E": 0, "W": 0}
    )

    emergency_cleared: int = 0
    total_emergencies: int = 0
    response_times: List[int] = field(default_factory=list)

    fairness_index: float = 0.5

    def update(self, obs):
        self.steps += 1

        self.total_wait_time += getattr(obs, "total_wait_time", 0.0)
        self.throughput = max(self.throughput, getattr(obs, "throughput", 0))

        for arm, q in [
            ("N", getattr(obs, "north_queue", 0)),
            ("S", getattr(obs, "south_queue", 0)),
            ("E", getattr(obs, "east_queue", 0)),
            ("W", getattr(obs, "west_queue", 0)),
        ]:
            if q > self.max_queue_seen[arm]:
                self.max_queue_seen[arm] = q

        if getattr(obs, "emergency_direction", None):
            self.total_emergencies += 1

        if getattr(obs, "emergency_steps_remaining", None) == 0:
            self.emergency_cleared += 1
            self.response_times.append(1)

    @property
    def avg_response_time(self):
        if not self.response_times:
            return 10.0
        return sum(self.response_times) / len(self.response_times)


def _clamp(value: float) -> float:
    return round(max(0.001, min(0.999, float(value))), 4)


def grade_task1(record: EpisodeRecord) -> float:
    steps = max(record.steps, 1)
    avg_wait = record.total_wait_time / steps
    throughput_rate = record.throughput / steps

    wait_score = 1.0 / (1.0 + avg_wait / 18.0)
    tp_score = min(0.999, throughput_rate / 3.0)

    score = 0.75 * wait_score + 0.25 * tp_score
    return _clamp(score)


def grade_task2(
    record: EpisodeRecord,
    response_steps: List[int],
    cleared: int,
    total: int,
) -> float:
    steps = max(record.steps, 1)
    avg_wait = record.total_wait_time / steps
    throughput_rate = record.throughput / steps

    if total == 0:
        emg_score = 0.5
    else:
        cleared_ratio = cleared / total
        avg_resp = (
            sum(response_steps) / len(response_steps)
            if response_steps
            else record.avg_response_time
        )
        speed = max(0.001, min(0.999, 1.0 / (1.0 + avg_resp / 4.0)))
        emg_score = 0.7 * cleared_ratio + 0.3 * speed

    wait_score = 1.0 / (1.0 + avg_wait / 18.0)
    tp_score = min(0.999, throughput_rate / 3.0)

    score = 0.45 * emg_score + 0.30 * wait_score + 0.25 * tp_score
    return _clamp(score)


def grade_task3(
    record: EpisodeRecord,
    response_steps: List[int],
    cleared: int,
    total: int,
) -> float:
    steps = max(record.steps, 1)
    avg_wait = record.total_wait_time / steps
    throughput_rate = record.throughput / steps

    if total == 0:
        emg_score = 0.35
    else:
        cleared_ratio = cleared / total
        avg_resp = (
            sum(response_steps) / len(response_steps)
            if response_steps
            else record.avg_response_time
        )
        speed = max(0.001, min(0.999, 1.0 / (1.0 + avg_resp / 4.5)))
        emg_score = 0.7 * cleared_ratio + 0.3 * speed

    tp_score = min(0.999, throughput_rate / 3.5)

    max_q = max(record.max_queue_seen.values()) if record.max_queue_seen else 1
    avg_q = (sum(record.max_queue_seen.values()) / 4) or 1
    imbalance = max_q / avg_q if avg_q > 0 else 3.0
    fairness_score = max(
        0.001, min(0.999, 1.0 / (1.0 + max(0.0, imbalance - 1.0)))
    )

    wait_score = 1.0 / (1.0 + avg_wait / 22.0)

    score = 0.30 * emg_score + 0.25 * tp_score + 0.20 * fairness_score + 0.25 * wait_score
    return _clamp(score)
