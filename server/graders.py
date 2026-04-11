def _clamp(score: float) -> float:
    return max(0.01, min(0.99, round(float(score), 4)))


def grade_task1(record):
    total_wait = getattr(record, "total_wait_time", 0.0)
    steps = max(getattr(record, "steps", 1), 1)
    throughput = getattr(record, "throughput", 0)

    avg_wait = total_wait / steps
    wait_score = 1.0 / (1.0 + avg_wait / 18.0)
    throughput_score = min(0.99, throughput / max(steps * 2.5, 1))

    score = 0.75 * wait_score + 0.25 * throughput_score
    return _clamp(score)


def grade_task2(record):
    cleared = getattr(record, "emergency_cleared", 0)
    total = max(getattr(record, "total_emergencies", 1), 1)
    response_time = getattr(record, "avg_response_time", 10.0)
    throughput = getattr(record, "throughput", 0)
    steps = max(getattr(record, "steps", 1), 1)

    emergency_score = cleared / total
    response_score = 1.0 / (1.0 + response_time / 5.0)
    throughput_score = min(0.99, throughput / max(steps * 2.5, 1))

    score = 0.5 * emergency_score + 0.3 * response_score + 0.2 * throughput_score
    return _clamp(score)


def grade_task3(record):
    throughput = getattr(record, "throughput", 0)
    wait = getattr(record, "total_wait_time", 0.0)
    steps = max(getattr(record, "steps", 1), 1)
    fairness = getattr(record, "fairness_index", 0.5)
    cleared = getattr(record, "emergency_cleared", 0)
    total = max(getattr(record, "total_emergencies", 1), 1)
    response_time = getattr(record, "avg_response_time", 10.0)

    throughput_score = min(0.99, throughput / max(steps * 3.5, 1))
    wait_score = 1.0 / (1.0 + wait / steps / 14.0)
    fairness_score = max(0.01, min(0.99, fairness))
    emergency_score = 0.7 * (cleared / total) + 0.3 * (1.0 / (1.0 + response_time / 6.0))

    score = (
        0.30 * throughput_score
        + 0.25 * wait_score
        + 0.20 * fairness_score
        + 0.25 * emergency_score
    )
    return _clamp(score)
