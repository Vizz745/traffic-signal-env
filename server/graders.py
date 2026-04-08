class Task1Grader:
    def grade(self, record):
        total_wait = getattr(record, "total_wait_time", 0)
        steps = max(getattr(record, "steps", 1), 1)
        avg_wait = total_wait / steps
        # avg_wait=0 → 1.0, so cap before clamp
        score = min(0.98, 1.0 / (1.0 + avg_wait / 10.0))
        return max(0.001, min(0.999, round(score, 4)))


class Task2Grader:
    def grade(self, record):
        cleared = getattr(record, "emergency_cleared", 0)
        total = max(getattr(record, "total_emergencies", 1), 1)
        response_time = getattr(record, "avg_response_time", 10)

        emergency_score = cleared / total
        response_score = 1.0 / (1.0 + response_time / 5.0)
        score = 0.6 * emergency_score + 0.4 * response_score
        return max(0.001, min(0.999, round(score, 4)))


class Task3Grader:
    def grade(self, record):
        throughput = getattr(record, "throughput", 0)
        wait = getattr(record, "total_wait_time", 0)
        steps = max(getattr(record, "steps", 1), 1)
        fairness = getattr(record, "fairness_index", 0.5)

        # Normalize throughput per step, cap at 1.0
        throughput_score = min(0.99, throughput / (steps * 5.0))
        wait_penalty = 1.0 / (1.0 + wait / steps / 10.0)
        score = 0.4 * throughput_score + 0.4 * wait_penalty + 0.2 * fairness
        return max(0.001, min(0.999, round(score, 4)))