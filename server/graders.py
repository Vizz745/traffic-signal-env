"""
server/graders.py — Traffic Signal Environment Graders

Each grader evaluates performance for a specific task.
Returns a normalized score strictly between (0, 1).
"""

# ---------------- TASK 1 ----------------
class Task1Grader:
    def __call__(self, record):
        """
        Goal: Minimize total waiting time (rush hour optimization)
        """
        total_wait = getattr(record, "total_wait_time", 0)
        steps = max(getattr(record, "steps", 1), 1)

        avg_wait = total_wait / steps

        # Lower wait = higher score
        score = 1.0 / (1.0 + avg_wait)

        return max(0.01, min(0.99, score))


# ---------------- TASK 2 ----------------
class Task2Grader:
    def __call__(self, record):
        """
        Goal: Handle emergency vehicles efficiently + traffic balance
        """
        cleared = getattr(record, "emergency_cleared", 0)
        total = max(getattr(record, "total_emergencies", 1), 1)

        response_time = getattr(record, "avg_response_time", 10)

        emergency_score = cleared / total
        response_score = 1.0 / (1.0 + response_time)

        score = 0.6 * emergency_score + 0.4 * response_score

        return max(0.01, min(0.99, score))


# ---------------- TASK 3 ----------------
class Task3Grader:
    def __call__(self, record):
        """
        Goal: Multi-objective optimization (throughput + fairness + emergency)
        """
        throughput = getattr(record, "throughput", 0)
        wait = getattr(record, "total_wait_time", 0)
        steps = max(getattr(record, "steps", 1), 1)

        fairness = getattr(record, "fairness_index", 0.5)

        throughput_score = throughput / (steps + 1)
        wait_penalty = 1.0 / (1.0 + wait / steps)

        score = 0.4 * throughput_score + 0.4 * wait_penalty + 0.2 * fairness

        return max(0.01, min(0.99, score))