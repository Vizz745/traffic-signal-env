class Task1Grader:
    def grade(self, record=None, **kwargs) -> float:
        try:
            baseline = 420.0
            total_wait = getattr(record, 'total_wait', baseline)
            raw = 1.0 - (total_wait / baseline)
            return max(0.01, min(0.99, raw))
        except Exception:
            return 0.5

class Task2Grader:
    def grade(self, record=None, **kwargs) -> float:
        try:
            response_steps = kwargs.get('response_steps') or []
            cleared = kwargs.get('cleared', 0)
            total = kwargs.get('total_emg', 0)
            if total == 0:
                emg_score = 0.8
            else:
                cleared_ratio = cleared / total
                avg_resp = sum(response_steps) / len(response_steps) if response_steps else 10.0
                speed = max(0.01, 1.0 - (avg_resp - 3) / 7.0)
                emg_score = 0.7 * cleared_ratio + 0.3 * speed
            throughput = getattr(record, 'throughput', 0)
            tp_score = min(0.99, throughput / 60)
            raw = 0.6 * emg_score + 0.4 * tp_score
            return max(0.01, min(0.99, raw))
        except Exception:
            return 0.5

class Task3Grader:
    def grade(self, record=None, **kwargs) -> float:
        try:
            response_steps = kwargs.get('response_steps') or []
            cleared = kwargs.get('cleared', 0)
            total = kwargs.get('total_emg', 0)
            if total == 0:
                emg_score = 0.8
            else:
                cleared_ratio = cleared / total
                avg_resp = sum(response_steps) / len(response_steps) if response_steps else 10.0
                speed = max(0.01, 1.0 - (avg_resp - 3) / 7.0)
                emg_score = 0.7 * cleared_ratio + 0.3 * speed
            throughput = getattr(record, 'throughput', 0)
            tp_score = min(0.99, throughput / 180)
            max_queue_seen = getattr(record, 'max_queue_seen', {"N":0,"S":0,"E":0,"W":0})
            max_q = max(max_queue_seen.values())
            avg_q = sum(max_queue_seen.values()) / 4 or 1
            fairness_score = max(0.01, 1.0 - (max_q / avg_q - 3) / 5.0)
            raw = 0.4 * emg_score + 0.3 * tp_score + 0.3 * fairness_score
            return max(0.01, min(0.99, raw))
        except Exception:
            return 0.5