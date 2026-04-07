import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasks import EpisodeRecord, grade_task1, grade_task2, grade_task3


class Task1Grader:
    def grade(self, record: EpisodeRecord, **kwargs) -> float:
        raw = grade_task1(record)
        return max(0.01, min(0.99, raw))


class Task2Grader:
    def grade(self, record: EpisodeRecord, response_steps=None, cleared=0, total_emg=0, **kwargs) -> float:
        raw = grade_task2(record, response_steps or [], cleared, total_emg)
        return max(0.01, min(0.99, raw))


class Task3Grader:
    def grade(self, record: EpisodeRecord, response_steps=None, cleared=0, total_emg=0, **kwargs) -> float:
        raw = grade_task3(record, response_steps or [], cleared, total_emg)
        return max(0.01, min(0.99, raw))