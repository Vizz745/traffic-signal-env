# client.py
from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TrafficSignalAction, TrafficSignalObservation


class TrafficSignalEnv(EnvClient[TrafficSignalAction, TrafficSignalObservation, State]):

    def _step_payload(self, action: TrafficSignalAction) -> Dict:
        return {
            "phase": action.phase,
            "task_id": action.task_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TrafficSignalObservation]:
        obs_data = payload.get("observation", payload)
        obs = TrafficSignalObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
            step=obs_data.get("step", 0),
            current_phase=obs_data.get("current_phase", "NS_GREEN"),
            phase_duration=obs_data.get("phase_duration", 0),
            north_queue=obs_data.get("north_queue", 0),
            south_queue=obs_data.get("south_queue", 0),
            east_queue=obs_data.get("east_queue", 0),
            west_queue=obs_data.get("west_queue", 0),
            emergency_direction=obs_data.get("emergency_direction"),
            emergency_steps_remaining=obs_data.get("emergency_steps_remaining"),
            total_wait_time=obs_data.get("total_wait_time", 0.0),
            throughput=obs_data.get("throughput", 0),
            task_id=obs_data.get("task_id", "task1"),
            hint=obs_data.get("hint", ""),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )