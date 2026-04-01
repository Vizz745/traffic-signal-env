from typing import Literal, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class TrafficSignalAction(Action):
    """Set the active signal phase for the intersection."""
    phase: Literal["NS_GREEN", "EW_GREEN"] = Field(
        ..., description="NS_GREEN = North+South get green. EW_GREEN = East+West get green."
    )
    task_id: Literal["task1", "task2", "task3"] = Field(
        default="task1", description="Which task scenario to run"
    )


class TrafficSignalObservation(Observation):
    """Full intersection state returned at each step."""
    # done, reward, metadata inherited from Observation

    step: int = Field(default=0)
    current_phase: str = Field(default="NS_GREEN")
    phase_duration: int = Field(default=0, description="Steps current phase has been active")

    north_queue: int = Field(default=0)
    south_queue: int = Field(default=0)
    east_queue:  int = Field(default=0)
    west_queue:  int = Field(default=0)

    emergency_direction: Optional[str] = Field(
        default=None, description="Direction of active emergency vehicle, or null"
    )
    emergency_steps_remaining: Optional[int] = Field(default=None)

    total_wait_time: float = Field(default=0.0)
    throughput: int = Field(default=0)
    task_id: str = Field(default="task1")
    hint: str = Field(default="", description="Plain-English situation summary for the agent")