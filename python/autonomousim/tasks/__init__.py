"""Tasks: scenarios with vectorised rewards and end conditions (see ``base.Task``)."""

from typing import Any

from autonomousim.tasks.base import Task, map_source
from autonomousim.tasks.car_waypoint import CarWaypointOffroad
from autonomousim.tasks.hover import QuadHover
from autonomousim.tasks.multi import MultiAgentTask, Team
from autonomousim.tasks.recover import QuadRecover
from autonomousim.tasks.road_follow import RoadFollowRural
from autonomousim.tasks.swarm import FormationHover, SwarmForestDrone, SwarmHover, SwarmWaypointForest
from autonomousim.tasks.trailer_reverse import TrailerReverse
from autonomousim.tasks.waypoint_forest import QuadWaypointForest

TASKS: dict[str, type[Task]] = {
    "hover": QuadHover,
    "recover": QuadRecover,
    "waypoint_forest": QuadWaypointForest,
    "car_waypoint": CarWaypointOffroad,
    "road_follow": RoadFollowRural,
    "trailer_reverse": TrailerReverse,
}


def make_task(task: str | Task, **kwargs: Any) -> Task:
    """A task instance from its name (``hover``, ``recover``, ``waypoint_forest``,
    ``car_waypoint``, ``road_follow``, ``trailer_reverse``) and keyword arguments, or the instance itself."""
    if isinstance(task, Task):
        if kwargs:
            raise TypeError("keyword arguments are only accepted with a task name")
        return task
    try:
        cls = TASKS[task]
    except KeyError:
        raise ValueError(f"unknown task {task!r}; available: {sorted(TASKS)}") from None
    return cls(**kwargs)


__all__ = [
    "TASKS",
    "CarWaypointOffroad",
    "FormationHover",
    "MultiAgentTask",
    "QuadHover",
    "QuadRecover",
    "QuadWaypointForest",
    "RoadFollowRural",
    "SwarmForestDrone",
    "SwarmHover",
    "SwarmWaypointForest",
    "Task",
    "Team",
    "TrailerReverse",
    "make_task",
    "map_source",
]
