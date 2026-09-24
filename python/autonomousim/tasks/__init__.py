"""Tasks: scenarios with vectorised rewards and end conditions (see ``base.Task``)."""

from typing import Any

from autonomousim.tasks.base import Task, map_source
from autonomousim.tasks.hover import QuadHover
from autonomousim.tasks.recover import QuadRecover
from autonomousim.tasks.waypoint_forest import QuadWaypointForest

TASKS: dict[str, type[Task]] = {"hover": QuadHover, "recover": QuadRecover, "waypoint_forest": QuadWaypointForest}


def make_task(task: str | Task, **kwargs: Any) -> Task:
    """A task instance from its name (``hover``, ``recover``, ``waypoint_forest``) and keyword
    arguments, or the instance itself."""
    if isinstance(task, Task):
        if kwargs:
            raise TypeError("keyword arguments are only accepted with a task name")
        return task
    try:
        cls = TASKS[task]
    except KeyError:
        raise ValueError(f"unknown task {task!r}; available: {sorted(TASKS)}") from None
    return cls(**kwargs)


__all__ = ["TASKS", "QuadHover", "QuadRecover", "QuadWaypointForest", "Task", "make_task", "map_source"]
