"""Tasks: scenarios with vectorised rewards and end conditions (see ``base.Task``)."""

from typing import Any

from autonomousim.tasks.base import Task, map_source
from autonomousim.tasks.car_parking import CarParking, ParkingPilot
from autonomousim.tasks.car_waypoint import CarWaypointOffroad
from autonomousim.tasks.fixed_wing_waypoints import FixedWingWaypoints
from autonomousim.tasks.heli_landing_zone import HeliLandingZone
from autonomousim.tasks.hover import QuadHover
from autonomousim.tasks.hover_pad import QuadHoverPad
from autonomousim.tasks.land_on_car import DroneLandOnCar
from autonomousim.tasks.motorcycle_road import MotorcycleRoadRural
from autonomousim.tasks.multi import MultiAgentTask, Team
from autonomousim.tasks.recover import QuadRecover
from autonomousim.tasks.road_follow import RoadFollowRural
from autonomousim.tasks.rooftop_delivery import DroneRooftopDelivery
from autonomousim.tasks.swarm import FormationHover, SwarmForestDrone, SwarmHover, SwarmWaypointForest
from autonomousim.tasks.tiltrotor_delivery import TiltrotorDelivery
from autonomousim.tasks.tracked_cross_country import TrackedCrossCountry
from autonomousim.tasks.trailer_reverse import TrailerReverse
from autonomousim.tasks.waypoint_forest import QuadWaypointForest

TASKS: dict[str, type[Task]] = {
    "hover": QuadHover,
    "hover_pad": QuadHoverPad,
    "recover": QuadRecover,
    "waypoint_forest": QuadWaypointForest,
    "car_waypoint": CarWaypointOffroad,
    "road_follow": RoadFollowRural,
    "trailer_reverse": TrailerReverse,
    "tracked_cross_country": TrackedCrossCountry,
    "motorcycle_road": MotorcycleRoadRural,
    "fixed_wing_waypoints": FixedWingWaypoints,
    "heli_landing_zone": HeliLandingZone,
    "tiltrotor_delivery": TiltrotorDelivery,
    "land_on_car": DroneLandOnCar,
    "rooftop_delivery": DroneRooftopDelivery,
    "car_parking": CarParking,
}


def make_task(task: str | Task, **kwargs: Any) -> Task:
    """A task instance from its name (``hover``, ``hover_pad``, ``recover``, ``waypoint_forest``,
    ``car_waypoint``, ``road_follow``, ``trailer_reverse``, ``tracked_cross_country``,
    ``motorcycle_road``, ``fixed_wing_waypoints``, ``heli_landing_zone``, ``tiltrotor_delivery``, ``land_on_car``, ``rooftop_delivery``,
    ``car_parking``) and keyword
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


__all__ = [
    "TASKS",
    "CarParking",
    "CarWaypointOffroad",
    "DroneLandOnCar",
    "DroneRooftopDelivery",
    "FixedWingWaypoints",
    "FormationHover",
    "HeliLandingZone",
    "MotorcycleRoadRural",
    "MultiAgentTask",
    "ParkingPilot",
    "QuadHover",
    "QuadHoverPad",
    "QuadRecover",
    "QuadWaypointForest",
    "RoadFollowRural",
    "SwarmForestDrone",
    "SwarmHover",
    "SwarmWaypointForest",
    "Task",
    "Team",
    "TiltrotorDelivery",
    "TrackedCrossCountry",
    "TrailerReverse",
    "make_task",
    "map_source",
]
