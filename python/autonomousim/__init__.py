"""autonomousim: a 3D simulator for training autonomous aerial and ground vehicles.

Importing the package registers the Gymnasium environments:

- ``autonomousim/QuadHover-v0``: fly to a goal up to 2 m away and hold it;
- ``autonomousim/QuadHoverPad-v0``: find a landing pad with a downward camera and hover over
  it (``Dict`` observations with an image);
- ``autonomousim/QuadRecover-v0``: recover from any attitude and hold position;
- ``autonomousim/QuadWaypointForest-v0``: fly through waypoints in generated forests with
  LiDAR;
- ``autonomousim/CarWaypointOffroad-v0``: drive a 4×4 through waypoints over generated
  off-road terrain with LiDAR;
- ``autonomousim/RoadFollowRural-v0``: drive a car along a route over the roads of generated
  farmland to a farm yard;
- ``autonomousim/TrailerReverse-v0``: back a tractor's semitrailer into a bay at the far side
  of a farm yard;
- ``autonomousim/TrackedCrossCountry-v0``: drive a tracked APC across rural fields, soft soil
  and ditches through waypoints along a planned path;
- ``autonomousim/MotorcycleRoadRural-v0``: ride a motorcycle along a route over the roads of
  generated farmland to a farm yard, leaning into the bends;
- ``autonomousim/FixedWingWaypoints-v0``: fly a small fixed-wing UAV through waypoints
  kilometres apart over a large mountainous map in wind and turbulence;
- ``autonomousim/HeliLandingZone-v0``: fly a model helicopter from forward flight to a flat,
  open landing zone a few hundred metres away and set it down there;
- ``autonomousim/TiltrotorDelivery-v0``: take a quad tiltrotor off from a farm yard, cruise
  kilometres on the wing across farmland and land on another farm's yard;
- ``autonomousim/DroneLandOnCar-v0``: follow a car driving on rural roads with a downward
  camera and land on its roof (``Dict`` observations with an image);
- ``autonomousim/DroneRooftopDelivery-v0``: fly a quadrotor across a generated city from a
  sidewalk or a roof to a landing pad on another roof, with LiDAR;
- ``autonomousim/CarParking-v0``: park a car in a free bay of a generated city, reversing
  into a parking lot's bay or parallel parking between cars on the street;
- ``autonomousim/CarUrbanDrive-v0``: drive a car along a lane-level route of 0.5–1.5 km
  through a generated city in traffic, with buses, cyclists, pedestrians and traffic lights.

``gym.make_vec(id, num_envs=N, ...)`` gives the native vector environment
(``autonomousim.vector_env``); ``gym.make(id)`` a single environment. Worlds with several
agents use ``MultiAgentVectorEnv`` (``autonomousim.multiagent``) with a ``MultiAgentTask``. Keyword arguments
configure the task (``autonomousim.tasks``): vehicle, action mode, map, rates, episode time,
wind, randomisation, observations and scenario overrides.
"""

import gymnasium as _gym

from autonomousim._native import (
    SEMANTIC_CLASSES,
    STATE_DIM,
    STATE_FIELDS,
    BatchSim,
    native_version,
    render_adapter,
    set_render_adapter,
    trailer_presets,
    vehicle_presets,
)
from autonomousim.events import TERMINAL, Event
from autonomousim.multiagent import MultiAgentVectorEnv
from autonomousim.scenario import STATE, default_scenario, load_scenario

__version__ = native_version()

ENVS = {
    "QuadHover-v0": "hover",
    "QuadHoverPad-v0": "hover_pad",
    "QuadRecover-v0": "recover",
    "QuadWaypointForest-v0": "waypoint_forest",
    "CarWaypointOffroad-v0": "car_waypoint",
    "RoadFollowRural-v0": "road_follow",
    "TrailerReverse-v0": "trailer_reverse",
    "TrackedCrossCountry-v0": "tracked_cross_country",
    "MotorcycleRoadRural-v0": "motorcycle_road",
    "FixedWingWaypoints-v0": "fixed_wing_waypoints",
    "HeliLandingZone-v0": "heli_landing_zone",
    "TiltrotorDelivery-v0": "tiltrotor_delivery",
    "DroneLandOnCar-v0": "land_on_car",
    "DroneRooftopDelivery-v0": "rooftop_delivery",
    "CarParking-v0": "car_parking",
    "CarUrbanDrive-v0": "car_urban_drive",
}

for _name, _task in ENVS.items():
    if f"autonomousim/{_name}" not in _gym.registry:
        _gym.register(
            id=f"autonomousim/{_name}",
            entry_point="autonomousim.env:AutonomousimEnv",
            vector_entry_point="autonomousim.vector_env:AutonomousimVectorEnv",
            kwargs={"task": _task},
        )

__all__ = [
    "ENVS",
    "SEMANTIC_CLASSES",
    "STATE",
    "STATE_DIM",
    "STATE_FIELDS",
    "TERMINAL",
    "BatchSim",
    "Event",
    "MultiAgentVectorEnv",
    "__version__",
    "default_scenario",
    "load_scenario",
    "render_adapter",
    "set_render_adapter",
    "trailer_presets",
    "vehicle_presets",
]
