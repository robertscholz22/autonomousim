"""autonomousim: a 3D simulator for training autonomous aerial and ground vehicles.

Importing the package registers the Gymnasium environments:

- ``autonomousim/QuadHover-v0``: fly to a goal up to 2 m away and hold it;
- ``autonomousim/QuadRecover-v0``: recover from any attitude and hold position;
- ``autonomousim/QuadWaypointForest-v0``: fly through waypoints in generated forests with
  LiDAR;
- ``autonomousim/CarWaypointOffroad-v0``: drive a 4×4 through waypoints over generated
  off-road terrain with LiDAR.

``gym.make_vec(id, num_envs=N, ...)`` gives the native vector environment
(``autonomousim.vector_env``); ``gym.make(id)`` a single environment. Keyword arguments
configure the task (``autonomousim.tasks``): vehicle, action mode, map, rates, episode time,
wind, randomisation, observations and scenario overrides.
"""

import gymnasium as _gym

from autonomousim._native import STATE_DIM, STATE_FIELDS, BatchSim, native_version, vehicle_presets
from autonomousim.events import TERMINAL, Event
from autonomousim.scenario import STATE, default_scenario, load_scenario

__version__ = native_version()

ENVS = {
    "QuadHover-v0": "hover",
    "QuadRecover-v0": "recover",
    "QuadWaypointForest-v0": "waypoint_forest",
    "CarWaypointOffroad-v0": "car_waypoint",
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
    "STATE",
    "STATE_DIM",
    "STATE_FIELDS",
    "TERMINAL",
    "BatchSim",
    "Event",
    "__version__",
    "default_scenario",
    "load_scenario",
    "vehicle_presets",
]
