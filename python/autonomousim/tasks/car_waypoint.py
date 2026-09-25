"""CarWaypointOffroad-v0: drive a 4×4 through a chain of waypoints over generated off-road
terrain between the trees, with LiDAR and ``(v, κ)`` actions."""

from typing import Any

import numpy as np

from autonomousim.events import Event
from autonomousim.scenario import STATE
from autonomousim.tasks.base import Task

GOAL_REACHED = int(Event.GOAL_REACHED)
FINISHED = int(Event.FINISHED)
STUCK = int(Event.STUCK)

#: 3 rings (−10°, −3°, 3°) × 72 azimuths (5° apart, so a 0.4 m trunk falls between beams
#: only beyond about 4.5 m), 30 m, 10 Hz, on the roof: the lowest ring meets flat ground about
#: 9 m ahead (slopes, banks and water edges), the others see trunks and rocks.
LIDAR = {
    "pattern": {"type": "rings", "elevations": [-10.0, -3.0, 3.0], "azimuths": 72, "azimuth_fov": 360.0},
    "max_range": 30.0,
    "mount": {"position": [0.0, 0.0, 1.0]},
}


class CarWaypointOffroad(Task):
    """An off-road 4×4 (``offroad_4x4``) in ``vk`` mode (speed up to ``max_speed`` forward
    and 0.3 of it in reverse, path curvature up to the steering lock) spawns on a generated
    ``offroad`` wild map (a pool of ``map_count`` 512 m maps from ``map_seed``; evaluate on
    another ``map_seed`` for unseen maps) and drives through ``goals`` waypoints on drivable
    ground, each ``goal_distance`` from the previous one and reachable from it along routes
    with ``drivable_margin`` of room beside the vehicle (3 m: gaps between trunks about 8 m
    wide, room to steer round; with 1 m the scripted driver and learned policies crash into
    trunks in the tight gaps far more often). A waypoint
    counts once the vehicle's centre is within ``goal_radius``; the next one then becomes the
    goal.

    Observation: goal in the heading frame (scaled 1/20, clipped to ±3), speed, body velocity
    and rates, pitch and roll, steering angle, last action and a LiDAR scan (``LIDAR``: 3
    rings × 72 azimuths, 30 m, log ranges; ``lidar`` replaces its settings): 231 values.

    Reward per step (``d``: horizontal distance to the current goal, ``c``: distance from the
    vehicle's centre to the nearest solid obstacle, from the state row):

    - ``progress_weight·(d_before − d_after)``, measured to the current goal from the
      positions before and after the step (so switching goals does not jump);
    - ``goal_bonus`` per waypoint reached;
    - ``−proximity_weight·max(0, 1 − c/proximity_distance)²``;
    - ``−smooth_weight·‖Δa‖²``;
    - ``−time_weight`` per step;
    - ``−terminal_penalty`` on a crash, rollover, water, leaving the map or getting stuck
      (moving less than 0.5 m in ``stuck_time`` seconds).

    The episode succeeds when the last waypoint is reached and is truncated after
    ``episode_time`` (90 s: routes detour round trees, and a car in a forest needs
    three-point turns a drone does not). The policy runs at 20 Hz, the physics at 1 kHz.
    """

    name = "car_waypoint"
    default_episode_time = 90.0
    has_success = True
    failure_events = STUCK

    def __init__(
        self,
        *,
        goals: int = 3,
        goal_distance: tuple[float, float] = (25.0, 50.0),
        goal_radius: float = 3.0,
        max_speed: float = 8.0,
        stuck_time: float = 4.0,
        drivable_margin: float = 3.0,
        lidar: dict[str, Any] | None = None,
        progress_weight: float = 1.0,
        goal_bonus: float = 10.0,
        proximity_weight: float = 0.2,
        proximity_distance: float = 3.0,
        smooth_weight: float = 0.02,
        time_weight: float = 0.0,
        terminal_penalty: float = 50.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("vehicle", "offroad_4x4")
        kwargs.setdefault("action_mode", "vk")
        kwargs.setdefault("map", "offroad")
        kwargs.setdefault("policy_hz", 20)
        super().__init__(**kwargs)
        self.goals = goals
        self.goal_distance = goal_distance
        self.goal_radius = goal_radius
        self.max_speed = max_speed
        self.stuck_time = stuck_time
        self.drivable_margin = drivable_margin
        self.lidar = LIDAR if lidar is None else lidar
        self.progress_weight = progress_weight
        self.goal_bonus = goal_bonus
        self.proximity_weight = proximity_weight
        self.proximity_distance = proximity_distance
        self.smooth_weight = smooth_weight
        self.time_weight = time_weight
        self.terminal_penalty = terminal_penalty

    def group(self) -> dict[str, Any]:
        return {
            "drivable": {"margin": self.drivable_margin},
            "spawn": {"margin": 40.0},
            "goals": {
                "kind": "random",
                "count": self.goals,
                "distance": list(self.goal_distance),
                "margin": 40.0,
                "radius": self.goal_radius,
            },
            "ground_action_limits": {"speed": self.max_speed},
            "sensors": [{"name": "lidar", "type": "lidar", **self.lidar}],
            "obs": [
                {"term": "goal_rel_heading", "scale": 0.05, "clip": 3.0},
                {"term": "speed", "scale": 0.2},
                {"term": "lin_vel_body", "scale": 0.2},
                {"term": "ang_vel_body", "scale": 0.5},
                {"term": "pitch_roll"},
                {"term": "steering"},
                {"term": "last_action"},
                {"term": "lidar_log", "sensor": "lidar"},
            ],
        }

    def settings(self) -> dict[str, Any]:
        return {"events": {"bounds_margin": 5.0, "ground": {"stuck_time": self.stuck_time}}}

    # ------------------------------------------------------------------ episodes

    def bind(self, num_envs: int, policy_dt: float, act_dim: int) -> None:
        super().bind(num_envs, policy_dt, act_dim)
        self.prev_position = np.zeros((num_envs, 2))

    def reset(self, mask: np.ndarray | None = None, state: np.ndarray | None = None) -> None:
        super().reset(mask, state)
        if state is not None:
            m = slice(None) if mask is None else mask
            self.prev_position[m] = state[m, STATE["position"]][:, :2]

    def succeeded(self, state: np.ndarray, events: np.ndarray) -> np.ndarray:
        return (events & FINISHED) != 0

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        position = state[:, STATE["position"]][:, :2]
        goal = state[:, STATE["goal"]][:, :2]
        before = np.linalg.norm(goal - self.prev_position, axis=1)
        after = np.linalg.norm(goal - position, axis=1)
        self.prev_position[:] = position
        clearance = state[:, STATE["clearance"]][:, 0]
        near = np.clip(1.0 - clearance / self.proximity_distance, 0.0, 1.0)
        da = action - prev_action
        return (
            self.progress_weight * (before - after)
            + self.goal_bonus * ((events & GOAL_REACHED) != 0)
            - self.proximity_weight * near**2
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
            - self.time_weight
        )
