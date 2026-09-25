"""RoadFollowRural-v0: drive a car along a route over the roads of generated farmland to a
farm yard, keeping to the right-hand lane, with ``(v, κ)`` actions."""

from typing import Any

import numpy as np

from autonomousim.events import Event
from autonomousim.scenario import STATE
from autonomousim.tasks.base import Task

GOAL_REACHED = int(Event.GOAL_REACHED)
FINISHED = int(Event.FINISHED)
STUCK = int(Event.STUCK)

#: 2 rings (−3°, 3°) × 36 azimuths (10° apart), 40 m, 10 Hz, on the roof: hedges, fences,
#: buildings and trees beside the road, and other traffic later.
LIDAR = {
    "pattern": {"type": "rings", "elevations": [-3.0, 3.0], "azimuths": 36, "azimuth_fov": 360.0},
    "max_range": 40.0,
    "mount": {"position": [0.0, 0.0, 1.4]},
}


class RoadFollowRural(Task):
    """A car (``sedan_like``) in ``vk`` mode (speed up to ``max_speed`` forward and 0.3 of it
    in reverse, path curvature up to the steering lock) starts in a lane of a random road on a
    generated ``rural`` map (a pool of ``map_count`` 512 m maps from ``map_seed``; evaluate on
    another ``map_seed`` for unseen maps) and follows its route along the roads to a farm
    yard ``route_length`` away (paved roads keep right; gravel roads and tracks are driven on
    their centre line). Goals lie every ``goal_step`` m along the route; one counts once the
    car's centre is within ``goal_radius``.

    Observation (97 values): the ``road`` term (offset from the lane, heading error, lane
    curvature 5–40 m ahead), the ``route`` term (lane points 5–40 m ahead in the heading
    frame, scaled 1/20), ``on_road``, speed, body velocity and rates, steering angle, last
    action and a LiDAR scan (``LIDAR``: 2 rings × 36 azimuths, 40 m, log ranges; ``lidar``
    replaces its settings).

    Reward per step (``d``: horizontal distance to the current goal; ``e``, ``ψ`` and ``o``:
    lane offset, heading error and distance off the road surface, from the state row):

    - ``progress_weight·(d_before − d_after)``, measured to the current goal from the
      positions before and after the step (so switching goals does not jump);
    - ``goal_bonus`` per goal reached and ``finish_bonus`` at the yard;
    - ``−lane_weight·|e| − heading_weight·|ψ|``;
    - ``−off_road_weight·o``;
    - ``−smooth_weight·‖Δa‖²``;
    - ``−terminal_penalty`` on a crash, rollover, water, leaving the map, getting stuck
      (moving less than 0.5 m in ``stuck_time`` seconds) or ending up more than
      ``max_off_road`` m off the road.

    The episode succeeds at the yard and is truncated after ``episode_time`` (60 s: 400 m
    at an average of 7 m/s). The policy runs at 20 Hz, the physics at 1 kHz.
    """

    name = "road_follow"
    default_episode_time = 60.0
    has_success = True
    failure_events = STUCK

    def __init__(
        self,
        *,
        route_length: tuple[float, float] = (150.0, 400.0),
        goal_step: float = 20.0,
        goal_radius: float = 5.0,
        max_speed: float = 15.0,
        stuck_time: float = 5.0,
        max_off_road: float = 3.0,
        lidar: dict[str, Any] | None = None,
        progress_weight: float = 1.0,
        goal_bonus: float = 1.0,
        finish_bonus: float = 20.0,
        lane_weight: float = 0.1,
        heading_weight: float = 0.1,
        off_road_weight: float = 0.5,
        smooth_weight: float = 0.02,
        terminal_penalty: float = 20.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("vehicle", "sedan_like")
        kwargs.setdefault("action_mode", "vk")
        kwargs.setdefault("map", "rural")
        kwargs.setdefault("policy_hz", 20)
        super().__init__(**kwargs)
        self.route_length = route_length
        self.goal_step = goal_step
        self.goal_radius = goal_radius
        self.max_speed = max_speed
        self.stuck_time = stuck_time
        self.max_off_road = max_off_road
        self.lidar = LIDAR if lidar is None else lidar
        self.progress_weight = progress_weight
        self.goal_bonus = goal_bonus
        self.finish_bonus = finish_bonus
        self.lane_weight = lane_weight
        self.heading_weight = heading_weight
        self.off_road_weight = off_road_weight
        self.smooth_weight = smooth_weight
        self.terminal_penalty = terminal_penalty

    def group(self) -> dict[str, Any]:
        return {
            "spawn": {"on_road": True},
            "goals": {
                "kind": "route",
                "distance": list(self.route_length),
                "radius": self.goal_radius,
                "route": {"destination": "yard", "step": self.goal_step},
            },
            "ground_action_limits": {"speed": self.max_speed},
            "sensors": [{"name": "lidar", "type": "lidar", **self.lidar}],
            "obs": [
                {"term": "road"},
                {"term": "route", "scale": 0.05},
                {"term": "on_road"},
                {"term": "speed", "scale": 0.1},
                {"term": "lin_vel_body", "scale": 0.1},
                {"term": "ang_vel_body", "scale": 0.5},
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

    def failed(self, state: np.ndarray) -> np.ndarray:
        return state[:, STATE["road"]][:, 2] > self.max_off_road

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
        road = state[:, STATE["road"]]
        da = action - prev_action
        return (
            self.progress_weight * (before - after)
            + self.goal_bonus * ((events & GOAL_REACHED) != 0)
            + self.finish_bonus * ((events & FINISHED) != 0)
            - self.lane_weight * np.abs(road[:, 0])
            - self.heading_weight * np.abs(road[:, 1])
            - self.off_road_weight * road[:, 2]
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
        )
