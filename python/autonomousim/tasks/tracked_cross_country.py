"""TrackedCrossCountry-v0: drive a tracked APC across a rural map's fields, soft soil and
ditches through a chain of waypoints off the roads, along a planned path, with ``(v, ω)``
actions; and a scripted path-following driver that proves the task solvable."""

from typing import Any

import numpy as np

from autonomousim.events import Event
from autonomousim.scenario import STATE
from autonomousim.tasks.base import Task

GOAL_REACHED = int(Event.GOAL_REACHED)
FINISHED = int(Event.FINISHED)
STUCK = int(Event.STUCK)

#: 2 rings (−10°, 0°) × 36 azimuths (10° apart), 30 m, 10 Hz, on the roof: the lower ring meets
#: flat ground about 12 m ahead (ditch banks, slopes and water edges), the upper one sees
#: hedges, trees and buildings.
LIDAR = {
    "pattern": {"type": "rings", "elevations": [-10.0, 0.0], "azimuths": 36, "azimuth_fov": 360.0},
    "max_range": 30.0,
    "mount": {"position": [0.0, 0.0, 1.2]},
}

#: Scale of the ``route`` term (points 5, 10, 20 and 40 m ahead along the path).
ROUTE_SCALE = 0.05


class TrackedCrossCountry(Task):
    """The tracked APC (``tracked_apc``: torque converter, controlled-differential steering)
    in ``vw`` mode (speed up to ``max_speed`` forward and ``max_speed/2`` in reverse, yaw rate
    up to ``max_yaw_rate``) spawns at rest on a generated ``rural`` map (a pool of
    ``map_count`` 512 m farmland maps from ``map_seed``) on ground at most 10° steep and drives
    through ``goals`` waypoints off the roads, each ``goal_distance`` from the previous one and
    reachable from it. A waypoint counts once the vehicle's centre is within ``goal_radius``.

    Each episode plans the cheapest drivable path from the spawn through the waypoints (the
    drive grid's A* with ``resistance_cost`` weighting the APC's motion resistance on the
    ground: plowed fields and mud cost more than meadow and tracks; slopes over 30° and water are
    avoided, and solid obstacles (buildings, fences, hedge cores, trunks) kept 2 m clear of the
    hull's sides; ditches are crossed). The vehicle starts facing along the path, within 30°. The path to the current waypoint is the agent's
    route: the ``route`` observation term gives its points 5, 10, 20 and 40 m ahead, and the
    ``road`` state column the offset from it and its heading.

    Observation (95 values): route points ahead (heading frame, scaled 1/20, clipped to ±3),
    goal in the heading frame (scaled 1/50, clipped to ±3), speed, body velocity and rates,
    pitch and roll, the tracks' mean sinkage (×10, m), last action and a LiDAR scan (``LIDAR``,
    log ranges; ``lidar`` replaces its settings).

    Reward per step (``Δt``: the policy step; ``t̂``: the path's direction at the vehicle;
    ``e``: the lateral offset from the path):

    - ``progress_weight·(v·t̂)·Δt``: metres made good along the path;
    - ``goal_bonus`` per waypoint reached;
    - ``−offset_weight·min(1, (e/offset_scale)²)``;
    - ``−smooth_weight·‖Δa‖²``;
    - ``−terminal_penalty`` on a crash (the hull meeting the ground or a solid obstacle, or the
      running gear hitting it faster than 5 m/s), a rollover, water, leaving the map or getting
      stuck (moving less than 0.5 m in ``stuck_time`` seconds: in a hedge, or climbing out of a
      muddy ditch too slowly).

    The episode succeeds when the last waypoint is reached and is truncated after
    ``episode_time`` (120 s). The policy runs at 20 Hz, the physics at 1 kHz.
    ``scripted(obs)`` gives the actions of a path-following driver for comparison.
    """

    name = "tracked_cross_country"
    default_episode_time = 120.0
    has_success = True
    failure_events = STUCK

    def __init__(
        self,
        *,
        goals: int = 3,
        goal_distance: tuple[float, float] = (40.0, 100.0),
        goal_radius: float = 4.0,
        max_speed: float = 6.0,
        max_yaw_rate: float = 1.0,
        stuck_time: float = 6.0,
        resistance_cost: float = 30.0,
        lidar: dict[str, Any] | None = None,
        progress_weight: float = 1.0,
        goal_bonus: float = 10.0,
        offset_weight: float = 0.05,
        offset_scale: float = 5.0,
        smooth_weight: float = 0.02,
        terminal_penalty: float = 50.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("vehicle", "tracked_apc")
        kwargs.setdefault("action_mode", "vw")
        kwargs.setdefault("map", "rural")
        kwargs.setdefault("policy_hz", 20)
        super().__init__(**kwargs)
        self.goals = goals
        self.goal_distance = goal_distance
        self.goal_radius = goal_radius
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.stuck_time = stuck_time
        self.resistance_cost = resistance_cost
        self.lidar = LIDAR if lidar is None else lidar
        self.progress_weight = progress_weight
        self.goal_bonus = goal_bonus
        self.offset_weight = offset_weight
        self.offset_scale = offset_scale
        self.smooth_weight = smooth_weight
        self.terminal_penalty = terminal_penalty

    def group(self) -> dict[str, Any]:
        return {
            "drivable": {
                "max_slope_deg": 30.0,
                "spawn_slope_deg": 10.0,
                "margin": 2.0,
                "resistance_cost": self.resistance_cost,
            },
            "spawn": {"margin": 40.0, "yaw_deg": [-30.0, 30.0]},
            "goals": {
                "kind": "random",
                "count": self.goals,
                "distance": list(self.goal_distance),
                "margin": 40.0,
                "radius": self.goal_radius,
                "off_road": True,
                "path": True,
            },
            "ground_action_limits": {
                "speed": self.max_speed,
                "reverse": 0.5 * self.max_speed,
                "yaw_rate": self.max_yaw_rate,
            },
            "sensors": [{"name": "lidar", "type": "lidar", **self.lidar}],
            "obs": [
                {"term": "route", "scale": ROUTE_SCALE, "clip": 3.0},
                {"term": "goal_rel_heading", "scale": 0.02, "clip": 3.0},
                {"term": "speed", "scale": 0.2},
                {"term": "lin_vel_body", "scale": 0.2},
                {"term": "ang_vel_body", "scale": 0.5},
                {"term": "pitch_roll"},
                {"term": "sinkage", "scale": 10.0},
                {"term": "last_action"},
                {"term": "lidar_log", "sensor": "lidar"},
            ],
        }

    def settings(self) -> dict[str, Any]:
        # Sprockets and idlers meet ditch banks at up to 4 m/s: only harder hits are crashes.
        return {
            "events": {"crash_speed": 5.0, "bounds_margin": 5.0, "ground": {"stuck_time": self.stuck_time}}
        }

    # ------------------------------------------------------------------ episodes

    def succeeded(self, state: np.ndarray, events: np.ndarray) -> np.ndarray:
        return (events & FINISHED) != 0

    @staticmethod
    def path_errors(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Speed along the planned path (m/s) and the lateral offset from it (m, + left)."""
        road = state[:, STATE["road"]]
        q = state[:, STATE["orientation"]]
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        tangent = heading + road[:, 1]
        v = state[:, STATE["velocity"]]
        return v[:, 0] * np.cos(tangent) + v[:, 1] * np.sin(tangent), road[:, 0]

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        along, offset = self.path_errors(state)
        da = action - prev_action
        return (
            self.progress_weight * along * self.policy_dt
            + self.goal_bonus * ((events & GOAL_REACHED) != 0)
            - self.offset_weight * np.minimum(1.0, (offset / self.offset_scale) ** 2)
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
        )

    # ------------------------------------------------------------------ scripted driver

    def scripted(
        self,
        obs: np.ndarray,
        speed: float = 4.5,
        heading_gain: float = 1.2,
        slow_down: float = 5.0,
        min_speed: float = 3.0,
    ) -> np.ndarray:
        """Normalised ``vw`` actions of a path-following driver for the observations ``obs``
        (the task's default terms: ``route`` first).

        It aims at the path's point 10 m ahead (a pure-pursuit look-ahead): yaw rate
        ``heading_gain`` times the bearing of that point; speed ``speed``, less
        ``slow_down`` m/s per radian of the larger of that bearing and the bearing of the point
        20 m ahead (bends and turns onto the path), at least ``min_speed`` (slower, it stalls
        climbing out of ditches). It finishes about three in four episodes.
        """
        route = obs[:, 0:8].reshape(-1, 4, 2) / ROUTE_SCALE
        bearing = np.arctan2(route[:, :, 1], route[:, :, 0])
        e = bearing[:, 1]
        turn = np.maximum(np.abs(e), np.abs(bearing[:, 2]))
        v = np.clip(speed - slow_down * turn, min_speed, speed)
        return np.stack(
            [v / self.max_speed, np.clip(heading_gain * e / self.max_yaw_rate, -1.0, 1.0)], 1
        ).astype(np.float32)
