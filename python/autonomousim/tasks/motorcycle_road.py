"""MotorcycleRoadRural-v0: ride a motorcycle along a route over the roads of generated
farmland to a farm yard, leaning into the bends, with ``(v, κ)`` actions (or unbalanced
``raw`` ones)."""

from typing import Any

import numpy as np

from autonomousim.tasks.road_follow import RoadFollowRural

#: Standard gravity (m/s²).
G = 9.80665

#: 2 rings (−3°, 3°) × 36 azimuths (10° apart), 40 m, 10 Hz, at the rider's eye height.
LIDAR = {
    "pattern": {"type": "rings", "elevations": [-3.0, 3.0], "azimuths": 36, "azimuth_fov": 360.0},
    "max_range": 40.0,
    "mount": {"position": [0.3, 0.0, 1.2]},
}

#: Index of each observation term in the task's default layout.
ROAD, ROUTE, ROAD_CLASS, SPEED, LAST_ACTION = slice(0, 6), slice(6, 14), slice(15, 18), 18, slice(32, 34)


class MotorcycleRoadRural(RoadFollowRural):
    """A motorcycle (``motorcycle_sport``) in ``vk`` mode (speed up to ``max_speed``, path
    curvature up to what a steady lean of ``max_lean`` gives at the commanded speed; the rider
    controller balances and countersteers) starts on its feet in a lane of a random road on a
    generated ``rural`` map and rides its route along the roads to a farm yard, as in
    ``RoadFollowRural`` (same goals, reward, end conditions and parameters). The roads are
    paved, gravel or dirt tracks, whose grip differs; ``road_class`` tells them apart.

    ``action_mode="raw"`` is the harder variant: throttle or brake, steering torque and rider
    lean, and the agent balances the motorcycle itself.

    Observation (106 values): ``road`` (lane offset, heading error, curvature 5–40 m ahead),
    ``route`` (lane points 5–40 m ahead, scaled 1/20), ``on_road``, ``road_class`` (paved,
    gravel, track), speed, body velocity and rates, ``lean`` (roll and its rate), ``steering``
    (the steering head's angle and rate), ``rider_lean``, ``feet``, last action and a LiDAR
    scan (``LIDAR``).

    Falling over ends the episode (``ROLLOVER``, or the rider or bodywork touching the
    ground: ``CRASH_TERRAIN``) with ``terminal_penalty``. The episode is truncated after
    ``episode_time`` (60 s).
    """

    name = "motorcycle_road"

    def __init__(self, *, max_speed: float = 25.0, max_lean: float = 0.45, lidar: dict[str, Any] | None = None, **kwargs: Any):
        kwargs.setdefault("vehicle", "motorcycle_sport")
        super().__init__(max_speed=max_speed, lidar=LIDAR if lidar is None else lidar, **kwargs)
        self.max_lean = max_lean

    def group(self) -> dict[str, Any]:
        g = super().group()
        g["ground_action_limits"] = {"speed": self.max_speed, "lean": self.max_lean}
        g["obs"] = [
            {"term": "road"},
            {"term": "route", "scale": 0.05},
            {"term": "on_road"},
            {"term": "road_class"},
            {"term": "speed", "scale": 0.1},
            {"term": "lin_vel_body", "scale": 0.1},
            {"term": "ang_vel_body", "scale": 0.5},
            {"term": "lean"},
            {"term": "steering"},
            {"term": "rider_lean"},
            {"term": "feet"},
            {"term": "last_action"},
            {"term": "lidar_log", "sensor": "lidar"},
        ]
        return g

    def curvature_limit(self, speed: np.ndarray, full: float) -> np.ndarray:
        """Full-scale curvature of ``vk`` at the commanded ``speed`` (m/s): the steering
        lock's ``full`` (1/m), or a steady lean of ``max_lean``'s, whichever is smaller."""
        return np.minimum(full, G * np.tan(self.max_lean) / np.maximum(speed, 1e-3) ** 2)

    def scripted(
        self,
        obs: np.ndarray,
        full_curvature: float,
        lateral: tuple[float, float, float] = (2.5, 1.5, 1.2),
        top: tuple[float, float, float] = (18.0, 12.0, 8.0),
        braking: float = 2.5,
        min_speed: float = 2.5,
        lookahead: float = 1.5,
        smoothing: float = 0.5,
        max_turn: float = 1.0,
    ) -> np.ndarray:
        """Normalised ``vk`` actions of a scripted rider for the observations ``obs`` (the
        task's default terms); ``full_curvature`` is the action map's full-scale curvature
        (``group_info(0)["full_scale"]["curvature"]`` of the native ``BatchSim``).

        Pure pursuit of the lane point ``clip(lookahead·v, 8, 40)`` m ahead (interpolated
        along the ``route`` term), its curvature low-passed over ``smoothing`` s (from the last
        action): the lean lags the steering, so shorter look-aheads or sudden turns weave.
        Speed: at most ``top`` m/s, ``√(lateral/κ)`` for the curvature κ it steers and, for
        the lane's curvature κ 5–40 m ahead, ``√(lateral/κ)`` plus what ``braking`` m/s²
        takes off before each bend (from 4 m short of it), at least ``min_speed``;
        ``lateral`` and ``top`` by road class (paved, gravel, track; off the road as on a
        track).
        """
        speed = obs[:, SPEED] * 10.0
        route = obs[:, ROUTE].reshape(-1, 4, 2) * 20.0
        ahead = np.clip(lookahead * speed, 8.0, 40.0)
        # Between the points 5, 10, 20 and 40 m ahead.
        marks = np.array([5.0, 10.0, 20.0, 40.0])
        i = np.clip(np.searchsorted(marks, ahead) - 1, 0, 2)
        w = ((ahead - marks[i]) / (marks[i + 1] - marks[i]))[:, None]
        rows = np.arange(len(obs))
        point = (1 - w) * route[rows, i] + w * route[rows, i + 1]
        curvature = 2.0 * point[:, 1] / np.maximum((point**2).sum(1), 1.0)
        last = obs[:, LAST_ACTION]
        last_curvature = last[:, 1] * self.curvature_limit(last[:, 0] * self.max_speed, full_curvature)
        alpha = min(1.0, self.policy_dt / smoothing) if smoothing > 0.0 else 1.0
        curvature = last_curvature + alpha * (curvature - last_curvature)
        cls = obs[:, ROAD_CLASS]
        k = np.where(cls.sum(1) > 0.5, cls.argmax(1), 2)
        a_lat, v_top = np.asarray(lateral)[k], np.asarray(top)[k]
        bends = np.abs(obs[:, ROAD][:, 2:6])
        v_bend = np.sqrt(a_lat[:, None] / np.maximum(bends, 1e-3) + 2.0 * braking * (marks - 4.0))
        v_turn = np.sqrt(a_lat / np.maximum(np.abs(curvature), 1e-3))
        target = np.clip(np.minimum(np.minimum(v_top, v_bend.min(1)), v_turn), min_speed, None)
        turn = curvature / self.curvature_limit(target, full_curvature)
        return np.stack([target / self.max_speed, np.clip(turn, -max_turn, max_turn)], 1).astype(np.float32)
