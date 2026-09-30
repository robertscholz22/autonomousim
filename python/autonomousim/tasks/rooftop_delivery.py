"""DroneRooftopDelivery-v0: fly a quadrotor across a generated city to a landing pad on a
flat roof, over and between the buildings, with LiDAR; and a scripted pilot that proves the
task solvable."""

from typing import Any

import numpy as np

from autonomousim.events import Event
from autonomousim.scenario import STATE
from autonomousim.tasks.base import Task

LANDED = int(Event.LANDED)

#: ``velocity`` limits: horizontal speed (m/s; the iris-like controller's limit is 12) and
#: vertical speed (m/s).
SPEED_XY = 10.0
SPEED_Z = 3.0

#: Observation scales.
GOAL_SCALE = 0.01
FINE_SCALE = 0.1
VEL_SCALE = 0.1
RATE_SCALE = 0.5
AGL_SCALE = 0.02

#: The ``rl64`` layout: 4 rings (elevations in degrees) × 16 azimuths from straight ahead,
#: counter-clockwise, 40 m.
ELEVATIONS = (-24.0, -8.0, 8.0, 24.0)
AZIMUTHS = 16
LIDAR_RANGE = 40.0
#: The downward rangefinder's range (m).
RANGE_MAX = 40.0


def _beams() -> np.ndarray:
    """Unit beam directions of the ``rl64`` layout in the body frame, ring by ring."""
    el = np.radians(np.repeat(ELEVATIONS, AZIMUTHS))
    az = np.radians(np.tile(np.arange(AZIMUTHS) * 360.0 / AZIMUTHS, len(ELEVATIONS)))
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)


BEAMS = _beams()


class DroneRooftopDelivery(Task):
    """An iris-like quadrotor in ``velocity`` mode (horizontal speed up to ``SPEED_XY`` in the
    heading frame, vertical speed up to ``SPEED_Z``, yaw rate) delivers to a landing pad on a
    flat roof of a generated urban map (a pool of ``map_count`` 512 m training cities from
    ``map_seed``). It starts on its gear on a sidewalk, or (a share ``roof_start`` of the
    episodes) on another rooftop pad, ``goal_distance`` (horizontally) from the goal pad.

    Wind: a mean of ``wind`` m/s from a uniform direction. Terrain or obstacle strikes (the
    buildings, street trees, lamp posts), water and touchdowns faster than 2 m/s are crashes.

    Observation (88 values): the goal pad in the heading frame (scaled 1/100, clipped to ±8)
    and again at a fine scale for the last metres (1/10, clipped to ±1), body velocity (1/10)
    and rates (1/2), rot6d, height above the ground (1/50, clipped to ±5; the terrain, not the
    roofs), the downward rangefinder (range / 40 m, 1 without a return), the last action and
    an ``rl64`` LiDAR scan (log ranges).

    Reward per step (``d``: distance to the pad): ``progress_weight`` × the decrease of
    ``d``, ``−time_weight``, ``−proximity_weight·max(0, 1 − c/proximity_distance)²`` for the
    clearance ``c`` to the nearest terrain or solid obstacle (not within 5 m of the pad),
    ``−smooth_weight·‖Δa‖²``, ``landing_bonus`` on landing within ``landing_radius`` of the
    pad's centre (the success, which ends the episode) and ``−terminal_penalty`` on a crash,
    water or leaving the map. Setting down elsewhere is allowed: it can lift off again.

    The episode is truncated after ``episode_time`` (150 s). The policy runs at 10 Hz.
    ``scripted(obs)`` gives the actions of a pilot that climbs out, flies toward the pad
    above ``cruise_above`` m over it, climbing instead wherever the LiDAR sees something ahead
    not well below it, and descends onto the pad from overhead.
    """

    name = "rooftop_delivery"
    default_episode_time = 150.0
    has_success = True

    def __init__(
        self,
        *,
        goal_distance: tuple[float, float] = (200.0, 600.0),
        roof_start: float = 0.3,
        landing_radius: float = 3.0,
        progress_weight: float = 0.05,
        time_weight: float = 0.002,
        proximity_weight: float = 0.1,
        proximity_distance: float = 2.0,
        smooth_weight: float = 0.01,
        landing_bonus: float = 20.0,
        terminal_penalty: float = 20.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("vehicle", "iris_like")
        kwargs.setdefault("action_mode", "velocity")
        kwargs.setdefault("map", "urban")
        kwargs.setdefault("map_count", 8)
        kwargs.setdefault("policy_hz", 10)
        kwargs.setdefault("wind", (0.0, 3.0))
        super().__init__(**kwargs)
        self.goal_distance = goal_distance
        self.roof_start = roof_start
        self.landing_radius = landing_radius
        self.progress_weight = progress_weight
        self.time_weight = time_weight
        self.proximity_weight = proximity_weight
        self.proximity_distance = proximity_distance
        self.smooth_weight = smooth_weight
        self.landing_bonus = landing_bonus
        self.terminal_penalty = terminal_penalty

    def group(self) -> dict[str, Any]:
        return {
            "spawn": {"on_ground": True},
            "goals": {
                "kind": "rooftop",
                "distance": list(self.goal_distance),
                "agl": [0.0, 0.0],
                "radius": 0.0,
                "rooftop": {"roof_start": self.roof_start},
            },
            "action_limits": {"speed_xy": SPEED_XY, "speed_z": SPEED_Z},
            "sensors": [
                {"name": "lidar", "type": "lidar"},
                {"name": "down", "type": "rangefinder", "max_range": RANGE_MAX},
            ],
            "obs": [
                {"term": "goal_rel_heading", "scale": GOAL_SCALE, "clip": 8.0},
                {"term": "goal_rel_heading", "scale": FINE_SCALE, "clip": 1.0},
                {"term": "lin_vel_body", "scale": VEL_SCALE},
                {"term": "ang_vel_body", "scale": RATE_SCALE},
                {"term": "rot6d"},
                {"term": "agl", "scale": AGL_SCALE, "clip": 5.0},
                {"term": "range", "sensor": "down"},
                {"term": "last_action"},
                {"term": "lidar_log", "sensor": "lidar"},
            ],
        }

    def settings(self) -> dict[str, Any]:
        return {"events": {"crash_speed": 2.0}}

    # ------------------------------------------------------------------ episodes

    def bind(self, num_envs: int, policy_dt: float, act_dim: int) -> None:
        super().bind(num_envs, policy_dt, act_dim)
        self.prev_distance = np.zeros(num_envs)

    def reset(self, mask: np.ndarray | None = None, state: np.ndarray | None = None) -> None:
        super().reset(mask, state)
        if state is not None:
            m = slice(None) if mask is None else mask
            self.prev_distance[m] = self.distance(state[m])

    def distance(self, state: np.ndarray) -> np.ndarray:
        """Distance to the pad's centre (m)."""
        return np.linalg.norm(state[:, STATE["goal"]] - state[:, STATE["position"]], axis=1)

    def succeeded(self, state: np.ndarray, events: np.ndarray) -> np.ndarray:
        offset = state[:, STATE["goal"]] - state[:, STATE["position"]]
        on_pad = (np.hypot(offset[:, 0], offset[:, 1]) < self.landing_radius) & (np.abs(offset[:, 2]) < 1.0)
        return ((events & LANDED) != 0) & on_pad

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        d = self.distance(state)
        progress = self.prev_distance - d
        self.prev_distance[:] = d
        clearance = state[:, STATE["clearance"]][:, 0]
        near = np.clip(1.0 - clearance / self.proximity_distance, 0.0, 1.0) * (d > 5.0)
        da = action - prev_action
        return (
            self.progress_weight * progress
            - self.time_weight
            - self.proximity_weight * near**2
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
            + self.landing_bonus * self.succeeded(state, events)
        )

    # ------------------------------------------------------------------ scripted pilot

    def scripted(
        self,
        obs: np.ndarray,
        cruise_above: float = 8.0,
        takeoff: float = 3.0,
        margin: float = 3.0,
        lookahead: float = 10.0,
        lookahead_time: float = 1.5,
        cone: float = 0.7,
        floor: float = 4.0,
        decel: float = 1.5,
        position_gain: float = 0.8,
        approach: float = 10.0,
        overhead: float = 1.5,
        descent: float = 1.5,
        touchdown: float = 0.5,
    ) -> np.ndarray:
        """Normalised ``velocity`` actions of a pilot for the observations ``obs`` (the task's
        default terms).

        On the ground it first climbs ``takeoff`` m (by the rangefinder). It is blocked while
        a LiDAR hit within ``cone`` rad of the direction to the pad lies less than
        ``margin`` m below it and closer (horizontally) than ``lookahead`` m plus
        ``lookahead_time`` s at its speed: it then brakes and climbs at full speed. Otherwise
        it flies toward the pad at the speed from which it can stop at ``decel`` m/s² (at most
        ``SPEED_XY``; ``position_gain`` times the distance close in), climbing while lower
        than ``cruise_above`` m over the pad (beyond ``approach`` m) or ``floor`` m over what
        lies below (beyond three times ``overhead``). Within
        ``overhead`` m of the pad's centre it descends at ``descent`` m/s, and at
        ``touchdown`` m/s for the last 3 m.
        """
        n = len(obs)
        goal = obs[:, 0:3] / GOAL_SCALE
        fine = obs[:, 3:6] / FINE_SCALE
        near = np.abs(goal).max(axis=1) < 9.0
        goal[near] = fine[near]
        v_body = obs[:, 6:9] / VEL_SCALE
        c1, c2 = obs[:, 12:15], obs[:, 15:18]
        rot = np.stack([c1, c2, np.cross(c1, c2)], 2)  # body → world, [n, 3, 3]
        below = np.where(obs[:, 19] < 0.999, obs[:, 19] * RANGE_MAX, np.inf)
        scan = obs[:, 24:24 + len(BEAMS)]
        hit = scan < 0.999
        ranges = np.expm1(scan * np.log1p(LIDAR_RANGE))
        # Hits relative to the drone in the heading frame: the world frame turned by −yaw.
        yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
        world = np.einsum("nij,bj->nbi", rot, BEAMS) * ranges[:, :, None]
        cy, sy = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        hx = cy * world[:, :, 0] + sy * world[:, :, 1]
        hy = -sy * world[:, :, 0] + cy * world[:, :, 1]
        hz = world[:, :, 2]

        d = np.maximum(np.hypot(goal[:, 0], goal[:, 1]), 1e-6)
        bearing = np.arctan2(goal[:, 1], goal[:, 0])
        speed_now = np.linalg.norm((rot @ v_body[:, :, None])[:, :2, 0], axis=1)
        reach = lookahead + lookahead_time * speed_now
        off = np.abs((np.arctan2(hy, hx) - bearing[:, None] + np.pi) % (2 * np.pi) - np.pi)
        ahead = hit & (off < cone) & (hz > -margin) & (np.hypot(hx, hy) < reach[:, None])
        blocked = ahead.any(axis=1) & (d > overhead)

        height = -goal[:, 2]  # above the pad
        grounded = below < takeoff
        speed = np.minimum(np.minimum(np.sqrt(2.0 * decel * d), position_gain * d), SPEED_XY)
        speed = np.where(blocked | grounded, 0.0, speed)
        vx, vy = speed * goal[:, 0] / d, speed * goal[:, 1] / d
        # Far out: up to the cruise height; always clear of what lies below.
        far = d > approach
        climb = blocked | grounded | ((height < cruise_above) & far) | ((below < floor) & (d > 3.0 * overhead))
        vz = np.where(climb, SPEED_Z, 0.0)
        # Over the pad: down.
        over = (d < overhead) & ~blocked
        vz = np.where(over, -np.where(height > 3.0, descent, touchdown), vz)
        out = np.stack([vx / SPEED_XY, vy / SPEED_XY, vz / SPEED_Z, np.zeros(n)], 1)
        return out.clip(-1.0, 1.0).astype(np.float32)
