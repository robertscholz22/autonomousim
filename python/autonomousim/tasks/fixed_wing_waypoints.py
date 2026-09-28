"""FixedWingWaypoints-v0: fly a small fixed-wing UAV through waypoints kilometres apart over a
large mountainous wild map in wind and turbulence, with ``attitude`` actions; and a scripted
pilot that proves the task solvable."""

from typing import Any

import numpy as np

from autonomousim.events import Event
from autonomousim.scenario import STATE
from autonomousim.tasks.base import Task

GOAL_REACHED = int(Event.GOAL_REACHED)
FINISHED = int(Event.FINISHED)
STALL = int(Event.STALL)

#: Terrain fan ahead and down: 4 rings (−30°, −15°, −6°, 0°) × 7 azimuths over 90°, 600 m,
#: 10 Hz, on the nose. At 25 m/s the level ring sees 24 s ahead, the lowest the ground about
#: 300 m ahead from 150 m up.
LIDAR_RINGS = (-30.0, -15.0, -6.0, 0.0)
LIDAR_AZIMUTHS = 7
LIDAR = {
    "pattern": {"type": "rings", "elevations": list(LIDAR_RINGS), "azimuths": LIDAR_AZIMUTHS, "azimuth_fov": 90.0},
    "max_range": 600.0,
}

#: Observation scales.
GOAL_SCALE = 0.002
AGL_SCALE = 0.01
AIR_SCALE = 0.04


class FixedWingWaypoints(Task):
    """The Aerosonde-like UAV (``aerosonde_like``) in ``attitude`` mode (bank up to ±45°,
    pitch up to ±20°, airspeed over the normal range) starts in the air, trimmed for level
    flight at ``spawn_agl`` above the ground of a generated ``large`` wild map (16 km,
    mountains up to 900 m of relief; ``map`` takes a smaller map source for quick tests and
    training, e.g. the same preset with ``config = {"size": 4096.0}``, whose tiles all fit in
    the cache) and flies through ``goals`` waypoints, each ``goal_distance`` (horizontally)
    from the previous one at ``goal_agl`` above the ground, and climbing or descending at most
    ``goal_grade`` per metre from it. Spawns and waypoints keep ``margin`` m from the map's
    edges. A waypoint counts once the aircraft's centre is within ``goal_radius``.

    Wind: a mean of ``wind`` m/s from a uniform direction, Dryden turbulence of intensity
    ``turbulence`` (W20, m/s) and occasional gusts. Terrain, trees and water are crashes.

    Observation (46 values): goal in the heading frame (scaled 1/500, clipped to ±3), height
    above ground (1/100, clipped to ±5), air data (airspeed, α, β; ×0.04), body velocity (1/25)
    and rates, pitch and roll, last action and a LiDAR fan ahead and down (``LIDAR``: range
    over max range; ``lidar`` replaces its settings).

    Reward per step (``d``: horizontal distance to the current goal; ``h``: height above
    ground):

    - ``progress_weight·(d_before − d_after)/100``: hectometres made good towards the goal;
    - ``goal_bonus`` per waypoint reached;
    - ``−altitude_weight·(1 − h/safe_agl)²`` below ``safe_agl``;
    - ``−stall_weight`` on steps with a stall;
    - ``−smooth_weight·‖Δa‖²``;
    - ``−terminal_penalty`` on a crash, water or leaving the map.

    The episode succeeds when the last waypoint is reached and is truncated after
    ``episode_time`` (600 s). The policy runs at 10 Hz, the physics at 500 Hz.
    ``scripted(obs)`` gives the actions of a pilot that flies at the waypoints and climbs
    over the terrain the fan sees.
    """

    name = "fixed_wing_waypoints"
    default_episode_time = 600.0
    has_success = True

    def __init__(
        self,
        *,
        goals: int = 3,
        goal_distance: tuple[float, float] = (1000.0, 3000.0),
        goal_agl: tuple[float, float] = (100.0, 200.0),
        goal_grade: float = 0.08,
        goal_radius: float = 50.0,
        spawn_agl: tuple[float, float] = (150.0, 150.0),
        turbulence: tuple[float, float] = (0.0, 7.7),
        margin: float = 1000.0,
        lidar: dict[str, Any] | None = None,
        progress_weight: float = 1.0,
        goal_bonus: float = 10.0,
        safe_agl: float = 50.0,
        altitude_weight: float = 0.5,
        stall_weight: float = 0.5,
        smooth_weight: float = 0.05,
        terminal_penalty: float = 50.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("vehicle", "aerosonde_like")
        kwargs.setdefault("action_mode", "attitude")
        kwargs.setdefault("map", "large")
        kwargs.setdefault("map_count", 1)
        kwargs.setdefault("policy_hz", 10)
        kwargs.setdefault("wind", (0.0, 6.0))
        super().__init__(**kwargs)
        self.goals = goals
        self.goal_distance = goal_distance
        self.goal_agl = goal_agl
        self.goal_grade = goal_grade
        self.goal_radius = goal_radius
        self.spawn_agl = spawn_agl
        self.turbulence = turbulence
        self.margin = margin
        self.lidar = LIDAR if lidar is None else lidar
        self.progress_weight = progress_weight
        self.goal_bonus = goal_bonus
        self.safe_agl = safe_agl
        self.altitude_weight = altitude_weight
        self.stall_weight = stall_weight
        self.smooth_weight = smooth_weight
        self.terminal_penalty = terminal_penalty

    def group(self) -> dict[str, Any]:
        return {
            "spawn": {"agl": list(self.spawn_agl), "clearance": 20.0, "margin": self.margin},
            "goals": {
                "kind": "random",
                "count": self.goals,
                "distance": list(self.goal_distance),
                "agl": list(self.goal_agl),
                "grade": self.goal_grade,
                "clearance": 20.0,
                "margin": self.margin,
                "radius": self.goal_radius,
            },
            "sensors": [{"name": "lidar", "type": "lidar", **self.lidar}],
            "obs": [
                {"term": "goal_rel_heading", "scale": GOAL_SCALE, "clip": 3.0},
                {"term": "agl", "scale": AGL_SCALE, "clip": 5.0},
                {"term": "air_data", "scale": AIR_SCALE},
                {"term": "lin_vel_body", "scale": 0.04},
                {"term": "ang_vel_body", "scale": 1.0},
                {"term": "pitch_roll"},
                {"term": "last_action"},
                {"term": "lidar", "sensor": "lidar"},
            ],
        }

    def settings(self) -> dict[str, Any]:
        return {
            "randomize_environment": {
                "turbulence_w20": list(self.turbulence),
                "gust_rate": 1.0,
                "gust_speed": [1.0, 4.0],
                "gust_duration": [2.0, 6.0],
                "gust_horizon": self.episode_time,
            },
            "events": {"bounds_margin": 200.0},
        }

    # ------------------------------------------------------------------ episodes

    def bind(self, num_envs: int, policy_dt: float, act_dim: int) -> None:
        super().bind(num_envs, policy_dt, act_dim)
        self.prev_position = np.zeros((num_envs, 3))

    def reset(self, mask: np.ndarray | None = None, state: np.ndarray | None = None) -> None:
        super().reset(mask, state)
        if state is not None:
            m = slice(None) if mask is None else mask
            self.prev_position[m] = state[m, STATE["position"]]

    def succeeded(self, state: np.ndarray, events: np.ndarray) -> np.ndarray:
        return (events & FINISHED) != 0

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        position = state[:, STATE["position"]]
        goal = state[:, STATE["goal"]]
        before = np.linalg.norm((goal - self.prev_position)[:, :2], axis=1)
        after = np.linalg.norm((goal - position)[:, :2], axis=1)
        self.prev_position[:] = position
        low = np.clip(1.0 - state[:, STATE["agl"]][:, 0] / self.safe_agl, 0.0, 1.0)
        da = action - prev_action
        return (
            self.progress_weight * (before - after) / 100.0
            + self.goal_bonus * ((events & GOAL_REACHED) != 0)
            - self.altitude_weight * low**2
            - self.stall_weight * ((events & STALL) != 0)
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
        )

    # ------------------------------------------------------------------ scripted pilot

    def scripted(
        self,
        obs: np.ndarray,
        bank_gain: float = 1.5,
        climb_gain: float = 0.01,
        max_climb: float = 0.12,
        clearance: float = 60.0,
        escape_bank: float = 0.8,
        climb_range: float = 400.0,
        escape_range: float = 300.0,
        turn_space: float = 250.0,
        airspeed: float = 0.3,
    ) -> np.ndarray:
        """Normalised ``attitude`` actions of a pilot for the observations ``obs`` (the task's
        default terms and ``LIDAR``).

        It banks ``bank_gain`` times the goal's bearing (up to the full ±45°) and flies a
        flight-path angle of ``climb_gain`` times the goal's height difference, or the gradient
        that keeps ``clearance`` m over every terrain point the LiDAR fan sees within
        ``climb_range`` m and nearer than the goal, whichever is higher, within ±``max_climb`` (about the Aerosonde's steepest climb); the pitch is that
        angle plus the current angle of attack. Terrain within ``escape_range`` m that needs a
        steeper climb turns it away at ``escape_bank`` (a fraction of the largest bank) towards the side of the fan
        with the longer ranges, and keeps turning that way until the climb ahead is feasible.
        A goal more than 57° off the nose and closer than ``turn_space`` m is left behind
        straight first, so that the turn back does not circle it. The airspeed action stays at
        ``airspeed``.
        """
        goal = obs[:, 0:3] / GOAL_SCALE
        alpha = obs[:, 5] / AIR_SCALE
        nose_up = -obs[:, 13]  # pitch_roll: Z-Y-X Euler pitch, positive nose down
        rings, azimuths = len(LIDAR_RINGS), LIDAR_AZIMUTHS
        scan = obs[:, 18 : 18 + rings * azimuths].reshape(-1, rings, azimuths)
        bearing = np.arctan2(goal[:, 1], goal[:, 0])
        horizontal = np.maximum(np.hypot(goal[:, 0], goal[:, 1]), 1.0)
        gamma = climb_gain * goal[:, 2] * np.minimum(1.0, 500.0 / horizontal)
        # Terrain points of the fan relative to the aircraft (wings-level approximation).
        max_range = float(self.lidar.get("max_range", 600.0))
        elevation = np.deg2rad(np.array(LIDAR_RINGS))[None, :, None] + nose_up[:, None, None]
        r = scan * max_range
        dz = r * np.sin(elevation)
        dx = np.maximum(r * np.cos(elevation), 1.0)
        grade = np.where(scan < 0.999, (dz + clearance) / dx, -np.inf)
        # Terrain beyond the goal does not matter until the goal is reached.
        reach = np.minimum(horizontal, climb_range)[:, None, None]
        need = np.where(dx < reach, grade, -np.inf).max(axis=(1, 2))
        near = np.where(dx < np.minimum(reach, escape_range), grade, -np.inf).max(axis=(1, 2))
        gamma = np.clip(np.maximum(gamma, need), -max_climb, max_climb)
        bank = np.clip(bank_gain * -bearing, -1.0, 1.0)
        # A goal behind and closer than two turn radii: straight on first, so that the turn
        # back does not circle it.
        bank = np.where((horizontal < turn_space) & (np.abs(bearing) > 1.0), 0.0, bank)
        # Azimuths run from the right (−45°) to the left (+45°); an escape turn keeps its
        # direction (the last roll action) until the terrain ahead is clear.
        half = azimuths // 2
        right, left = scan[:, :, :half].mean(axis=(1, 2)), scan[:, :, -half:].mean(axis=(1, 2))
        last = obs[:, 15]
        side = np.where(np.abs(np.abs(last) - escape_bank) < 1e-3, np.sign(last), np.where(right > left, 1.0, -1.0))
        bank = np.where(near > max_climb, side * escape_bank, bank)
        pitch = (gamma + alpha) / np.deg2rad(20.0)
        return np.stack([bank, np.clip(pitch, -1.0, 1.0), np.full_like(bank, airspeed)], 1).astype(np.float32)
