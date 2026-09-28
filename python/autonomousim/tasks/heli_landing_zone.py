"""HeliLandingZone-v0: fly a small helicopter from forward flight to a landing zone a few
hundred metres away over generated off-road terrain and set it down there, with
``velocity`` actions; and a scripted pilot that proves the task solvable."""

from typing import Any

import numpy as np

from autonomousim.events import Event
from autonomousim.scenario import STATE
from autonomousim.tasks.base import Task

LANDED = int(Event.LANDED)

#: ``velocity`` speeds (m/s) at ±1: forward, sideways (and backward), vertical (below the
#: 2 m/s crash sink rate, so that a full-down touchdown survives); and the yaw rate (rad/s).
SPEEDS = (15.0, 6.0, 1.5)
YAW_RATE = 0.5

#: Observation scales.
GOAL_SCALE = 0.01
FINE_SCALE = 0.1
AGL_SCALE = 0.05
VEL_SCALE = 0.1


class HeliLandingZone(Task):
    """The X-Cell-like model helicopter (``xcell60_like``) in ``velocity`` mode (forward,
    sideways and vertical speed in the heading frame up to ``SPEEDS``, and a yaw rate up to
    ``YAW_RATE``) starts in straight and level flight at ``spawn_airspeed`` and
    ``spawn_agl`` above the ground of a generated ``offroad`` wild map (low relief, open
    forest with clearings) and lands in a landing zone ``goal_distance`` (horizontally) away:
    a spot where the ground rises or falls at most ``landing_slope`` per metre within
    ``landing_clearance`` m, with no water within that radius and no tree or rock within it
    of the vertical up to 40 m (see the scenario's ``goals.landing_slope``).

    Wind: a mean of ``wind`` m/s from a uniform direction and Dryden turbulence of intensity
    ``turbulence`` (W20, m/s). Terrain or obstacle strikes and
    touchdowns faster than 2 m/s are crashes.

    Observation (19 values): goal in the heading frame (scaled 1/100, clipped to ±5), height
    above ground (1/20, clipped to ±5), body velocity (1/10) and rates, pitch and roll, the
    last action and the goal again at a fine scale for the last metres (1/10, clipped to ±1).

    Reward per step (``d``: horizontal distance to the zone; ``h``: height above ground): the
    decrease of the potential ``φ = √(d² + h²)`` over 10 m (closing in and descending both
    pay, descending the more the closer the zone), ``−time_weight``, ``−smooth_weight·‖Δa‖²``,
    ``landing_bonus`` on landing within ``landing_radius`` of the zone's centre (the
    success, which ends the episode) and ``−terminal_penalty`` on a crash, water or leaving
    the map. Landing elsewhere is allowed: the helicopter can lift off again.

    The episode is truncated after ``episode_time`` (120 s). The policy runs at 20 Hz, the
    physics at 500 Hz. ``scripted(obs)`` gives the actions of a pilot that flies to the zone
    at its spawn height, stops over it and descends.
    """

    name = "heli_landing_zone"
    default_episode_time = 120.0
    has_success = True

    def __init__(
        self,
        *,
        goal_distance: tuple[float, float] = (100.0, 300.0),
        landing_slope: float = 0.1,
        landing_clearance: float = 3.0,
        landing_radius: float = 2.0,
        spawn_agl: tuple[float, float] = (35.0, 50.0),
        spawn_airspeed: tuple[float, float] = (4.0, 10.0),
        turbulence: tuple[float, float] = (0.0, 3.0),
        margin: float = 50.0,
        time_weight: float = 0.005,
        smooth_weight: float = 0.02,
        landing_bonus: float = 20.0,
        terminal_penalty: float = 20.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("vehicle", "xcell60_like")
        kwargs.setdefault("action_mode", "velocity")
        kwargs.setdefault("map", "offroad")
        kwargs.setdefault("policy_hz", 20)
        kwargs.setdefault("wind", (0.0, 3.0))
        super().__init__(**kwargs)
        self.goal_distance = goal_distance
        self.landing_slope = landing_slope
        self.landing_clearance = landing_clearance
        self.landing_radius = landing_radius
        self.spawn_agl = spawn_agl
        self.spawn_airspeed = spawn_airspeed
        self.turbulence = turbulence
        self.margin = margin
        self.time_weight = time_weight
        self.smooth_weight = smooth_weight
        self.landing_bonus = landing_bonus
        self.terminal_penalty = terminal_penalty

    def group(self) -> dict[str, Any]:
        return {
            "spawn": {
                "agl": list(self.spawn_agl),
                "airspeed": list(self.spawn_airspeed),
                "clearance": 10.0,
                "margin": self.margin,
            },
            "goals": {
                "kind": "random",
                "distance": list(self.goal_distance),
                "agl": [0.0, 0.0],
                "landing_slope": self.landing_slope,
                "clearance": self.landing_clearance,
                "margin": self.margin,
                "radius": 0.0,
            },
            "helicopter_action_limits": {"speed": list(SPEEDS), "yaw_rate": YAW_RATE},
            "obs": [
                {"term": "goal_rel_heading", "scale": GOAL_SCALE, "clip": 5.0},
                {"term": "agl", "scale": AGL_SCALE, "clip": 5.0},
                {"term": "lin_vel_body", "scale": VEL_SCALE},
                {"term": "ang_vel_body", "scale": 1.0},
                {"term": "pitch_roll"},
                {"term": "last_action"},
                {"term": "goal_rel_heading", "scale": FINE_SCALE, "clip": 1.0},
            ],
        }

    def settings(self) -> dict[str, Any]:
        return {"randomize_environment": {"turbulence_w20": list(self.turbulence)}}

    # ------------------------------------------------------------------ episodes

    def bind(self, num_envs: int, policy_dt: float, act_dim: int) -> None:
        super().bind(num_envs, policy_dt, act_dim)
        self.prev_potential = np.zeros(num_envs)

    def reset(self, mask: np.ndarray | None = None, state: np.ndarray | None = None) -> None:
        super().reset(mask, state)
        if state is not None:
            m = slice(None) if mask is None else mask
            self.prev_potential[m] = self.potential(state[m])

    def distance(self, state: np.ndarray) -> np.ndarray:
        """Horizontal distance to the landing zone (m)."""
        return np.linalg.norm((state[:, STATE["goal"]] - state[:, STATE["position"]])[:, :2], axis=1)

    def potential(self, state: np.ndarray) -> np.ndarray:
        d = self.distance(state)
        h = np.maximum(state[:, STATE["agl"]][:, 0], 0.0)
        return np.hypot(d, h)

    def succeeded(self, state: np.ndarray, events: np.ndarray) -> np.ndarray:
        return ((events & LANDED) != 0) & (self.distance(state) < self.landing_radius)

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        potential = self.potential(state)
        progress = (self.prev_potential - potential) / 10.0
        self.prev_potential[:] = potential
        da = action - prev_action
        return (
            progress
            - self.time_weight
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
            + self.landing_bonus * self.succeeded(state, events)
        )

    # ------------------------------------------------------------------ scripted pilot

    def scripted(
        self,
        obs: np.ndarray,
        decel: float = 0.8,
        position_gain: float = 0.5,
        yaw_gain: float = 1.5,
        overhead: float = 1.5,
        flare_agl: float = 4.0,
        descent: float = 1.2,
        touchdown: float = 0.5,
    ) -> np.ndarray:
        """Normalised ``velocity`` actions of a pilot for the observations ``obs`` (the task's
        default terms).

        It turns towards the zone at ``yaw_gain`` times its bearing while it is more than 10 m
        away, and flies towards it at the speed from which it can stop at ``decel`` m/s² (at
        most the full forward speed; ``position_gain`` times the distance close in), holding
        its height. Within ``overhead`` m of the centre it descends at ``descent`` m/s, and
        at ``touchdown`` m/s below ``flare_agl`` m, down to the ground.
        """
        goal = obs[:, 0:3] / GOAL_SCALE
        agl = obs[:, 3] / AGL_SCALE
        d = np.maximum(np.hypot(goal[:, 0], goal[:, 1]), 1e-6)
        bearing = np.arctan2(goal[:, 1], goal[:, 0])
        speed = np.minimum(np.sqrt(2.0 * decel * d), position_gain * d)
        # Towards the zone in the heading frame: forward up to the forward speed, sideways
        # and backward up to the sideways speed.
        vx = speed * goal[:, 0] / d
        vy = speed * goal[:, 1] / d
        forward = np.where(vx >= 0.0, vx / SPEEDS[0], vx / SPEEDS[1])
        side = vy / SPEEDS[1]
        scale = np.maximum(1.0, np.maximum(np.abs(forward), np.abs(side)))
        forward, side = forward / scale, side / scale
        yaw = np.where(d > 10.0, np.clip(yaw_gain * bearing / YAW_RATE, -1.0, 1.0), 0.0)
        down = np.where(agl > flare_agl, descent, touchdown)
        vertical = np.where(d < overhead, -down / SPEEDS[2], 0.0)
        return np.stack([forward, side, vertical, yaw], 1).clip(-1.0, 1.0).astype(np.float32)
