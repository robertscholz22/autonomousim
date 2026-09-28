"""TiltrotorDelivery-v0: take a quad tiltrotor off vertically from a farm yard, transition to
the wing, cruise kilometres across farmland, transition back and land on another farm's
yard, with ``velocity`` actions; and a scripted pilot that proves the task solvable."""

from typing import Any

import numpy as np

from autonomousim.events import Event
from autonomousim.scenario import STATE
from autonomousim.tasks.base import Task

LANDED = int(Event.LANDED)

#: ``velocity`` speeds (m/s) at ±1: forward (on the wing; the mounts convert with it),
#: sideways and backward (in hover), vertical (below the 2 m/s crash sink rate, so that a
#: full-down touchdown survives); and the yaw rate (rad/s; on the wing, turns).
SPEEDS = (25.0, 4.0, 1.5)
YAW_RATE = 0.5

#: Observation scales.
GOAL_SCALE = 0.001
FINE_SCALE = 0.1
AGL_SCALE = 0.02
AIR_SCALE = 0.04
VEL_SCALE = 0.04


class TiltrotorDelivery(Task):
    """The quad tiltrotor (``quadtilt_like``) in ``velocity`` mode (forward speed up to
    ``SPEEDS[0]``, sideways and vertical speed, yaw rate up to ``YAW_RATE``; the controller
    converts the mounts with the forward speed) starts on its gear on the pad of a farm yard
    of a generated ``delivery`` farmland map (6 km, farms 400 m apart) and delivers to the pad
    of another farm yard ``goal_distance`` (horizontally) away. Pads are the most open spot of
    their yard (clear of buildings and trees up to 40 m). ``map="farmland"`` (2 km maps, fast
    to generate) with a shorter ``goal_distance`` suits training.

    Wind: a mean of ``wind`` m/s from a uniform direction and Dryden turbulence of intensity
    ``turbulence`` (W20, m/s). Terrain or obstacle strikes, water and touchdowns faster than
    2 m/s are crashes.

    Observation (22 values): goal in the heading frame (scaled 1/1000, clipped to ±5), height
    above ground (1/50, clipped to ±5), air data (airspeed, α, β; ×0.04), body velocity (1/25)
    and rates, pitch and roll, the last action and the goal again at a fine scale for the
    last metres (1/10, clipped to ±1).

    Reward per step (``d``: horizontal distance to the pad; ``h``: height above ground): the
    decrease of the potential ``φ = √(d² + h²)`` over 100 m, ``−time_weight``,
    ``−smooth_weight·‖Δa‖²``, ``−stall_weight`` on steps with a stall, ``landing_bonus`` on
    landing within ``landing_radius`` of the pad (the success, which ends the episode) and
    ``−terminal_penalty`` on a crash, water or leaving the map. Setting down elsewhere is
    allowed: it can lift off again.

    The episode is truncated after ``episode_time`` (360 s). The policy runs at 10 Hz, the
    physics at 500 Hz. ``scripted(obs)`` gives the actions of a pilot that climbs out
    vertically, turns toward the pad, converts and cruises at ``cruise_agl``, slows down so as
    to stop over the pad and descends onto it.
    """

    name = "tiltrotor_delivery"
    default_episode_time = 360.0
    has_success = True

    def __init__(
        self,
        *,
        goal_distance: tuple[float, float] = (2000.0, 4000.0),
        landing_radius: float = 3.0,
        turbulence: tuple[float, float] = (0.0, 2.0),
        time_weight: float = 0.002,
        smooth_weight: float = 0.02,
        stall_weight: float = 0.2,
        landing_bonus: float = 20.0,
        terminal_penalty: float = 20.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("vehicle", "quadtilt_like")
        kwargs.setdefault("action_mode", "velocity")
        kwargs.setdefault("map", "delivery")
        kwargs.setdefault("map_count", 1)
        kwargs.setdefault("policy_hz", 10)
        kwargs.setdefault("wind", (0.0, 4.0))
        super().__init__(**kwargs)
        self.goal_distance = goal_distance
        self.landing_radius = landing_radius
        self.turbulence = turbulence
        self.time_weight = time_weight
        self.smooth_weight = smooth_weight
        self.stall_weight = stall_weight
        self.landing_bonus = landing_bonus
        self.terminal_penalty = terminal_penalty

    def group(self) -> dict[str, Any]:
        return {
            "spawn": {"on_ground": True},
            "goals": {"kind": "yard", "distance": list(self.goal_distance), "agl": [0.0, 0.0], "radius": 0.0},
            "tiltrotor_action_limits": {"speed": list(SPEEDS), "yaw_rate": YAW_RATE},
            "obs": [
                {"term": "goal_rel_heading", "scale": GOAL_SCALE, "clip": 5.0},
                {"term": "agl", "scale": AGL_SCALE, "clip": 5.0},
                {"term": "air_data", "scale": AIR_SCALE},
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
        """Horizontal distance to the pad (m)."""
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
        progress = (self.prev_potential - potential) / 100.0
        self.prev_potential[:] = potential
        da = action - prev_action
        stall = (events & int(Event.STALL)) != 0
        return (
            progress
            - self.time_weight
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
            - self.stall_weight * stall
            + self.landing_bonus * self.succeeded(state, events)
        )

    # ------------------------------------------------------------------ scripted pilot

    def scripted(
        self,
        obs: np.ndarray,
        cruise_agl: float = 50.0,
        climb_out_agl: float = 25.0,
        decel: float = 0.6,
        position_gain: float = 0.4,
        height_gain: float = 0.2,
        yaw_gain: float = 1.0,
        aligned: float = 0.3,
        overhead: float = 1.5,
        flare_agl: float = 4.0,
        descent: float = 1.2,
        touchdown: float = 0.5,
    ) -> np.ndarray:
        """Normalised ``velocity`` actions of a pilot for the observations ``obs`` (the task's
        default terms).

        It climbs straight up to ``climb_out_agl`` m, turning toward the pad at ``yaw_gain``
        times its bearing (while more than 10 m away); once within ``aligned`` rad of the
        bearing and clear of the ground it flies toward the pad at the speed from which it can
        stop at ``decel`` m/s² (at most the full forward speed; ``position_gain`` times the
        distance close in), holding ``cruise_agl`` m above the ground at ``height_gain`` per
        second. The controller converts the mounts with the speed. Within ``overhead`` m of
        the pad it descends at ``descent`` m/s, and at ``touchdown`` m/s below ``flare_agl``
        m, down to the ground.
        """
        goal = obs[:, 0:3] / GOAL_SCALE
        fine = obs[:, 19:21] / FINE_SCALE
        # Close in, the fine goal term resolves the last metres.
        near = np.hypot(goal[:, 0], goal[:, 1]) < 9.0
        goal[near, :2] = fine[near]
        agl = obs[:, 3] / AGL_SCALE
        d = np.maximum(np.hypot(goal[:, 0], goal[:, 1]), 1e-6)
        bearing = np.arctan2(goal[:, 1], goal[:, 0])
        speed = np.minimum(np.sqrt(2.0 * decel * d), position_gain * d)
        # Hold still until clear of the ground and facing the pad (far out).
        go = (agl > climb_out_agl) | (d < 50.0)
        go &= (np.abs(bearing) < aligned) | (d < 50.0)
        speed = np.where(go, speed, 0.0)
        vx = speed * goal[:, 0] / d
        vy = speed * goal[:, 1] / d
        forward = np.where(vx >= 0.0, vx / SPEEDS[0], vx / SPEEDS[1])
        side = vy / SPEEDS[1]
        scale = np.maximum(1.0, np.maximum(np.abs(forward), np.abs(side)))
        forward, side = forward / scale, side / scale
        yaw = np.where(d > 10.0, np.clip(yaw_gain * bearing / YAW_RATE, -1.0, 1.0), 0.0)
        hold = np.clip(height_gain * (cruise_agl - agl), -SPEEDS[2], SPEEDS[2]) / SPEEDS[2]
        # Far out: up to the cruise height; over the pad: down.
        down = np.where(agl > flare_agl, descent, touchdown)
        vertical = np.where(d < overhead, -down / SPEEDS[2], np.where(d < 50.0, 0.0, hold))
        return np.stack([forward, side, vertical, yaw], 1).clip(-1.0, 1.0).astype(np.float32)
