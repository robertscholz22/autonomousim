"""QuadHover-v0: fly to a goal up to 2 m away and hold it."""

from typing import Any

import numpy as np

from autonomousim.scenario import STATE, quat_up_z
from autonomousim.tasks.base import Task


class QuadHover(Task):
    """Spawn 1.5–3.5 m above the ground with up to 30° tilt, 1 m/s and 1 rad/s; the goal is
    up to 2 m away horizontally at 1.5–3.5 m AGL. The observation is the 19-value default
    (goal − position, rot6d, velocity, rates, last action).

    Reward per step: ``exp(−‖goal − position‖) − spin_weight·‖ω‖ − smooth_weight·‖Δa‖²``,
    minus ``terminal_penalty`` on the last step when the episode ends early. The episode ends
    on a terminal event (crash, water, out of bounds, NaN), when the goal error exceeds
    ``bounds`` on any axis (a box of ±5 m around the goal) or when the tilt exceeds
    ``max_tilt_deg``; it is truncated after ``episode_time`` (10 s).
    """

    name = "hover"
    default_episode_time = 10.0

    def __init__(
        self,
        *,
        goal_distance: tuple[float, float] = (0.0, 2.0),
        agl: tuple[float, float] = (1.5, 3.5),
        tilt_deg: float = 30.0,
        speed: float = 1.0,
        rates: float = 1.0,
        bounds: float = 5.0,
        max_tilt_deg: float = 90.0,
        spin_weight: float = 0.05,
        smooth_weight: float = 0.01,
        terminal_penalty: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.goal_distance = goal_distance
        self.agl = agl
        self.tilt_deg = tilt_deg
        self.speed = speed
        self.rates = rates
        self.bounds = bounds
        self.min_up = float(np.cos(np.radians(max_tilt_deg)))
        self.spin_weight = spin_weight
        self.smooth_weight = smooth_weight
        self.terminal_penalty = terminal_penalty

    def group(self) -> dict[str, Any]:
        return {
            "spawn": {
                "agl": list(self.agl),
                "tilt_deg": self.tilt_deg,
                "speed": self.speed,
                "rates": self.rates,
                "clearance": 1.0,
                "margin": 20.0,
            },
            "goals": {
                "kind": "random",
                "count": 1,
                "distance": list(self.goal_distance),
                "agl": list(self.agl),
                "clearance": 1.0,
                "margin": 20.0,
            },
        }

    def failed(self, state: np.ndarray) -> np.ndarray:
        err = state[:, STATE["goal"]] - state[:, STATE["position"]]
        return (np.abs(err) > self.bounds).any(axis=1) | (quat_up_z(state[:, STATE["orientation"]]) < self.min_up)

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        err = state[:, STATE["goal"]] - state[:, STATE["position"]]
        rates = state[:, STATE["rates"]]
        da = action - prev_action
        return (
            np.exp(-np.sqrt(np.einsum("ij,ij->i", err, err)))
            - self.spin_weight * np.sqrt(np.einsum("ij,ij->i", rates, rates))
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
        )
