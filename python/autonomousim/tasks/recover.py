"""QuadRecover-v0: recover from any attitude and return to the spawn point."""

from typing import Any

import numpy as np

from autonomousim.scenario import STATE, quat_up_z
from autonomousim.tasks.base import Task


class QuadRecover(Task):
    """Spawn 3–5 m above the ground with any tilt (uniform angle up to 180° about a uniform
    horizontal axis, uniform heading), up to 5 rad/s and 3 m/s; hold the spawn position.

    Reward per step: ``exp(−‖spawn − position‖) + upright_weight·up_z − spin_weight·‖ω‖ −
    smooth_weight·‖Δa‖²`` (``up_z``: world z of the body z axis), minus ``terminal_penalty``
    when the episode ends early: on a terminal event or when the position error exceeds
    ``bounds`` on any axis. Truncated after ``episode_time`` (5 s).
    """

    name = "recover"
    default_episode_time = 5.0

    def __init__(
        self,
        *,
        agl: tuple[float, float] = (3.0, 5.0),
        tilt_deg: float = 180.0,
        speed: float = 3.0,
        rates: float = 5.0,
        bounds: float = 10.0,
        upright_weight: float = 0.5,
        spin_weight: float = 0.05,
        smooth_weight: float = 0.01,
        terminal_penalty: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.agl = agl
        self.tilt_deg = tilt_deg
        self.speed = speed
        self.rates = rates
        self.bounds = bounds
        self.upright_weight = upright_weight
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
                "clearance": 1.5,
                "margin": 20.0,
            },
            "goals": {"kind": "spawn"},
        }

    def failed(self, state: np.ndarray) -> np.ndarray:
        err = state[:, STATE["goal"]] - state[:, STATE["position"]]
        return (np.abs(err) > self.bounds).any(axis=1)

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        err = state[:, STATE["goal"]] - state[:, STATE["position"]]
        rates = state[:, STATE["rates"]]
        da = action - prev_action
        return (
            np.exp(-np.sqrt(np.einsum("ij,ij->i", err, err)))
            + self.upright_weight * quat_up_z(state[:, STATE["orientation"]])
            - self.spin_weight * np.sqrt(np.einsum("ij,ij->i", rates, rates))
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
        )
