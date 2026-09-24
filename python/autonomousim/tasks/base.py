"""Base class of the single-agent tasks: a scenario plus vectorised reward and termination.

The scenario (maps, vehicle, spawn and goal sampling, randomisation, observations) runs in
Rust. The task only computes rewards and end conditions in numpy from the state rows and
event bits of all worlds at once, so reward shaping stays in Python.
"""

from typing import Any

import numpy as np

from autonomousim._native import TERMINAL_EVENTS
from autonomousim.scenario import deep_merge

#: Map shortcuts accepted by ``map=``. A dict is used as the map source itself.
MAPS = ("flat", "forest", "wild")


def map_source(name: str | dict[str, Any], seed: int, count: int) -> dict[str, Any]:
    """Map source of the scenario for a shortcut: ``flat`` (200 m grass plane), ``forest``
    (200 m hilly test forest, 150 trees/ha) or ``wild`` (a pool of ``count`` generated
    512 m training maps, one per episode)."""
    if isinstance(name, dict):
        return name
    if name == "flat":
        return {"type": "testworld", "kind": "flat", "size": 200.0}
    if name == "forest":
        return {"type": "testworld", "kind": "forest_patch", "size": 200.0, "density": 150.0, "seed": seed}
    if name == "wild":
        return {"type": "wild", "seed": seed, "count": count, "preset": "training"}
    raise ValueError(f"unknown map {name!r}; use one of {MAPS} or a map source dict")


class Task:
    """One agent per world. Subclasses define ``group()`` (spawn, goals, observations) and
    ``reward()``, optionally ``settings()`` (scenario-level entries), ``failed()`` and
    ``succeeded()``; the base class handles episode length, terminal events and the previous
    action.

    Keyword arguments (all tasks):
        vehicle: preset name (``cf2x``, ``iris_like``) or path to a vehicle TOML file.
        action_mode: how the normalised actions in [−1, 1] command the vehicle (full-scale
            values are the scenario's ``action_limits``):

            - ``motors``: rotor speeds from idle to maximum, one per rotor;
            - ``ctbr`` (default): roll, pitch and yaw rates (±2π, ±2π, ±π rad/s), collective
              thrust (−1: none, 1: maximum);
            - ``attitude``: tilt x and y (a disc of 35°), yaw rate (±π/2 rad/s), thrust;
            - ``velocity``: horizontal velocity in the heading frame (a disc of 5 m/s),
              vertical velocity (±2 m/s), yaw rate;
            - ``position``: offset from the current position in the heading frame
              (±5, ±5, ±2 m), heading change (±π).
        map, map_seed, map_count: see ``map_source``.
        physics_hz, policy_hz: simulation and action rates (the policy rate must divide the
            physics rate); ``physics_hz=None`` lets the scenario choose (500 Hz for aerial
            vehicles, 1 kHz with ground vehicles).
        episode_time: seconds until truncation.
        wind: ``[min, max]`` mean wind speed per episode (m/s, uniform direction), or None.
        randomize: relative spreads of the vehicle parameters, e.g. ``{"mass": 0.1}``.
        obs: observation terms replacing the task's (see ``docs/PLAN.md``).
        overrides: dict deep-merged into the final scenario (``autonomousim.scenario.deep_merge``).
    """

    name = "task"
    default_episode_time = 10.0
    #: Whether the task defines success (``succeeded``); evaluations then report its rate.
    has_success = False

    def __init__(
        self,
        *,
        vehicle: str = "cf2x",
        action_mode: str = "ctbr",
        map: str | dict[str, Any] = "flat",
        map_seed: int = 0,
        map_count: int = 16,
        physics_hz: int | None = None,
        policy_hz: int = 50,
        episode_time: float | None = None,
        wind: tuple[float, float] | None = None,
        randomize: dict[str, float] | None = None,
        obs: list[dict[str, Any]] | None = None,
        overrides: dict[str, Any] | None = None,
    ):
        self.vehicle = vehicle
        self.action_mode = action_mode
        self.map = map
        self.map_seed = map_seed
        self.map_count = map_count
        self.physics_hz = physics_hz
        self.policy_hz = policy_hz
        self.episode_time = self.default_episode_time if episode_time is None else episode_time
        self.wind = wind
        self.randomize = randomize or {}
        self.obs = obs
        self.overrides = overrides or {}
        self.num_envs = 0
        self.max_steps = 0

    # ------------------------------------------------------------------ scenario

    def group(self) -> dict[str, Any]:
        """The agent group of the scenario (spawn, goals, ...), without vehicle and mode."""
        return {}

    def settings(self) -> dict[str, Any]:
        """Scenario-level entries besides the map and the group (e.g. ``events``)."""
        return {}

    def scenario(self) -> dict[str, Any]:
        """The complete scenario dict passed to ``BatchSim``."""
        group = {
            "name": "agent",
            "count": 1,
            "vehicle": self.vehicle,
            "action_mode": self.action_mode,
            "randomize": self.randomize,
            **self.group(),
        }
        if self.obs is not None:
            group["obs"] = self.obs
        sc: dict[str, Any] = {
            "name": self.name,
            "physics_hz": self.physics_hz or 0,
            "policy_hz": self.policy_hz,
            "map": map_source(self.map, self.map_seed, self.map_count),
            "groups": [group],
        }
        if self.wind is not None:
            sc["randomize_environment"] = {"wind_speed": list(self.wind)}
        return deep_merge(deep_merge(sc, self.settings()), self.overrides)

    # ------------------------------------------------------------------ episodes

    def bind(self, num_envs: int, policy_dt: float, act_dim: int) -> None:
        """Allocate per-world buffers once the simulation exists."""
        self.num_envs = num_envs
        self.policy_dt = policy_dt
        self.max_steps = max(1, round(self.episode_time / policy_dt))
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.prev_action = np.zeros((num_envs, act_dim), dtype=np.float32)
        self.success = np.zeros(num_envs, dtype=bool)

    def reset(self, mask: np.ndarray | None = None, state: np.ndarray | None = None) -> None:
        """Start new episodes in the worlds of ``mask`` (all if None). ``state`` holds the
        first state rows of the new episodes (all worlds)."""
        m = slice(None) if mask is None else mask
        self.steps[m] = 0
        self.prev_action[m] = 0.0
        self.success[m] = False

    def compute(
        self, state: np.ndarray, events: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reward (float64), terminated and truncated (bool) of every world after a step.

        ``state`` is ``[num_envs, STATE_DIM]``, ``events`` ``[num_envs]`` (uint32) and
        ``actions`` the actions just applied, ``[num_envs, act_dim]``. Episodes end on
        success (``succeeded``) or failure (a terminal event or ``failed``); failures cost
        ``terminal_penalty``. ``self.success`` marks the worlds that succeeded on this step.
        """
        # The simulator clips actions and reads non-finite values as 0; so does the reward.
        a = np.clip(np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        self.steps += 1
        failure = ((events & TERMINAL_EVENTS) != 0) | self.failed(state)
        self.success[:] = self.succeeded(state, events) & ~failure
        terminated = failure | self.success
        reward = self.reward(state, a, self.prev_action, events)
        reward[failure] -= self.terminal_penalty
        truncated = (self.steps >= self.max_steps) & ~terminated
        self.prev_action[:] = a
        return reward, terminated, truncated

    terminal_penalty = 0.0

    def failed(self, state: np.ndarray) -> np.ndarray:
        """Task-specific end conditions besides the terminal events."""
        return np.zeros(len(state), dtype=bool)

    def succeeded(self, state: np.ndarray, events: np.ndarray) -> np.ndarray:
        """Worlds whose episode ends successfully on this step (none by default)."""
        return np.zeros(len(state), dtype=bool)

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError
