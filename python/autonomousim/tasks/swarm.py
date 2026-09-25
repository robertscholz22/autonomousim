"""SwarmHover-v0: a swarm of drones flies into formation and holds it without touching."""

from typing import Any

import numpy as np

from autonomousim.scenario import STATE, quat_up_z
from autonomousim.tasks.base import Task
from autonomousim.tasks.multi import MULTI_TASKS, MultiAgentTask, Team


class FormationHover(Task):
    """One drone of a swarm (``velocity`` mode): fly to its formation slot and hold it while
    keeping clear of the others.

    The group spawns 1.5–3.5 m above the ground inside a ``cluster`` × ``cluster`` m square
    placed at random on the map, at least ``min_separation`` apart, with up to 10° tilt and
    0.5 m/s. The slots form a ``formation`` (``grid`` or ``circle``) with neighbours
    ``spacing`` apart, centred on the spawns' centroid, 2–3 m above the ground; agent ``k``
    flies to slot ``k``, so paths cross. The observation is the goal, velocity, attitude,
    rates and last action in the heading frame, the ``neighbors`` nearest other agents and
    the distance to the nearest one (``7·neighbors + 20`` values).

    Reward per step: ``exp(−‖e‖) − distance_weight·‖e‖ − proximity_weight·max(0, 1 − d/safe_distance)
    − spin_weight·‖ω‖ − smooth_weight·‖Δa‖²`` with ``e`` the goal error and ``d`` the
    surface distance to the nearest other agent, minus ``terminal_penalty`` when the drone
    stops early: on a terminal event (a crash, including into another agent), when its goal
    error exceeds ``bounds`` on any axis or its tilt exceeds ``max_tilt_deg``.
    """

    name = "formation_hover"
    default_episode_time = 15.0

    def __init__(
        self,
        *,
        cluster: float = 8.0,
        min_separation: float = 1.5,
        formation: str = "grid",
        spacing: float = 2.0,
        neighbors: int = 3,
        bounds: float = 12.0,
        max_tilt_deg: float = 90.0,
        safe_distance: float = 1.5,
        distance_weight: float = 0.05,
        proximity_weight: float = 1.0,
        spin_weight: float = 0.05,
        smooth_weight: float = 0.01,
        terminal_penalty: float = 5.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("action_mode", "velocity")
        super().__init__(**kwargs)
        self.cluster = cluster
        self.min_separation = min_separation
        self.formation = formation
        self.spacing = spacing
        self.neighbors = neighbors
        self.bounds = bounds
        self.min_up = float(np.cos(np.radians(max_tilt_deg)))
        self.safe_distance = safe_distance
        self.distance_weight = distance_weight
        self.proximity_weight = proximity_weight
        self.spin_weight = spin_weight
        self.smooth_weight = smooth_weight
        self.terminal_penalty = terminal_penalty

    def group(self) -> dict[str, Any]:
        return {
            "spawn": {
                "cluster": self.cluster,
                "min_separation": self.min_separation,
                "agl": [1.5, 3.5],
                "tilt_deg": 10.0,
                "speed": 0.5,
                "clearance": 1.0,
                "margin": 20.0,
            },
            "goals": {
                "kind": "formation",
                "formation": self.formation,
                "spacing": self.spacing,
                "agl": [2.0, 3.0],
                "margin": 20.0,
            },
            "obs": [
                {"term": "goal_rel_heading", "scale": 0.5, "clip": 5.0},
                {"term": "rot6d"},
                {"term": "lin_vel_heading", "scale": 0.5, "clip": 5.0},
                {"term": "ang_vel_body", "scale": 0.1, "clip": 5.0},
                {"term": "last_action"},
                {"term": "neighbors", "count": self.neighbors, "range": 10.0, "scale": 0.5, "clip": 5.0},
                {"term": "nearest_agent", "range": 5.0, "scale": 0.5},
            ],
        }

    def settings(self) -> dict[str, Any]:
        return {"events": {"crash_speed": 1.0}}

    def failed(self, state: np.ndarray) -> np.ndarray:
        err = state[:, STATE["goal"]] - state[:, STATE["position"]]
        return (np.abs(err) > self.bounds).any(axis=1) | (quat_up_z(state[:, STATE["orientation"]]) < self.min_up)

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        err = np.linalg.norm(state[:, STATE["goal"]] - state[:, STATE["position"]], axis=1)
        near = np.clip(1.0 - state[:, STATE["agent_clearance"]][:, 0] / self.safe_distance, 0.0, 1.0)
        rates = np.linalg.norm(state[:, STATE["rates"]], axis=1)
        da = action - prev_action
        return (
            np.exp(-err)
            - self.distance_weight * err
            - self.proximity_weight * near
            - self.spin_weight * rates
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
        )


class SwarmHover(MultiAgentTask):
    """SwarmHover-v0: ``count`` drones (default 8, one group ``drones`` sharing a policy) fly
    from a random cluster into formation slots and hold them without touching (see
    ``FormationHover``, which takes the remaining keyword arguments). Truncated after
    ``episode_time`` (15 s) on a flat 200 m map by default.

    ``formation_error(state)`` gives the goal error of every agent; an episode is a success
    for the swarm when, at its end, the mean error is below 0.3 m and no agent touched
    another (see ``examples/ppo_multiagent.py``).
    """

    name = "swarm_hover"
    default_episode_time = 15.0

    def __init__(
        self,
        *,
        count: int = 8,
        vehicle: str = "cf2x",
        map: str | dict[str, Any] = "flat",
        map_seed: int = 0,
        map_count: int = 16,
        physics_hz: int | None = None,
        policy_hz: int | None = None,
        episode_time: float | None = None,
        wind: tuple[float, float] | None = None,
        overrides: dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ):
        super().__init__(
            map=map,
            map_seed=map_seed,
            map_count=map_count,
            physics_hz=physics_hz,
            policy_hz=policy_hz,
            episode_time=episode_time,
            wind=wind,
            overrides=overrides,
        )
        self.count = count
        self.agent = FormationHover(vehicle=vehicle, **agent_kwargs)

    def teams(self) -> dict[str, Team]:
        return {"drones": Team(self.agent, self.count)}

    @staticmethod
    def formation_error(state: np.ndarray) -> np.ndarray:
        """Distance of every agent from its slot, ``state[..., STATE_DIM] → [...]`` (m)."""
        return np.linalg.norm(state[..., STATE["goal"]] - state[..., STATE["position"]], axis=-1)


MULTI_TASKS["swarm_hover"] = SwarmHover
