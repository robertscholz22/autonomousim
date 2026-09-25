"""Swarm tasks: SwarmHover-v0 (fly into formation and hold it without touching) and
SwarmWaypointForest-v0 (every drone flies its own waypoints through generated forests)."""

from typing import Any

import numpy as np

from autonomousim.events import Event
from autonomousim.scenario import STATE, quat_up_z
from autonomousim.tasks.base import Task
from autonomousim.tasks.multi import MULTI_TASKS, MultiAgentTask, Team
from autonomousim.tasks.waypoint_forest import QuadWaypointForest


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


class SwarmForestDrone(QuadWaypointForest):
    """One drone of a swarm in the forest: ``QuadWaypointForest`` with the group spawning
    together and the other drones in view.

    The group spawns inside a ``cluster`` × ``cluster`` m square placed at random on the map,
    at least ``min_separation`` apart; every drone then flies its own chain of waypoints from
    its spawn. The observation adds the ``neighbors`` nearest other drones within 20 m and the
    distance to the nearest one (``148 + 7·neighbors + 1`` values). The reward adds
    ``−agent_weight·max(0, 1 − d/agent_distance)²`` with ``d`` the surface distance to the
    nearest other drone; touching another drone faster than 2 m/s is a crash, which costs
    ``agent_crash_penalty`` on top of the terminal penalty.
    """

    name = "swarm_forest_drone"

    def __init__(
        self,
        *,
        cluster: float = 20.0,
        min_separation: float = 4.0,
        neighbors: int = 3,
        agent_weight: float = 2.0,
        agent_distance: float = 4.0,
        agent_crash_penalty: float = 50.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.cluster = cluster
        self.min_separation = min_separation
        self.neighbors = neighbors
        self.agent_weight = agent_weight
        self.agent_distance = agent_distance
        self.agent_crash_penalty = agent_crash_penalty

    def group(self) -> dict[str, Any]:
        g = super().group()
        g["spawn"] = {**g["spawn"], "cluster": self.cluster, "min_separation": self.min_separation, "clearance": 2.0}
        g["obs"] = [
            *g["obs"],
            {"term": "neighbors", "count": self.neighbors, "range": 20.0, "scale": 0.1, "clip": 3.0},
            {"term": "nearest_agent", "range": 10.0, "scale": 0.2},
        ]
        return g

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        near = np.clip(1.0 - state[:, STATE["agent_clearance"]][:, 0] / self.agent_distance, 0.0, 1.0)
        crash = (events & Event.CRASH_AGENT) != 0
        return (
            super().reward(state, action, prev_action, events)
            - self.agent_weight * near**2
            - self.agent_crash_penalty * crash
        )


class SwarmWaypointForest(MultiAgentTask):
    """SwarmWaypointForest-v0: ``count`` iris-like drones (default 8, one group ``drones``
    sharing a policy) start together in a generated forest and each flies its own chain of
    waypoints (see ``SwarmForestDrone`` and ``QuadWaypointForest``, which take the remaining
    keyword arguments). A drone stops when it reaches its last waypoint (success), crashes
    (into a tree, the ground or another drone) or breaks the task's limits; the episode is
    truncated after ``episode_time`` (60 s). Maps: a pool of ``map_count`` generated 512 m
    maps from ``map_seed``; evaluate on another ``map_seed`` for unseen maps. Wind: 0–3 m/s.
    """

    name = "swarm_waypoint_forest"
    default_episode_time = 60.0

    def __init__(
        self,
        *,
        count: int = 8,
        vehicle: str = "iris_like",
        map: str | dict[str, Any] = "wild",
        map_seed: int = 0,
        map_count: int = 16,
        physics_hz: int | None = None,
        policy_hz: int | None = None,
        episode_time: float | None = None,
        wind: tuple[float, float] | None = (0.0, 3.0),
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
        self.agent = SwarmForestDrone(vehicle=vehicle, **agent_kwargs)

    def teams(self) -> dict[str, Team]:
        return {"drones": Team(self.agent, self.count)}


MULTI_TASKS["swarm_waypoint_forest"] = SwarmWaypointForest
