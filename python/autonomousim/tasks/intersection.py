"""IntersectionMulti-v0: several learning cars cross one unsignalized junction or roundabout
of a generated city to their exits, optionally among NPC traffic."""

import json
from typing import Any

import numpy as np

from autonomousim._native import TERMINAL_EVENTS, BatchSim
from autonomousim.events import Event
from autonomousim.scenario import STATE, deep_merge
from autonomousim.tasks.base import Task
from autonomousim.tasks.multi import MULTI_TASKS, MultiAgentTask, Team

FINISHED = int(Event.FINISHED)
CRASH_AGENT = int(Event.CRASH_AGENT)
STUCK = int(Event.STUCK)

#: Kinds of junction crossed by default.
KINDS = ("stop", "yield", "uncontrolled", "roundabout")


class JunctionCar(Task):
    """One car of ``IntersectionMulti``: a ``sedan_like`` in ``vk`` mode (speed up to
    ``max_speed``) spawned on an entry of the episode's junction, ``distance`` m before the
    entry's end, to cross it to a goal ``exit`` m along a lane leaving it on another road
    (``junction`` goals: every car of the world on a different arm of the same junction).
    The car keeps its lane-level route through the junction for the ``route`` and ``road``
    terms.

    Observation (73 values): the goal in the heading frame (scaled 1/20), the ``route`` term
    (scaled 1/20), the ``road`` term, speed, steering angle, last action and the ``traffic``
    term (the ``traffic`` nearest other cars within 40 m: 13 values each).

    Reward per step (``d``: horizontal distance to the goal):

    - ``progress_weight·(d_before − d_after)``;
    - ``−time_weight`` (every step, so waiting costs);
    - ``−lane_weight·|lateral offset|``;
    - ``−smooth_weight·‖Δa‖²``;
    - ``finish_bonus`` at the goal (within ``goal_radius``: success);
    - ``−terminal_penalty`` on a crash (into another car, a building or a pole), rollover,
      leaving the map or getting stuck (moving less than 0.5 m in ``stuck_time`` s), and
      ``−crash_penalty`` more on touching another car.
    """

    name = "junction_car"
    default_episode_time = 40.0
    has_success = True
    failure_events = STUCK

    def __init__(
        self,
        *,
        kinds: tuple[str, ...] = KINDS,
        distance: tuple[float, float] = (15.0, 35.0),
        exit: float = 20.0,
        goal_radius: float = 3.0,
        max_speed: float = 10.0,
        stuck_time: float = 15.0,
        traffic: int = 4,
        progress_weight: float = 1.0,
        time_weight: float = 0.02,
        lane_weight: float = 0.05,
        smooth_weight: float = 0.02,
        finish_bonus: float = 20.0,
        crash_penalty: float = 10.0,
        terminal_penalty: float = 20.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("vehicle", "sedan_like")
        kwargs.setdefault("action_mode", "vk")
        kwargs.setdefault("map", "urban")
        kwargs.setdefault("policy_hz", 20)
        super().__init__(**kwargs)
        self.kinds = tuple(kinds)
        self.distance = distance
        self.exit = exit
        self.goal_radius = goal_radius
        self.max_speed = max_speed
        self.stuck_time = stuck_time
        self.traffic = traffic
        self.progress_weight = progress_weight
        self.time_weight = time_weight
        self.lane_weight = lane_weight
        self.smooth_weight = smooth_weight
        self.finish_bonus = finish_bonus
        self.crash_penalty = crash_penalty
        self.terminal_penalty = terminal_penalty

    def group(self) -> dict[str, Any]:
        return {
            "goals": {
                "kind": "junction",
                "distance": list(self.distance),
                "radius": self.goal_radius,
                "junction": {"kinds": list(self.kinds), "exit": self.exit},
            },
            "ground_action_limits": {"speed": self.max_speed},
            "obs": [
                {"term": "goal_rel_heading", "scale": 0.05},
                {"term": "route", "scale": 0.05},
                {"term": "road"},
                {"term": "speed", "scale": 0.1},
                {"term": "steering"},
                {"term": "last_action"},
                {"term": "traffic", "count": self.traffic, "range": 40.0, "scale": 0.1},
            ],
        }

    def settings(self) -> dict[str, Any]:
        return {"events": {"bounds_margin": 5.0, "ground": {"stuck_time": self.stuck_time}}}

    # ------------------------------------------------------------------ episodes

    def bind(self, num_envs: int, policy_dt: float, act_dim: int) -> None:
        super().bind(num_envs, policy_dt, act_dim)
        self.prev_position = np.zeros((num_envs, 2))

    def reset(self, mask: np.ndarray | None = None, state: np.ndarray | None = None) -> None:
        super().reset(mask, state)
        if state is not None:
            m = slice(None) if mask is None else mask
            self.prev_position[m] = state[m, STATE["position"]][:, :2]

    def succeeded(self, state: np.ndarray, events: np.ndarray) -> np.ndarray:
        return (events & FINISHED) != 0

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        position = state[:, STATE["position"]][:, :2]
        goal = state[:, STATE["goal"]][:, :2]
        before = np.linalg.norm(goal - self.prev_position, axis=1)
        after = np.linalg.norm(goal - position, axis=1)
        self.prev_position[:] = position
        da = action - prev_action
        return (
            self.progress_weight * (before - after)
            - self.time_weight
            - self.lane_weight * np.abs(state[:, STATE["road"]][:, 0])
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
            + self.finish_bonus * ((events & FINISHED) != 0)
            - self.crash_penalty * ((events & CRASH_AGENT) != 0)
        )


class IntersectionMulti(MultiAgentTask):
    """IntersectionMulti-v0: ``count`` learning cars (default 3, one group ``cars`` sharing a
    policy; at most the junction's arms) start on different entries of one stop, yield or
    uncontrolled junction or roundabout of a generated city (``urban``: a pool of
    ``map_count`` 512 m maps from ``map_seed``; evaluate on another ``map_seed`` for unseen
    maps) and cross it to their exits (see ``JunctionCar``, which takes the remaining keyword
    arguments). With ``npcs`` > 0, that many traffic cars (scripted, group ``traffic``) drive
    the city's lanes too, spawned on random lanes. A car stops at its goal (success), on a
    crash or when stuck; the episode is truncated after ``episode_time`` (40 s).

    ``scripted_scenario()`` is the same scenario with the traffic driver flying the cars along
    their routes through the junction: the scripted baseline.
    """

    name = "intersection_multi"
    default_episode_time = 40.0

    def __init__(
        self,
        *,
        count: int = 3,
        npcs: int = 0,
        map: str | dict[str, Any] = "urban",
        map_seed: int = 0,
        map_count: int = 16,
        physics_hz: int | None = None,
        policy_hz: int | None = None,
        episode_time: float | None = None,
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
            overrides=overrides,
        )
        self.count = count
        self.npcs = npcs
        self.agent = JunctionCar(**agent_kwargs)

    def teams(self) -> dict[str, Team]:
        return {"cars": Team(self.agent, self.count)}

    def scenario(self) -> dict[str, Any]:
        sc = super().scenario()
        if self.npcs > 0:
            npc = {
                "name": "traffic",
                "count": self.npcs,
                "vehicle": self.agent.vehicle,
                "driver": {"type": "traffic"},
                "spawn": {"on_road": True, "min_separation": 12.0},
            }
            sc = deep_merge(sc, {"groups": [*sc["groups"], npc]})
        return sc

    def scripted_scenario(self) -> dict[str, Any]:
        """The scenario with the traffic driver flying the cars (no respawns)."""
        sc = self.scenario()
        sc["groups"][0] = {**sc["groups"][0], "driver": {"type": "traffic", "respawn": 0.0}}
        return sc


def scripted_baseline(task: IntersectionMulti, num_envs: int, seed: int = 0, num_threads: int = 0) -> dict[str, float]:
    """One episode per world of ``task.scripted_scenario()``: the share of cars that reached
    their goal before any crash, the share that crashed (a terminal event) first and the
    mean time to the goal of those that reached it (s)."""
    sim = BatchSim(json.dumps(task.scripted_scenario()), num_envs, seed, num_threads)
    sim.reset()
    shape = (num_envs, task.count)
    done = np.zeros(shape, dtype=bool)
    success = np.zeros(shape, dtype=bool)
    crashed = np.zeros(shape, dtype=bool)
    time = np.zeros(shape)
    steps = max(1, round(task.episode_time / sim.policy_dt))
    for k in range(steps):
        sim.step([])
        ev = sim.events("cars").reshape(shape)
        crash = ~done & ((ev & (TERMINAL_EVENTS | CRASH_AGENT)) != 0)
        reached = ~done & ~crash & ((ev & FINISHED) != 0)
        crashed |= crash
        success |= reached
        time[reached] = (k + 1) * sim.policy_dt
        done |= crash | reached
        if done.all():
            break
    return {
        "success": float(success.mean()),
        "crashed": float(crashed.mean()),
        "time": float(time[success].mean()) if success.any() else float("nan"),
    }


MULTI_TASKS["intersection_multi"] = IntersectionMulti
