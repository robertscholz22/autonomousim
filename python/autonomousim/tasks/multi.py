"""Tasks with several agents per world: teams of agents, each team driven by a single-agent
task.

A team is an agent group of the scenario. Its task (any ``Task``) supplies the vehicle, the
action mode, the group entries (spawn, goals, sensors, observations) and the per-agent
reward, failure and success rules. The team task is bound over ``num_envs × count`` slots
and sees the state rows of its agents flattened to ``[num_envs·count, STATE_DIM]``, so the
single-agent reward code runs unchanged. Scenario-level settings (map, rates, episode time,
wind) come from the ``MultiAgentTask``; the teams' ``settings()`` are merged in team order.

Episodes: all agents of a world share one episode. An agent stops on a terminal event, one
of its task's failure events or ``failed``, or ``succeeded``; the environment then disables
it (frozen, out of contacts and sensors). The world's episode ends when every agent has
stopped (terminated) or after ``episode_time`` (truncated, for the agents still going).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from autonomousim.scenario import deep_merge
from autonomousim.tasks.base import Task, map_source


@dataclass
class Team:
    """``count`` agents driven by ``task``."""

    task: Task
    count: int = 1


class MultiAgentTask:
    """Teams of agents in one world. Subclasses define ``teams()`` (or pass ``teams``).

    Keyword arguments:
        teams: ``{name: Team}`` in group order (overrides ``teams()``).
        map, map_seed, map_count: see ``autonomousim.tasks.map_source``.
        physics_hz, policy_hz: as for ``Task``; by default those of the first team's task.
        episode_time: seconds until truncation (default ``default_episode_time``).
        wind: ``[min, max]`` mean wind speed per episode (m/s), or None.
        overrides: dict deep-merged into the final scenario.
    """

    name = "multi"
    default_episode_time = 20.0

    def __init__(
        self,
        *,
        teams: dict[str, Team] | None = None,
        map: str | dict[str, Any] = "flat",
        map_seed: int = 0,
        map_count: int = 16,
        physics_hz: int | None = None,
        policy_hz: int | None = None,
        episode_time: float | None = None,
        wind: tuple[float, float] | None = None,
        overrides: dict[str, Any] | None = None,
    ):
        self._teams = teams
        self.map = map
        self.map_seed = map_seed
        self.map_count = map_count
        self.physics_hz = physics_hz
        self.policy_hz = policy_hz
        self.episode_time = self.default_episode_time if episode_time is None else episode_time
        self.wind = wind
        self.overrides = overrides or {}
        self.num_envs = 0
        self.max_steps = 0

    def teams(self) -> dict[str, Team]:
        """The teams in group order."""
        if self._teams is None:
            raise NotImplementedError("pass teams= or override teams()")
        return self._teams

    @property
    def has_success(self) -> bool:
        return any(t.task.has_success for t in self.team_map.values())

    @property
    def team_map(self) -> dict[str, Team]:
        if not hasattr(self, "_team_cache"):
            teams = self.teams()
            if not teams:
                raise ValueError("a multi-agent task needs at least one team")
            for name, t in teams.items():
                if t.count < 1:
                    raise ValueError(f"team {name!r} needs at least one agent")
            self._team_cache = teams
        return self._team_cache

    # ------------------------------------------------------------------ scenario

    def scenario(self) -> dict[str, Any]:
        """The complete scenario dict passed to ``BatchSim``."""
        groups = []
        settings: dict[str, Any] = {}
        for name, team in self.team_map.items():
            t = team.task
            group = {
                "name": name,
                "count": team.count,
                "vehicle": t.vehicle,
                "action_mode": t.action_mode,
                "randomize": t.randomize,
                **t.group(),
            }
            if t.obs is not None:
                group["obs"] = t.obs
            groups.append(group)
            settings = deep_merge(settings, t.settings())
        first = next(iter(self.team_map.values())).task
        sc: dict[str, Any] = {
            "name": self.name,
            "physics_hz": self.physics_hz or first.physics_hz or 0,
            "policy_hz": self.policy_hz or first.policy_hz,
            "map": map_source(self.map, self.map_seed, self.map_count),
            "groups": groups,
        }
        if self.wind is not None:
            sc["randomize_environment"] = {"wind_speed": list(self.wind)}
        return deep_merge(deep_merge(sc, settings), self.overrides)

    # ------------------------------------------------------------------ episodes

    def bind(self, num_envs: int, policy_dt: float, act_dims: dict[str, int]) -> None:
        """Allocate buffers once the simulation exists (team tasks over all their slots)."""
        self.num_envs = num_envs
        self.policy_dt = policy_dt
        self.max_steps = max(1, round(self.episode_time / policy_dt))
        self.steps = np.zeros(num_envs, dtype=np.int64)
        for name, team in self.team_map.items():
            team.task.bind(num_envs * team.count, policy_dt, act_dims[name])
        self.success = {name: np.zeros((num_envs, t.count), dtype=bool) for name, t in self.team_map.items()}

    def reset(self, mask: np.ndarray | None, state: dict[str, np.ndarray]) -> None:
        """Start new episodes in the worlds of ``mask`` (all if None); ``state[g]`` holds the
        first state rows of the new episodes, ``[num_envs, count, STATE_DIM]``."""
        m = slice(None) if mask is None else mask
        self.steps[m] = 0
        for name, team in self.team_map.items():
            slots = None if mask is None else np.repeat(mask, team.count)
            team.task.reset(slots, _flat(state[name]))
            self.success[name][m] = False

    def compute(
        self, state: dict[str, np.ndarray], events: dict[str, np.ndarray], actions: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        """Per group: reward ``float64 [num_envs, count]`` and terminated (the agent stops on
        this step); per world: whether the time limit is reached. ``self.success[g]`` marks
        the agents that succeeded on this step. Rewards and ends of agents that had already
        stopped are computed too; the environment masks them."""
        self.steps += 1
        reward, terminated = {}, {}
        for name, team in self.team_map.items():
            shape = (self.num_envs, team.count)
            a = actions[name].reshape(shape[0] * shape[1], -1)
            r, term, _ = team.task.compute(_flat(state[name]), events[name].reshape(-1), a)
            reward[name] = r.reshape(shape)
            terminated[name] = term.reshape(shape)
            self.success[name][:] = team.task.success.reshape(shape)
        return reward, terminated, self.steps >= self.max_steps


def _flat(state: np.ndarray) -> np.ndarray:
    return state.reshape(-1, state.shape[-1])


#: Registered multi-agent tasks by name.
MULTI_TASKS: dict[str, type[MultiAgentTask]] = {}


def make_multi_task(task: str | MultiAgentTask, **kwargs: Any) -> MultiAgentTask:
    """A multi-agent task from its registered name and keyword arguments, or the instance
    itself."""
    if isinstance(task, MultiAgentTask):
        if kwargs:
            raise TypeError("keyword arguments are only accepted with a task name")
        return task
    try:
        cls = MULTI_TASKS[task]
    except KeyError:
        raise ValueError(f"unknown multi-agent task {task!r}; available: {sorted(MULTI_TASKS)}") from None
    return cls(**kwargs)
