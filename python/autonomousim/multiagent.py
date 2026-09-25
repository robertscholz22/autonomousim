"""Native multi-agent vector environment: ``num_envs`` worlds with several agent groups,
stepped in Rust in one call.

```python
from autonomousim.multiagent import MultiAgentVectorEnv
from autonomousim.tasks import QuadHover
from autonomousim.tasks.multi import MultiAgentTask, Team

task = MultiAgentTask(teams={"drones": Team(QuadHover(), 4)})
envs = MultiAgentVectorEnv(64, task, seed=0)
obs, info = envs.reset()                                    # obs["drones"]: [64, 4, obs_dim]
actions = {g: envs.action_space(g).sample() for g in envs.groups}
obs, reward, terminated, truncated, info = envs.step(actions)
```

Arrays are per group, keyed by group name: ``obs[g]`` ``float32 [num_envs, count, obs_dim]``,
``reward[g]`` ``float64 [num_envs, count]``, ``terminated[g]`` ``bool [num_envs, count]``;
``truncated`` is per world, ``bool [num_envs]``. ``step`` takes ``{group: [num_envs, count,
act_dim]}``, or one array when there is a single group.

Agents stop individually (see ``autonomousim.tasks.multi``): ``terminated[g]`` is true on
the step an agent stops, and from then on it is frozen, its reward is 0 and its
``terminated`` stays false. ``info["active"][g]`` marks the agents that were still going
before the step, i.e. the (world, agent) slots whose reward and ends count; train on those
only. A world's episode ends when all its agents have stopped or at the time limit
(``truncated``, which applies to the agents still going).

Autoreset is per world, ``SAME_STEP``: a world whose episode ended is reset within the same
``step``. ``info["final_obs"][g]`` holds the last observations of the finished episodes
(dense, valid where ``info["_final_obs"]``, per world, is true), for bootstrapping truncated
agents. ``autoreset=False`` leaves ended worlds alone; call
``reset(options={"reset_mask": mask})``.

``info`` holds ``events`` (per group, ``uint32 [num_envs, count]``) and ``active`` on every
step. When an episode ends it also holds ``episode = {"r": {g: return per agent}, "l":
length per world, "success": {g: per agent}}``, valid where ``info["_episode"]`` is true.
"""

import json
from typing import Any

import gymnasium as gym
import numpy as np

from autonomousim._native import BatchSim
from autonomousim.tasks.multi import MultiAgentTask, make_multi_task
from autonomousim.vector_env import _seeds


class MultiAgentVectorEnv:
    """``num_envs`` worlds of a multi-agent task (an instance, or a registered name with its
    keyword arguments), stepped in parallel on ``num_threads``
    threads (0: one per logical CPU). ``seed`` sets the worlds' base seeds before the first
    ``reset``. With ``copy=False``, ``reset`` and ``step`` return views of the observation
    buffers, which the next call overwrites."""

    def __init__(
        self,
        num_envs: int = 1,
        task: str | MultiAgentTask | None = None,
        *,
        seed: int = 0,
        num_threads: int = 0,
        autoreset: bool = True,
        copy: bool = True,
        **task_kwargs: Any,
    ):
        if task is None:
            raise ValueError("pass a MultiAgentTask or the name of one")
        self.task = make_multi_task(task, **task_kwargs)
        self.sim = BatchSim(json.dumps(self.task.scenario()), num_envs, seed, num_threads)
        self.num_envs = num_envs
        self.groups: list[str] = list(self.sim.group_names)
        self.autoreset = autoreset
        self.copy = copy
        self.count: dict[str, int] = {}
        self.obs_dim: dict[str, int] = {}
        self.act_dim: dict[str, int] = {}
        self.obs_layout: dict[str, Any] = {}
        for g in self.groups:
            info = self.sim.group_info(g)
            self.count[g] = int(info["count"])
            self.obs_dim[g] = int(info["obs_dim"])
            self.act_dim[g] = int(info["act_dim"])
            self.obs_layout[g] = info["obs_layout"]
        self.single_observation_spaces = {
            g: gym.spaces.Box(-np.inf, np.inf, (self.obs_dim[g],), np.float32) for g in self.groups
        }
        self.single_action_spaces = {g: gym.spaces.Box(-1.0, 1.0, (self.act_dim[g],), np.float32) for g in self.groups}
        # Views of the native output arrays (overwritten in place by every step and reset).
        self._obs = {g: self.sim.obs(g) for g in self.groups}
        self._state = {g: self.sim.state(g) for g in self.groups}
        self._events = {g: self.sim.events(g) for g in self.groups}
        self.task.bind(num_envs, self.sim.policy_dt, self.act_dim)
        shape = {g: (num_envs, self.count[g]) for g in self.groups}
        self._active = {g: np.ones(shape[g], dtype=bool) for g in self.groups}
        self._return = {g: np.zeros(shape[g]) for g in self.groups}
        self._success = {g: np.zeros(shape[g], dtype=bool) for g in self.groups}
        self._length = np.zeros(num_envs, dtype=np.int64)
        self._actions = {g: np.zeros((*shape[g], self.act_dim[g]), dtype=np.float32) for g in self.groups}

    # ------------------------------------------------------------------ spaces

    def observation_space(self, group: str) -> gym.spaces.Box:
        """Batched observation space of a group, ``[num_envs, count, obs_dim]``."""
        return gym.spaces.Box(-np.inf, np.inf, (self.num_envs, self.count[group], self.obs_dim[group]), np.float32)

    def action_space(self, group: str) -> gym.spaces.Box:
        """Batched action space of a group, ``[num_envs, count, act_dim]``."""
        return gym.spaces.Box(-1.0, 1.0, (self.num_envs, self.count[group], self.act_dim[group]), np.float32)

    # ------------------------------------------------------------------ API

    def reset(
        self, *, seed: int | list[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        mask = (options or {}).get("reset_mask")
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (self.num_envs,):
                raise ValueError(f"reset_mask must have shape ({self.num_envs},)")
        self.sim.reset(mask, _seeds(seed, self.num_envs))
        self._start(mask)
        return self._observation(), {"active": self._copy(self._active)}

    def step(
        self, actions: dict[str, np.ndarray] | np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
        if not isinstance(actions, dict):
            if len(self.groups) != 1:
                raise TypeError("pass a dict with one action array per group")
            actions = {self.groups[0]: actions}
        if set(actions) != set(self.groups):
            raise ValueError(f"expected actions for the groups {self.groups}, got {sorted(actions)}")
        for g in self.groups:
            self._actions[g][:] = np.asarray(actions[g], dtype=np.float32).reshape(self._actions[g].shape)
        self.sim.step([self._actions[g] for g in self.groups])

        active = self._copy(self._active)
        reward, terminated, time_up = self.task.compute(self._state, self._events, self._actions)
        still = np.zeros(self.num_envs, dtype=bool)
        for g in self.groups:
            # Terminal events have usually disabled the agent in Rust already (idempotent).
            end = terminated[g] & active[g]
            if end.any():
                self.sim.disable(g, end)
            terminated[g] = end
            reward[g] = np.where(active[g], reward[g], 0.0)
            self._active[g] &= ~end
            self._return[g] += reward[g]
            self._success[g] |= self.task.success[g] & active[g]
            still |= self._active[g].any(axis=1)
        self._length += 1
        truncated = time_up & still
        done = ~still | truncated

        info: dict[str, Any] = {"events": self._copy(self._events), "active": active}
        if done.any():
            info["episode"] = {
                "r": {g: np.where(done[:, None], self._return[g], 0.0) for g in self.groups},
                "l": np.where(done, self._length, 0),
                "success": {g: self._success[g] & done[:, None] for g in self.groups},
            }
            info["_episode"] = done
            if self.autoreset:
                info["final_obs"] = self._copy(self._obs)
                info["_final_obs"] = done
                self.sim.reset(done)
                self._start(done)
        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self.sim.close()

    # ------------------------------------------------------------------ helpers

    def _start(self, mask: np.ndarray | None) -> None:
        """Bookkeeping for new episodes in the worlds of ``mask`` (all if None)."""
        m = slice(None) if mask is None else mask
        self.task.reset(mask, self._state)
        for g in self.groups:
            self._active[g][m] = True
            self._return[g][m] = 0.0
            self._success[g][m] = False
        self._length[m] = 0

    def _observation(self) -> dict[str, np.ndarray]:
        return self._copy(self._obs) if self.copy else dict(self._obs)

    @staticmethod
    def _copy(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {g: a.copy() for g, a in arrays.items()}

    @property
    def state(self) -> dict[str, np.ndarray]:
        """Current state rows per group, ``float64 [num_envs, count, STATE_DIM]``; after an
        autoreset, those of the new episodes. Overwritten in place."""
        return self._state

    def __repr__(self) -> str:
        counts = ", ".join(f"{g}={self.count[g]}" for g in self.groups)
        return f"MultiAgentVectorEnv(task={self.task.name!r}, num_envs={self.num_envs}, {counts})"
