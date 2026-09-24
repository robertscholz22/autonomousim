"""Native Gymnasium ``VectorEnv``: all worlds step in Rust in one call.

```python
import gymnasium as gym
import autonomousim  # registers the environments

envs = gym.make_vec("autonomousim/QuadHover-v0", num_envs=256, num_threads=9, seed=0)
obs, info = envs.reset(seed=0)
obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
```

Autoreset (``metadata["autoreset_mode"]``):

- ``SAME_STEP`` (default): a world whose episode ended is reset within the same ``step``.
  The returned observation is the first one of the new episode. The last observation of the
  finished episode is in ``info["final_obs"]``, a dense ``float32 [num_envs, obs_dim]`` array
  (unlike Gymnasium's object array), valid where ``info["_final_obs"]`` is true. Truncated
  episodes can bootstrap from it.
- ``DISABLED``: nothing is reset automatically; call ``reset(options={"reset_mask": mask})``.
  A world whose episode ended keeps stepping; an agent stopped by a terminal event stays
  frozen.

``NEXT_STEP`` is not supported: all worlds step together.

``info`` holds ``events`` (``uint32 [num_envs]``, see ``autonomousim.events``) on every step.
When an episode ends it also holds ``episode = {"r": return, "l": length, "success": ...}``
(float64, int64, bool; success as defined by the task, always false for tasks without one),
valid where ``info["_episode"]`` is true (the layout of Gymnasium's
``RecordEpisodeStatistics``).
"""

import json
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from autonomousim._native import BatchSim
from autonomousim.tasks import Task, make_task


def _seeds(seed: int | list[int] | None, num_envs: int) -> np.ndarray | None:
    """Per-world seeds, Gymnasium style: ``seed + i`` for an int, one per world for a list."""
    if seed is None:
        return None
    if isinstance(seed, (int, np.integer)):
        return np.arange(num_envs, dtype=np.uint64) + np.uint64(seed)
    seeds = np.asarray(seed, dtype=np.uint64)
    if seeds.shape != (num_envs,):
        raise ValueError(f"expected {num_envs} seeds, got {len(seeds)}")
    return seeds


class AutonomousimVectorEnv(VectorEnv):
    """``num_envs`` worlds of a task, one agent each, stepped in parallel on ``num_threads``
    threads (0: one per logical CPU). ``seed`` sets the worlds' base seeds before the first
    ``reset``. Other keyword arguments go to the task (see ``autonomousim.tasks``). With
    ``copy=False``, ``reset`` and ``step`` return a view of the observation buffer. The next
    call overwrites that view."""

    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP, "render_modes": []}

    def __init__(
        self,
        num_envs: int = 1,
        task: str | Task = "hover",
        *,
        seed: int = 0,
        num_threads: int = 0,
        autoreset_mode: AutoresetMode | str = AutoresetMode.SAME_STEP,
        copy: bool = True,
        render_mode: str | None = None,
        **task_kwargs: Any,
    ):
        if render_mode is not None:
            raise ValueError("rendering is done by the viewer (autonomousim-viewer)")
        mode = AutoresetMode(autoreset_mode) if isinstance(autoreset_mode, str) else autoreset_mode
        if mode == AutoresetMode.NEXT_STEP:
            raise ValueError("NEXT_STEP autoreset is not supported; use SAME_STEP or DISABLED")
        self.task = make_task(task, **task_kwargs)
        self.sim = BatchSim(json.dumps(self.task.scenario()), num_envs, seed, num_threads)
        info = self.sim.group_info(0)
        if self.sim.num_groups != 1 or info["count"] != 1:
            raise ValueError("a task scenario must have one group with one agent")
        self.num_envs = num_envs
        self.obs_dim = int(info["obs_dim"])
        self.act_dim = int(info["act_dim"])
        self.obs_layout = info["obs_layout"]
        self.metadata = {**type(self).metadata, "autoreset_mode": mode}
        self.render_mode = None
        self.copy = copy
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        # Views of the native output arrays (overwritten in place by every step and reset).
        self._obs = self.sim.obs(0)[:, 0, :]
        self._state = self.sim.state(0)[:, 0, :]
        self._events = self.sim.events(0)[:, 0]
        self.task.bind(num_envs, self.sim.policy_dt, self.act_dim)
        self._return = np.zeros(num_envs, dtype=np.float64)
        self._length = np.zeros(num_envs, dtype=np.int64)

    # ------------------------------------------------------------------ gymnasium API

    def reset(
        self, *, seed: int | list[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if isinstance(seed, (int, np.integer)):
            super().reset(seed=int(seed))
        elif seed is not None:
            super().reset(seed=int(np.asarray(seed).flat[0]))
        mask = (options or {}).get("reset_mask")
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (self.num_envs,):
                raise ValueError(f"reset_mask must have shape ({self.num_envs},)")
        self.sim.reset(mask, _seeds(seed, self.num_envs))
        self.task.reset(mask, self._state)
        if mask is None:
            self._return[:] = 0.0
            self._length[:] = 0
        else:
            self._return[mask] = 0.0
            self._length[mask] = 0
        return self._observation(), {}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, self.act_dim)
        self.sim.step(actions)
        reward, terminated, truncated = self.task.compute(self._state, self._events, actions)
        self._return += reward
        self._length += 1
        info: dict[str, Any] = {"events": self._events.copy()}
        done = terminated | truncated
        if done.any():
            info["episode"] = {
                "r": np.where(done, self._return, 0.0),
                "l": np.where(done, self._length, 0),
                "success": self.task.success.copy(),
            }
            info["_episode"] = done
            self._return[done] = 0.0
            self._length[done] = 0
            if self.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP:
                info["final_obs"] = self._obs.copy()
                info["_final_obs"] = done
                self.sim.reset(done)
                self.task.reset(done, self._state)
        return self._observation(), reward, terminated, truncated, info

    def close_extras(self, **kwargs: Any) -> None:
        self.sim.close()

    # ------------------------------------------------------------------ helpers

    def _observation(self) -> np.ndarray:
        return self._obs.copy() if self.copy else self._obs

    @property
    def state(self) -> np.ndarray:
        """Current state rows ``float64 [num_envs, STATE_DIM]`` (``autonomousim.scenario.STATE``);
        after an autoreset, those of the new episode. Overwritten in place."""
        return self._state

    def __repr__(self) -> str:
        return f"AutonomousimVectorEnv(task={self.task.name!r}, num_envs={self.num_envs})"
