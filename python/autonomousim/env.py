"""Single-environment Gymnasium adapter (``gym.make``), e.g. for ``check_env`` and tools that
expect a ``gym.Env``. Training should use the native vector environment (``gym.make_vec``)."""

import json
from typing import Any

import gymnasium as gym
import numpy as np

from autonomousim._native import BatchSim
from autonomousim.tasks import Task, make_task


class AutonomousimEnv(gym.Env):
    """One world of a task. ``reset(seed=s)`` starts the same episode as world 0 of the vector
    environment after ``reset(seed=s)``. Keyword arguments go to the task."""

    metadata = {"render_modes": []}

    def __init__(
        self, task: str | Task = "hover", *, render_mode: str | None = None, num_threads: int = 1, **task_kwargs: Any
    ):
        if render_mode is not None:
            raise ValueError("rendering is done by the viewer (autonomousim-viewer)")
        self.task = make_task(task, **task_kwargs)
        self.sim = BatchSim(json.dumps(self.task.scenario()), 1, 0, num_threads)
        info = self.sim.group_info(0)
        if self.sim.num_groups != 1 or info["count"] != 1:
            raise ValueError("a task scenario must have one group with one agent")
        self.obs_dim = int(info["obs_dim"])
        self.act_dim = int(info["act_dim"])
        self.obs_layout = info["obs_layout"]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
        self._obs = self.sim.obs(0)[0, 0]
        self._state = self.sim.state(0)[:, 0, :]
        self._events = self.sim.events(0)[:, 0]
        self.task.bind(1, self.sim.policy_dt, self.act_dim)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.sim.reset(None, None if seed is None else [seed])
        self.task.reset(None, self._state)
        return self._obs.copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).reshape(1, self.act_dim)
        self.sim.step(a)
        reward, terminated, truncated = self.task.compute(self._state, self._events, a)
        info = {"events": int(self._events[0]), "success": bool(self.task.success[0])}
        return self._obs.copy(), float(reward[0]), bool(terminated[0]), bool(truncated[0]), info

    def close(self) -> None:
        self.sim.close()

    @property
    def state(self) -> np.ndarray:
        """Current state row ``float64 [STATE_DIM]`` (``autonomousim.scenario.STATE``)."""
        return self._state[0]
