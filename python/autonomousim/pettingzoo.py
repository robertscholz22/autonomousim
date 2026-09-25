"""PettingZoo ``ParallelEnv`` over one world of a multi-agent task.

```python
from autonomousim.pettingzoo import parallel_env
from autonomousim.tasks import QuadHover
from autonomousim.tasks.multi import MultiAgentTask, Team

env = parallel_env(MultiAgentTask(teams={"drones": Team(QuadHover(action_mode="velocity"), 4)}))
obs, infos = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
```

Agents are named ``"<group>_<k>"``. An agent that stops (see ``autonomousim.tasks.multi``)
reports ``terminations[agent] = True`` once and then leaves ``env.agents``. At the time
limit, every agent still going is truncated. The episode is over when ``env.agents`` is
empty; call ``reset`` for the next one. ``infos[agent]["events"]`` holds the agent's event
bits of the step (``autonomousim.events``). For training many worlds at once, use
``MultiAgentVectorEnv`` directly.
"""

import functools
from typing import Any

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv as _ParallelEnv

from autonomousim.multiagent import MultiAgentVectorEnv
from autonomousim.tasks.multi import MultiAgentTask, make_multi_task


class AutonomousimParallelEnv(_ParallelEnv):
    """One world of a multi-agent task (a ``MultiAgentTask`` or a registered name with its
    keyword arguments); ``seed`` is the base seed before the first ``reset``."""

    metadata = {"name": "autonomousim_v0", "render_modes": [], "is_parallelizable": True}

    def __init__(
        self,
        task: str | MultiAgentTask,
        *,
        seed: int = 0,
        num_threads: int = 1,
        render_mode: str | None = None,
        **task_kwargs: Any,
    ):
        if render_mode is not None:
            raise ValueError("rendering is done by the viewer (autonomousim-viewer)")
        self.render_mode = None
        self._venv = MultiAgentVectorEnv(
            1, make_multi_task(task, **task_kwargs), seed=seed, num_threads=num_threads, autoreset=False
        )
        v = self._venv
        self._slots = {f"{g}_{k}": (g, k) for g in v.groups for k in range(v.count[g])}
        self.possible_agents = list(self._slots)
        self.agents: list[str] = []
        self._obs_spaces = {a: v.single_observation_spaces[g] for a, (g, _) in self._slots.items()}
        self._act_spaces = {
            a: gym.spaces.Box(-1.0, 1.0, (v.act_dim[g],), np.float32) for a, (g, _) in self._slots.items()
        }
        self._actions = {g: np.zeros((1, v.count[g], v.act_dim[g]), np.float32) for g in v.groups}

    @property
    def task(self) -> MultiAgentTask:
        return self._venv.task

    @functools.cache  # noqa: B019 (one space object per agent, as PettingZoo requires)
    def observation_space(self, agent: str) -> gym.spaces.Box:
        return self._obs_spaces[agent]

    @functools.cache  # noqa: B019
    def action_space(self, agent: str) -> gym.spaces.Box:
        return self._act_spaces[agent]

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        obs, _ = self._venv.reset(seed=seed)
        self.agents = self.possible_agents[:]
        return (
            {a: obs[g][0, k] for a, (g, k) in self._slots.items()},
            {a: {} for a in self.agents},
        )

    def step(self, actions: dict[str, np.ndarray]) -> tuple[
        dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]
    ]:
        if not self.agents:
            raise RuntimeError("the episode is over; call reset()")
        for a in self._actions.values():
            a.fill(0.0)
        for agent, act in actions.items():
            g, k = self._slots[agent]
            self._actions[g][0, k] = act
        obs, reward, terminated, truncated, info = self._venv.step(self._actions)
        live = self.agents
        out = ({}, {}, {}, {}, {})
        for a in live:
            g, k = self._slots[a]
            out[0][a] = obs[g][0, k]
            out[1][a] = float(reward[g][0, k])
            out[2][a] = bool(terminated[g][0, k])
            out[3][a] = bool(truncated[0]) and not out[2][a]
            out[4][a] = {"events": int(info["events"][g][0, k])}
        self.agents = [a for a in live if not (out[2][a] or out[3][a])]
        return out

    def state(self) -> np.ndarray:
        """State rows of all agents in ``possible_agents`` order, ``float64 [n, STATE_DIM]``."""
        s = self._venv.state
        return np.concatenate([s[g][0] for g in self._venv.groups])

    def render(self) -> None:
        return None

    def close(self) -> None:
        self._venv.close()


def parallel_env(task: str | MultiAgentTask, **kwargs: Any) -> AutonomousimParallelEnv:
    """A PettingZoo ``ParallelEnv`` for ``task`` (see ``AutonomousimParallelEnv``)."""
    return AutonomousimParallelEnv(task, **kwargs)
