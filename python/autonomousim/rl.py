"""Framework-free helpers for training scripts: running statistics, observation and reward
normalisation, and a vectorised policy evaluation (numpy only, no torch)."""

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode

from autonomousim.scenario import STATE


class RunningMeanStd:
    """Running mean and variance over the first axis (parallel algorithm of Chan et al.)."""

    def __init__(self, shape: tuple[int, ...] = ()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        n = x.shape[0]
        if n == 0:
            return
        mean, var = x.mean(axis=0), x.var(axis=0)
        delta = mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        self.var = (self.var * self.count + var * n + delta**2 * self.count * n / total) / total
        self.count = total

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.mean, self.var, self.count = np.asarray(d["mean"]), np.asarray(d["var"]), d["count"]


class ObsNormalizer:
    """``(obs − mean) / std`` clipped to ±``clip``, as float32."""

    def __init__(self, dim: int, clip: float = 10.0):
        self.rms = RunningMeanStd((dim,))
        self.clip = clip

    def __call__(self, obs: np.ndarray, update: bool = False) -> np.ndarray:
        if update:
            self.rms.update(obs)
        return np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8), -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {**self.rms.state_dict(), "clip": self.clip}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.rms.load_state_dict(d)
        self.clip = d.get("clip", self.clip)


class RewardScaler:
    """Divides rewards by the running standard deviation of the discounted return (as
    Gymnasium's ``NormalizeReward``). With ``mask``, only those returns update the statistics
    (e.g. the active agents of a multi-agent environment)."""

    def __init__(self, num_envs: int, gamma: float):
        self.rms = RunningMeanStd()
        self.ret = np.zeros(num_envs)
        self.gamma = gamma

    def __call__(self, reward: np.ndarray, done: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        self.ret = self.ret * self.gamma + reward
        self.rms.update(self.ret if mask is None else self.ret[mask])
        self.ret[done] = 0.0
        return reward / np.sqrt(self.rms.var + 1e-8)


def evaluate(
    policy: Callable[[np.ndarray], np.ndarray],
    env_id: str,
    episodes: int = 64,
    seed: int = 1000,
    env_kwargs: dict[str, Any] | None = None,
    num_threads: int = 4,
) -> dict[str, float]:
    """One episode in each of ``episodes`` worlds (seeds ``seed + i``): mean return and
    length, the fraction that lasted until truncation and the median final distance to the
    goal of those. For tasks with a success condition also the fractions that succeeded and
    failed (crashed or broke a task limit) and the mean number of goals reached."""
    envs = gym.make_vec(
        env_id, num_envs=episodes, autoreset_mode=AutoresetMode.DISABLED, num_threads=num_threads, **(env_kwargs or {})
    )
    task = envs.unwrapped.task
    obs, _ = envs.reset(seed=seed)
    ret, length = np.zeros(episodes), np.zeros(episodes, np.int64)
    done, survived = np.zeros(episodes, bool), np.zeros(episodes, bool)
    success, failed = np.zeros(episodes, bool), np.zeros(episodes, bool)
    final_error, goals = np.full(episodes, np.nan), np.zeros(episodes)
    while not done.all():
        obs, reward, terminated, truncated, _ = envs.step(policy(obs))
        live = ~done
        ret[live] += reward[live]
        length[live] += 1
        ended = live & (terminated | truncated)
        survived |= ended & truncated
        success |= ended & task.success
        failed |= ended & terminated & ~task.success
        state = envs.unwrapped.state
        err = np.linalg.norm(state[:, STATE["goal"]] - state[:, STATE["position"]], axis=1)
        final_error[ended] = err[ended]
        goals[ended] = state[ended, STATE["goal_index"]][:, 0]
        done |= ended
    envs.close()
    result = {
        "return": float(ret.mean()),
        "length": float(length.mean()),
        "survived": float(survived.mean()),
        "final_error_m": float(np.median(final_error[survived])) if survived.any() else float("nan"),
    }
    if task.has_success:
        result.update(success=float(success.mean()), failed=float(failed.mean()), goals=float(goals.mean()))
    return result
