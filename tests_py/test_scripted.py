"""Scripted groups (``driver``): cars driving the roads of rural maps take no actions and are
absent from the arrays of the single, vector, multi-agent and PettingZoo environments."""

import json

import numpy as np
import pytest

from autonomousim import STATE_DIM, _native
from autonomousim.env import AutonomousimEnv
from autonomousim.multiagent import MultiAgentVectorEnv
from autonomousim.pettingzoo import parallel_env
from autonomousim.tasks import QuadHover
from autonomousim.tasks.multi import MultiAgentTask, Team
from autonomousim.vector_env import AutonomousimVectorEnv, learning_group

CARS = {
    "name": "cars",
    "count": 2,
    "vehicle": "sedan_like",
    "spawn": {"on_road": True, "min_separation": 40.0},
    "driver": {"type": "road"},
}
RURAL = {"type": "rural", "seed": 3, "count": 1, "cache": False}


def hover(**kw):
    """Drone hovering among the cars (the cars follow the drone's group)."""
    return QuadHover(map=RURAL, overrides={"groups": [{}, CARS]}, **kw)


def test_native_group_info_and_step():
    sim = _native.BatchSim(json.dumps(hover().scenario()), 2, 0, 1)
    assert sim.num_groups == 2
    drones, cars = sim.group_info(0), sim.group_info(1)
    assert not drones["scripted"] and drones["driver"] is None
    assert cars["scripted"] and cars["driver"] == "road" and cars["first_agent"] == 1
    start = sim.state(1)[..., :3].copy()
    for _ in range(250):
        sim.step(np.zeros((2, 1, 4), np.float32))  # one array: the drones
    moved = np.linalg.norm(sim.state(1)[..., :3] - start, axis=-1)
    assert (moved > 5.0).all()
    assert sim.state(1).shape == (2, 2, STATE_DIM)
    with pytest.raises(ValueError, match="one per learning group"):
        sim.step([np.zeros((2, 1, 4), np.float32)] * 2)


def test_single_and_vector_envs():
    env = AutonomousimEnv(hover())
    assert env.group == 0 and env.action_space.shape == (4,)
    obs, _ = env.reset(seed=1)
    for _ in range(5):
        obs, *_ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    env.close()

    envs = AutonomousimVectorEnv(3, hover())
    obs, _ = envs.reset(seed=0)
    assert obs.shape == (3, envs.obs_dim)
    obs, reward, *_ = envs.step(np.zeros((3, 4), np.float32))
    assert reward.shape == (3,)
    envs.close()


def test_multi_agent_and_pettingzoo():
    task = MultiAgentTask(
        teams={"drones": Team(QuadHover(), 2)},
        map=RURAL,
        overrides={"groups": [{"spawn": {"min_separation": 3.0}}, CARS]},
        episode_time=1.0,
    )
    envs = MultiAgentVectorEnv(2, task, seed=0)
    assert envs.groups == ["drones"] and envs.scripted == ["cars"]
    obs, _ = envs.reset(seed=0)
    assert set(obs) == {"drones"}
    obs, reward, terminated, truncated, info = envs.step(np.zeros((2, 2, 4), np.float32))
    assert set(reward) == set(terminated) == set(info["events"]) == {"drones"}
    envs.close()

    env = parallel_env(task)
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"drones_0", "drones_1"}
    env.close()


def test_a_task_needs_one_learning_agent():
    only_cars = {**hover().scenario(), "groups": [CARS]}
    sim = _native.BatchSim(json.dumps(only_cars), 1, 0, 1)
    sim.step([])  # nothing to act on
    with pytest.raises(ValueError, match="one learning group"):
        learning_group(sim)
