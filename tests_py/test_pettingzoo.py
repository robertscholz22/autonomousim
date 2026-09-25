"""PettingZoo ParallelEnv: the API and seed tests for one group and a mixed team."""

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test, parallel_seed_test

from autonomousim.pettingzoo import parallel_env
from autonomousim.tasks import CarWaypointOffroad, QuadHover
from autonomousim.tasks.multi import MultiAgentTask, Team


def hover_team():
    return MultiAgentTask(
        teams={"drones": Team(QuadHover(action_mode="velocity"), 4)},
        overrides={"groups": [{"spawn": {"min_separation": 3.0}}]},
    )


def mixed_team():
    return MultiAgentTask(
        teams={"drones": Team(QuadHover(action_mode="velocity"), 2), "car": Team(CarWaypointOffroad(), 1)},
        episode_time=4.0,
    )


TASKS = {"single_group": hover_team, "mixed_team": mixed_team}


@pytest.mark.parametrize("name", TASKS)
def test_parallel_api(name):
    env = parallel_env(TASKS[name]())
    parallel_api_test(env, num_cycles=1000)
    env.close()


@pytest.mark.parametrize("name", TASKS)
def test_parallel_seed(name):
    parallel_seed_test(lambda: parallel_env(TASKS[name]()), num_cycles=200)


def test_agents_leave_when_they_stop():
    env = parallel_env(hover_team())
    obs, infos = env.reset(seed=1)
    assert env.agents == ["drones_0", "drones_1", "drones_2", "drones_3"] and set(obs) == set(env.agents)
    assert env.state().shape == (4, 21)
    fly_off = np.array([1.0, 0.0, 0.0, 0.0], np.float32)  # 5 m/s: leaves its goal box
    stopped = None
    for _ in range(200):
        actions = {a: fly_off if a == "drones_2" else np.zeros(4, np.float32) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        if terms.get("drones_2"):
            stopped = obs
            break
    assert stopped is not None and "drones_2" not in env.agents and len(env.agents) == 3
    obs, rewards, terms, truncs, infos = env.step({a: np.zeros(4, np.float32) for a in env.agents})
    assert set(obs) == set(rewards) == set(infos) == {"drones_0", "drones_1", "drones_3"}
    env.close()


def test_time_limit_truncates_everyone():
    env = parallel_env(mixed_team())
    env.reset(seed=0)
    steps = 0
    while env.agents:
        live = env.agents
        _, _, terms, truncs, _ = env.step({a: np.zeros(env.action_space(a).shape, np.float32) for a in live})
        steps += 1
    assert steps == 200  # 4 s at 50 Hz
    assert all(truncs[a] for a in live) and not any(terms.values())
    with pytest.raises(RuntimeError):
        env.step({})
    env.close()
