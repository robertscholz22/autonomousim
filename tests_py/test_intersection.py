"""IntersectionMulti-v0: cars crossing one junction, and its scripted baseline."""

import numpy as np

import autonomousim
from autonomousim import STATE, MultiAgentVectorEnv
from autonomousim.tasks.intersection import IntersectionMulti, scripted_baseline
from autonomousim.tasks.multi import make_multi_task

URBAN = {"type": "urban", "seed": 2, "count": 2}


def test_intersection_spawns_on_different_arms_of_one_junction():
    n, count = 4, 3
    envs = MultiAgentVectorEnv(n, "intersection_multi", count=count, map=URBAN, num_threads=2)
    assert envs.groups == ["cars"]
    assert envs.single_observation_spaces["cars"].shape == (81,)
    assert envs.single_action_spaces["cars"].shape == (2,)
    obs, _ = envs.reset(seed=0)
    assert obs["cars"].shape == (n, count, 81) and np.isfinite(obs["cars"]).all()
    state = envs.sim.state("cars").reshape(n, count, -1)
    pos = state[..., STATE["position"]][..., :2]
    goal = state[..., STATE["goal"]][..., :2]
    for e in range(n):
        # Apart, each a few tens of metres from its goal, and all around one place.
        gaps = np.linalg.norm(pos[e, :, None] - pos[e, None], axis=-1)[np.triu_indices(count, 1)]
        assert (gaps > 5.0).all(), gaps
        d = np.linalg.norm(goal[e] - pos[e], axis=1)
        assert ((d > 15.0) & (d < 120.0)).all(), d
        centre = pos[e].mean(axis=0)
        assert (np.linalg.norm(pos[e] - centre, axis=1) < 80.0).all()
    # The route term follows the lane ahead: the first point is ahead and near the centre line.
    route = obs["cars"][..., 3:11] / 0.05
    assert (route[..., 0] > 3.0).all() and (np.abs(route[..., 1]) < 2.0).all(), route[..., :2]
    obs, reward, terminated, truncated, info = envs.step({"cars": np.zeros((n, count, 2), np.float32)})
    assert reward["cars"].shape == (n, count) and np.isfinite(reward["cars"]).all()
    envs.close()


def test_intersection_npcs_are_scripted():
    task = make_multi_task("intersection_multi", npcs=5)
    sc = task.scenario()
    cars, traffic = sc["groups"]
    assert cars["goals"]["kind"] == "junction" and cars["action_mode"] == "vk"
    assert traffic["driver"] == {"type": "traffic"} and traffic["count"] == 5
    scripted = task.scripted_scenario()["groups"][0]
    assert scripted["driver"]["type"] == "traffic"
    assert "driver" not in task.scenario()["groups"][0]


def test_intersection_rewards():
    task = IntersectionMulti().agent
    task.bind(3, 0.05, 2)
    state = np.zeros((3, autonomousim.STATE_DIM))
    state[:, STATE["goal"]] = [20.0, 0.0, 0.0]
    task.reset(None, state)
    state[:, STATE["position"]] = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [18.0, 0.0, 0.0]]
    state[1, STATE["road"]] = [0.5, 0.0, 0.0]
    events = np.array([0, int(autonomousim.Event.CRASH_AGENT), int(autonomousim.Event.FINISHED)], np.uint32)
    action = np.zeros((3, 2), np.float32)
    r = task.reward(state, action, action, events)
    np.testing.assert_allclose(r, [1.0 - 0.02, -0.02 - 0.05 * 0.5 - 10.0, 18.0 - 0.02 + 20.0])
    assert task.succeeded(state, events).tolist() == [False, False, True]


def test_scripted_baseline_crosses_without_crashes():
    result = scripted_baseline(IntersectionMulti(map=URBAN), num_envs=8, seed=0, num_threads=4)
    # The traffic drivers give way to each other and reach (nearly) every exit.
    assert result["success"] >= 0.85 and result["crashed"] == 0.0, result
    assert 2.0 < result["time"] < 30.0, result
