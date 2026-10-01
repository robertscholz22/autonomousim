"""CarParking-v0 and the Reeds–Shepp paths of its scripted pilot."""

import math

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.vector import AutoresetMode

import autonomousim
from autonomousim import STATE, reeds_shepp
from autonomousim.tasks import make_task

URBAN = {"type": "urban", "seed": 2, "count": 2}


def test_reeds_shepp_paths_reach_the_goal():
    rng = np.random.default_rng(0)
    for _ in range(300):
        start = tuple(rng.uniform([-5, -5, -math.pi], [5, 5, math.pi]))
        goal = tuple(rng.uniform([-5, -5, -math.pi], [5, 5, math.pi]))
        radius = rng.uniform(1.0, 6.0)
        paths = reeds_shepp.paths(start, goal, radius)
        # Every pose is reachable; the paths come shortest first and end on the goal.
        assert paths
        lengths = [reeds_shepp.length(p) for p in paths]
        assert lengths == sorted(lengths)
        poses = reeds_shepp.sample(start, paths[0], radius, 0.05)
        assert np.allclose(poses[0, :3], start)
        assert np.hypot(*(poses[-1, :2] - goal[:2])) < 1e-6
        assert abs((poses[-1, 2] - goal[2] + math.pi) % (2 * math.pi) - math.pi) < 1e-6
        # The samples are continuous: at most the step apart.
        assert (np.hypot(*np.diff(poses[:, :2], axis=0).T) < 0.05 + 1e-9).all()
        # No shorter than the straight line.
        assert lengths[0] * radius >= math.dist(start[:2], goal[:2]) - 1e-9
    # Straight back: one reversing segment; a U-turn on the spot's circle: half a turn.
    assert reeds_shepp.paths((0, 0, 0), (-3, 0, 0), 1.0)[0] == [("S", pytest.approx(-3.0))]
    u = reeds_shepp.paths((0, 0, 0), (0, 2, math.pi), 1.0)[0]
    assert len(u) == 1 and u[0][0] == "L" and abs(u[0][1]) == pytest.approx(math.pi)
    assert reeds_shepp.cusps([("L", 1.0), ("S", -1.0), ("R", -1.0), ("L", 1.0)]) == 2


def test_car_parking_env_checks():
    env = gym.make("autonomousim/CarParking-v0", map=URBAN, parked=4)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("kind", ["lot", "street"])
def test_car_parking_scripted_pilot_parks(kind):
    n = 8
    envs = gym.make_vec(
        "autonomousim/CarParking-v0",
        num_envs=n,
        num_threads=4,
        map=URBAN,
        kinds=(kind,),
        autoreset_mode=AutoresetMode.DISABLED,
    )
    task = envs.unwrapped.task
    assert [t[0] for t in envs.unwrapped.obs_layout] == ["trailer_goal", "speed", "steering", "last_action", "lidar_log"]
    assert envs.single_observation_space.shape == (80,)
    obs, _ = envs.reset(seed=0)
    d, psi = task.goal_errors(envs.unwrapped.state)
    # Lot: in the aisle, across the bay; street: in the lane beside, along it.
    turn = np.abs(psi) if kind == "street" else np.abs(np.abs(psi) - np.pi / 2)
    assert (d > 2.0).all() and (d < 14.0).all() and (turn < np.radians(10.5)).all(), (d, psi)
    # The LiDAR sees the parked cars around the bay, or buildings, within its range.
    for _ in range(3):
        obs, *_ = envs.step(np.zeros((n, 2), np.float32))
    lidar = obs[:, 8:]
    assert (lidar < lidar.max(axis=1, keepdims=True) - 0.1).any(axis=1).all(), lidar
    ret, done, success = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
    while not done.all():
        _, reward, terminated, truncated, info = envs.step(task.scripted(envs.unwrapped.state))
        live = ~done
        ret[live] += reward[live]
        success |= live & task.success
        assert not (live & terminated & ~task.success).any(), info["events"]
        done |= terminated | truncated
    # The pilot parks (nearly) every car without touching the parked ones.
    assert success.sum() >= n - 1, success
    assert (ret[success] > task.success_bonus).all(), ret


def test_car_parking_rewards():
    task = make_task("car_parking", map="flat")
    task.bind(3, 0.05, 2)
    state = np.zeros((3, autonomousim.STATE_DIM))
    state[:, STATE["goal"]] = [10.0, 0.0, 0.0]
    state[:, STATE["goal_yaw"]] = 0.0
    state[:, STATE["tail"]] = [[0.0, 0.0, 0.0], [5.0, 0.0, np.pi / 2], [9.8, 0.0, 0.0]]
    task.reset(None, state)
    # 1 m closer; 1 m closer, turned 90° 4 m off; in the bay, aligned and stopped: the bonus.
    state[:, STATE["tail"]] = [[1.0, 0.0, 0.0], [6.0, 0.0, np.pi / 2], [9.8, 0.1, 0.02]]
    action = np.zeros((3, 2), np.float32)
    events = np.zeros(3, np.uint32)
    r = task.reward(state, action, action, events)
    expected = [1.0, 1.0 - 0.05 * (np.pi / 2) * 0.75, 0.2 - np.hypot(0.2, 0.1) - 0.05 * 0.02 + 20.0]
    np.testing.assert_allclose(r, expected)
    assert task.succeeded(state, events).tolist() == [False, False, True]
    # Still rolling, or skewed by 6°: no success.
    state[2, STATE["velocity"]] = [0.5, 0.0, 0.0]
    assert not task.succeeded(state, events)[2]
    state[2, STATE["velocity"]] = 0.0
    state[2, STATE["tail"]] = [9.8, 0.0, np.radians(6.0)]
    assert not task.succeeded(state, events)[2]
    # Far off: a failure.
    state[0, STATE["tail"]] = [-16.0, 0.0, 0.0]
    assert task.failed(state).tolist() == [True, False, False]
    sc = make_task("car_parking").scenario()
    assert sc["map"]["type"] == "urban"
    ego, parked = sc["groups"]
    assert ego["goals"]["kind"] == "bay" and ego["action_mode"] == "vk"
    assert parked["driver"] == {"type": "parked"} and parked["spawn"]["near_bay_goals"]
    assert len(make_task("car_parking", parked=0).scenario()["groups"]) == 1
