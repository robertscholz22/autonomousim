"""CarUrbanDrive-v0: a lane-level route through a city in traffic, and its scripted baseline."""

import gymnasium as gym
import numpy as np

import autonomousim
from autonomousim import STATE
from autonomousim.tasks.car_urban_drive import CarUrbanDrive, scripted_baseline

URBAN = {"type": "urban", "seed": 2, "count": 2}
SMALL = {"map": URBAN, "distance": (200.0, 300.0), "npcs": 8, "cyclists": 2, "parked": 6, "pedestrians": 30}


def test_urban_drive_starts_on_a_route_through_the_city():
    n = 4
    envs = gym.make_vec(
        "autonomousim/CarUrbanDrive-v0", num_envs=n, vectorization_mode="vector_entry_point", num_threads=2, **SMALL
    )
    assert envs.single_observation_space.shape == (211,)
    assert envs.single_action_space.shape == (2,)
    obs, _ = envs.reset(seed=0)
    assert obs.shape == (n, 211) and np.isfinite(obs).all()
    state = envs.unwrapped.sim.state("agent").reshape(n, -1)
    pos = state[:, STATE["position"]][:, :2]
    goal = state[:, STATE["goal"]][:, :2]
    # The first goal 25 m along the route: ahead, at most 25 m away.
    d = np.linalg.norm(goal - pos, axis=1)
    assert ((d > 0.5) & (d < 25.5)).all(), d
    # Started on a lane, facing along it: the route ahead and near the centre line.
    route = obs[:, 3:11] / 0.05
    assert (route[:, 0] > 3.0).all() and (np.abs(route[:, 1]) < 2.0).all(), route[:, :2]
    for _ in range(5):
        obs, reward, terminated, truncated, info = envs.step(np.zeros((n, 2), np.float32))
    assert reward.shape == (n,) and np.isfinite(reward).all() and not terminated.any()
    envs.close()


def test_urban_drive_sensor_observation():
    task = CarUrbanDrive(observation="sensors", **SMALL)
    group = task.scenario()["groups"][0]
    assert [s["type"] for s in group["sensors"]] == ["lidar"]
    assert group["obs"][-1] == {"term": "lidar_log", "sensor": "lidar"}
    env = gym.make("autonomousim/CarUrbanDrive-v0", observation="sensors", **SMALL)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (199,) and np.isfinite(obs).all()
    env.close()


def test_urban_drive_scenario_is_busy():
    task = CarUrbanDrive()
    sc = task.scenario()
    names = [g["name"] for g in sc["groups"]]
    assert names == ["agent", "traffic", "buses", "cyclists", "parked"]
    assert sc["pedestrians"]["count"] == 100
    assert sc["groups"][0]["goals"]["route"] == {"destination": "lanes", "step": 25.0}
    assert "driver" not in sc["groups"][0]
    assert task.scripted_scenario()["groups"][0]["driver"] == {"type": "traffic", "respawn": 0.0}
    assert len(CarUrbanDrive(buses=0, cyclists=0, parked=0, pedestrians=0).scenario()["groups"]) == 2
    assert "pedestrians" not in CarUrbanDrive(pedestrians=0).scenario()


def test_urban_drive_rewards():
    task = CarUrbanDrive()
    task.bind(4, 0.05, 2)
    state = np.zeros((4, autonomousim.STATE_DIM))
    state[:, STATE["goal"]] = [20.0, 0.0, 0.0]
    task.reset(None, state)
    state[:, STATE["position"]] = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [18.0, 0.0, 0.0]]
    state[1, STATE["road"]] = [0.5, 0.0, 0.0]
    ev = autonomousim.Event
    events = np.array(
        [0, int(ev.RED_LIGHT), int(ev.WRONG_WAY) | int(ev.OFF_ROAD), int(ev.FINISHED)], np.uint32
    )
    action = np.zeros((4, 2), np.float32)
    r = task.reward(state, action, action, events)
    np.testing.assert_allclose(
        r, [0.2 - 0.01, -0.01 - 0.05 * 0.5 - 5.0, -0.01 - 0.2 - 0.2, 0.2 * 18.0 - 0.01 + 20.0]
    )
    assert task.succeeded(state, events).tolist() == [False, False, False, True]


def test_scripted_baseline_drives_the_routes():
    task = CarUrbanDrive(episode_time=120.0, **SMALL)
    result = scripted_baseline(task, num_envs=8, seed=0, num_threads=4)
    # The traffic driver follows the route, stops at red lights and gives way.
    assert result["success"] >= 0.75 and result["crashed"] == 0.0 and result["red_lights"] == 0.0, result
    assert 15.0 < result["time"] < 120.0, result
