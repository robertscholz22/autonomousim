"""Gymnasium environments: API conformance, vector semantics, seeding and the tasks."""

import json
import pathlib

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.vector import AutoresetMode

import autonomousim
from autonomousim import STATE, _native
from autonomousim.scenario import load_scenario, normalize, quat_up_z
from autonomousim.tasks import QuadHover, make_task
from autonomousim.tasks.base import map_source
from autonomousim.vector_env import AutonomousimVectorEnv

IDS = [f"autonomousim/{name}" for name in autonomousim.ENVS]
# Small maps for the API tests (the waypoint task defaults to a pool of 16 generated maps).
RURAL = {"type": "rural", "seed": 0, "count": 2, "cache": False}
KWARGS = {
    "autonomousim/QuadWaypointForest-v0": {"map": "forest"},
    "autonomousim/CarWaypointOffroad-v0": {"map": "flat"},
    "autonomousim/RoadFollowRural-v0": {"map": RURAL},
    "autonomousim/TrailerReverse-v0": {"map": RURAL},
}
OBS_DIM = {
    "autonomousim/QuadWaypointForest-v0": 148,
    "autonomousim/CarWaypointOffroad-v0": 231,
    "autonomousim/RoadFollowRural-v0": 97,
    "autonomousim/TrailerReverse-v0": 31,
}
ACT_DIM = {
    "autonomousim/CarWaypointOffroad-v0": 2,
    "autonomousim/RoadFollowRural-v0": 2,
    "autonomousim/TrailerReverse-v0": 2,
}

# ctbr: roll, pitch, yaw rate, thrust. Rotors off: the drone falls and crashes.
FALL = np.array([0.0, 0.0, 0.0, -1.0], np.float32)


@pytest.mark.parametrize("env_id", IDS)
@pytest.mark.filterwarnings("ignore:.*Box observation space (minimum|maximum) value is")
def test_check_env(env_id):
    env = gym.make(env_id, **KWARGS.get(env_id, {}))
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_id", IDS)
def test_vector_spaces_and_dtypes(env_id):
    envs = gym.make_vec(env_id, num_envs=4, num_threads=2, **KWARGS.get(env_id, {}))
    assert isinstance(envs.unwrapped, AutonomousimVectorEnv)
    assert envs.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
    d, a = OBS_DIM.get(env_id, 19), ACT_DIM.get(env_id, 4)
    assert envs.single_observation_space.shape == (d,) and envs.single_action_space.shape == (a,)
    assert envs.observation_space.shape == (4, d) and envs.action_space.shape == (4, a)
    obs, info = envs.reset(seed=3)
    assert obs.dtype == np.float32 and obs.shape == (4, d) and envs.observation_space.contains(obs)
    for _ in range(5):
        obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
        assert envs.observation_space.contains(obs)
        assert reward.dtype == np.float64 and reward.shape == (4,)
        assert terminated.dtype == bool and truncated.dtype == bool
        assert info["events"].dtype == np.uint32 and info["events"].shape == (4,)
    envs.close()


def test_same_step_autoreset_returns_the_final_observation():
    envs = gym.make_vec("autonomousim/QuadHover-v0", num_envs=3, num_threads=1)
    obs, _ = envs.reset(seed=0)
    actions = np.zeros((3, 4), np.float32)
    actions[1] = FALL  # only world 1 falls
    for step in range(1, 100):
        obs, reward, terminated, truncated, info = envs.step(actions)
        if terminated.any():
            break
    assert list(terminated) == [False, True, False] and not truncated.any()
    assert list(info["_final_obs"]) == [False, True, False] and list(info["_episode"]) == [False, True, False]
    assert info["episode"]["l"][1] == step and info["episode"]["r"][1] < 0  # includes the terminal penalty
    assert autonomousim.events.is_terminal(info["events"][1])
    final = info["final_obs"][1]
    # The final observation shows the crash (goal above, sinking faster than the 2 m/s crash
    # speed); the returned one starts the next episode (spawned at up to 1 m/s).
    assert final[2] > 0 and final[11] < -1.0  # goal_rel_world z, lin_vel_world z (both scaled 0.5)
    assert abs(obs[1][11]) <= 0.5 + 1e-6 and not np.array_equal(final, obs[1])
    np.testing.assert_array_equal(info["final_obs"][[0, 2]], obs[[0, 2]])
    # The new episode continues normally.
    obs, reward, terminated, truncated, info = envs.step(np.zeros((3, 4), np.float32))
    assert not terminated.any() and "final_obs" not in info
    assert envs.unwrapped.task.steps[1] == 1


def test_truncation_after_the_episode_time():
    envs = gym.make_vec("autonomousim/QuadHover-v0", num_envs=4, episode_time=0.2, num_threads=1)
    envs.reset(seed=1)
    for step in range(1, 11):
        obs, reward, terminated, truncated, info = envs.step(np.zeros((4, 4), np.float32))
        assert not terminated.any()
        assert truncated.all() == (step == 10) and truncated.any() == (step == 10)
    assert (info["episode"]["l"] == 10).all() and info["_final_obs"].all()
    # Rewards are positive near the goal and sum to the episode return.
    assert (reward > 0).all()


def test_seeding_is_reproducible():
    def rollout(seed, n=60):
        envs = gym.make_vec("autonomousim/QuadRecover-v0", num_envs=5, num_threads=2)
        rng = np.random.default_rng(0)
        out = [envs.reset(seed=seed)[0]]
        for _ in range(n):
            obs, reward, terminated, truncated, info = envs.step(rng.uniform(-1, 1, (5, 4)).astype(np.float32))
            out += [obs, reward, terminated, truncated]
        return out

    a, b, c = rollout(7), rollout(7), rollout(8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])


def test_world_zero_matches_the_single_environment():
    env = gym.make("autonomousim/QuadHover-v0")
    envs = gym.make_vec("autonomousim/QuadHover-v0", num_envs=3, num_threads=2)
    rng = np.random.default_rng(1)
    o1, _ = env.reset(seed=42)
    o3, _ = envs.reset(seed=42)
    np.testing.assert_array_equal(o1, o3[0])
    for _ in range(30):
        a = rng.uniform(-0.3, 0.3, (3, 4)).astype(np.float32)
        o1, r1, *_ = env.step(a[0])
        o3, r3, *_ = envs.step(a)
        np.testing.assert_array_equal(o1, o3[0])
        assert r1 == r3[0]


def test_disabled_autoreset_and_reset_mask():
    envs = AutonomousimVectorEnv(3, "hover", num_threads=1, autoreset_mode=AutoresetMode.DISABLED)
    assert envs.metadata["autoreset_mode"] == AutoresetMode.DISABLED
    envs.reset(seed=0)
    ended = np.zeros(3, bool)
    for _ in range(100):
        obs, reward, terminated, truncated, info = envs.step(np.tile(FALL, (3, 1)))
        assert "final_obs" not in info
        ended |= terminated
        if ended.all():
            break
    assert ended.all()
    frozen = envs.state.copy()
    envs.step(np.tile(FALL, (3, 1)))
    np.testing.assert_array_equal(envs.state[:, STATE["position"]], frozen[:, STATE["position"]])
    obs, _ = envs.reset(options={"reset_mask": np.array([True, False, True])})
    assert envs.state[0, 2] > frozen[0, 2] and envs.state[1, 2] == frozen[1, 2]
    assert list(envs.task.steps) == [0, envs.task.steps[1], 0] and envs.task.steps[1] > 0


def test_recover_starts_from_any_attitude():
    envs = gym.make_vec("autonomousim/QuadRecover-v0", num_envs=64, num_threads=2)
    envs.reset(seed=0)
    state = envs.unwrapped.state
    up = quat_up_z(state[:, STATE["orientation"]])
    assert up.min() < -0.5 and up.max() > 0.5
    speed = np.linalg.norm(state[:, STATE["velocity"]], axis=1)
    assert 0 < speed.max() <= 3.0 + 1e-9
    # The goal is the spawn point.
    np.testing.assert_allclose(state[:, STATE["goal"]], state[:, STATE["position"]])


def test_task_options():
    envs = gym.make_vec(
        "autonomousim/QuadHover-v0",
        num_envs=2,
        vehicle="iris_like",
        action_mode="motors",
        map="forest",
        policy_hz=100,
        wind=(1.0, 3.0),
        randomize={"mass": 0.1},
        overrides={"events": {"max_agl": 20.0}},
    )
    sim = envs.unwrapped.sim
    info = sim.group_info(0)
    assert (info["vehicle"], info["action_mode"], info["act_dim"]) == ("iris_like", "motors", 4)
    assert sim.policy_dt == pytest.approx(0.01)
    sc = json.loads(sim.scenario_json)
    assert sc["map"]["kind"] == "forest_patch" and sc["events"]["max_agl"] == 20.0
    assert sc["randomize_environment"]["wind_speed"] == [1.0, 3.0] and sc["groups"][0]["randomize"]["mass"] == 0.1
    obs = [{"term": "goal_rel_body"}, {"term": "gravity_body"}, {"term": "ang_vel_body", "scale": 0.1}]
    envs = AutonomousimVectorEnv(2, QuadHover(obs=obs), num_threads=1)
    assert envs.single_observation_space.shape == (9,)
    with pytest.raises(ValueError, match="unknown task"):
        make_task("dance")
    with pytest.raises(ValueError, match="unknown map"):
        make_task("hover", map="moon").scenario()
    with pytest.raises(ValueError, match="NEXT_STEP"):
        AutonomousimVectorEnv(1, autoreset_mode=AutoresetMode.NEXT_STEP)
    with pytest.raises(ValueError):
        AutonomousimVectorEnv(1, "hover", action_mode="warp")


def test_wild_map_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTONOMOUSIM_MAP_CACHE", str(tmp_path))
    envs = gym.make_vec("autonomousim/QuadHover-v0", num_envs=4, map="wild", map_count=2, num_threads=2)
    sim = envs.unwrapped.sim
    assert len(sim.map_hashes) == 2 and len(list(tmp_path.rglob("*.map"))) == 2
    envs.reset(seed=0)
    for _ in range(5):
        envs.step(np.zeros((4, 4), np.float32))
    assert {sim.map_index(i) for i in range(4)} <= {0, 1}


def test_rural_map_pool_with_road_terms(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTONOMOUSIM_MAP_CACHE", str(tmp_path))
    group = {
        "name": "cars",
        "count": 2,
        "vehicle": "sedan_like",
        "spawn": {"on_road": True, "min_separation": 10.0},
        "goals": {"kind": "route", "distance": [100.0, 300.0], "radius": 5.0},
        "obs": [{"term": "road"}, {"term": "route"}, {"term": "on_road"}],
    }
    config = {"name": "rural", "map": map_source("rural", 4, 2), "groups": [group]}
    sim = _native.BatchSim(json.dumps(config), 3, 0, 2)
    assert len(sim.map_hashes) == 2 and len(list(tmp_path.rglob("*.map"))) == 2
    obs = sim.obs(0)
    assert obs.shape == (3, 2, 15)
    # In the lane, facing along the route, on the road.
    assert np.all(np.abs(obs[..., 0]) < 0.6) and np.all(np.abs(obs[..., 1]) < 0.3)
    assert np.all(obs[..., 6] > 0.0) and np.all(obs[..., 14] == 1.0)


def test_hover_scenario_file_matches_the_task():
    root = pathlib.Path(__file__).resolve().parent.parent
    assert normalize(QuadHover().scenario()) == load_scenario(root / "assets/scenarios/hover.toml")


def goal_seeker(obs: np.ndarray) -> np.ndarray:
    """Velocity-mode pilot for the waypoint task: fly straight at the goal at up to 5 m/s,
    slowing down within 5 m. The goal (first observation term, body frame, scaled by 1/20) is
    rotated into the heading frame with the rot6d term (body x and y axes in the world)."""
    bx, by = obs[:, 9:12], obs[:, 12:15]
    rel_world = obs[:, :3, None] / 0.05 * np.stack([bx, by, np.cross(bx, by)], axis=1)
    rel_world = rel_world.sum(axis=1)
    yaw = np.arctan2(bx[:, 1], bx[:, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    rel = np.stack(
        [c * rel_world[:, 0] + s * rel_world[:, 1], -s * rel_world[:, 0] + c * rel_world[:, 1], rel_world[:, 2]], 1
    )
    a = np.zeros((len(obs), 4), np.float32)
    xy = rel[:, :2] / 5.0
    a[:, :2] = xy / np.maximum(1.0, np.linalg.norm(xy, axis=1, keepdims=True))
    a[:, 2] = np.clip(rel[:, 2] / 2.0, -1.0, 1.0)
    return a


def test_waypoint_forest_on_open_ground():
    from autonomousim import Event

    n = 4
    envs = gym.make_vec(
        "autonomousim/QuadWaypointForest-v0",
        num_envs=n,
        num_threads=2,
        map="flat",
        autoreset_mode=AutoresetMode.DISABLED,
    )
    task = envs.unwrapped.task
    assert envs.unwrapped.obs_layout[-1] == ("lidar_log", 20, 128)
    obs, _ = envs.reset(seed=5)
    start = envs.unwrapped.state[:, STATE["position"]].copy()
    ret, reached, done = np.zeros(n), np.zeros(n, int), np.zeros(n, bool)
    success = np.zeros(n, bool)
    while not done.all():
        obs, reward, terminated, truncated, info = envs.step(goal_seeker(obs))
        live = ~done
        ret[live] += reward[live]
        reached[live] += (info["events"][live] & Event.GOAL_REACHED) != 0
        success |= live & task.success
        done |= terminated | truncated
    # Every waypoint reached within the time, the last one ending the episode as a success.
    assert success.all() and (reached == 3).all(), (success, reached)
    assert (envs.unwrapped.state[:, STATE["goal_index"]] == 3).all()
    # Progress sums to the path length (from the start through the goals) minus the last
    # 2 m, plus 3 bonuses, minus small penalties.
    assert (ret > 3 * 10.0 + 3 * 20.0 - 2.0 - 5.0).all(), ret
    assert not (start == envs.unwrapped.state[:, STATE["position"]]).any()
    envs.close()


def test_waypoint_forest_height_limit():
    envs = gym.make_vec(
        "autonomousim/QuadWaypointForest-v0",
        num_envs=2,
        map="flat",
        max_agl=6.0,
        wind=None,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    envs.reset(seed=0)
    climb = np.tile(np.array([0.0, 0.0, 1.0, 0.0], np.float32), (2, 1))
    ended, last_reward, last_events = np.zeros(2, bool), np.zeros(2), np.zeros(2, np.uint32)
    for _ in range(200):
        _, reward, terminated, _, info = envs.step(climb)
        new = terminated & ~ended
        assert not envs.unwrapped.task.success[new].any()
        assert (envs.unwrapped.state[new, STATE["agl"]] > 5.9).all()
        last_reward[new], last_events[new] = reward[new], info["events"][new]
        ended |= terminated
        if ended.all():
            break
    assert ended.all() and (last_events & autonomousim.Event.OUT_OF_BOUNDS).all() and (last_reward < -20.0).all()
    envs.close()


def test_evaluate_reports_success():
    from autonomousim.rl import evaluate

    result = evaluate(goal_seeker, "autonomousim/QuadWaypointForest-v0", episodes=4, env_kwargs={"map": "flat"})
    assert result["success"] == 1.0 and result["failed"] == 0.0 and result["goals"] == 3.0
    assert "success" not in evaluate(lambda o: np.zeros((len(o), 4), np.float32), "autonomousim/QuadHover-v0", 4)


def car_driver(obs: np.ndarray) -> np.ndarray:
    """``vk`` actions steering at the goal (in the heading frame, the first observation term)
    at 60 % of the speed limit."""
    goal = obs[:, 0:2]
    heading_error = np.arctan2(goal[:, 1], goal[:, 0])
    return np.stack([np.full(len(obs), 0.6), np.clip(2.0 * heading_error, -1.0, 1.0)], 1).astype(np.float32)


def test_car_waypoint_on_open_ground():
    from autonomousim import Event

    n = 4
    envs = gym.make_vec(
        "autonomousim/CarWaypointOffroad-v0",
        num_envs=n,
        num_threads=2,
        map="flat",
        autoreset_mode=AutoresetMode.DISABLED,
    )
    task = envs.unwrapped.task
    assert envs.unwrapped.obs_layout[-1] == ("lidar_log", 15, 216)
    assert envs.unwrapped.sim.dt == 0.001 and envs.unwrapped.sim.decimation == 50
    obs, _ = envs.reset(seed=5)
    # No obstacles: the clearance of ground vehicles leaves the terrain out.
    assert (envs.unwrapped.state[:, STATE["clearance"]] == 20.0).all()
    ret, reached, done = np.zeros(n), np.zeros(n, int), np.zeros(n, bool)
    success = np.zeros(n, bool)
    while not done.all():
        obs, reward, terminated, truncated, info = envs.step(car_driver(obs))
        live = ~done
        ret[live] += reward[live]
        reached[live] += (info["events"][live] & Event.GOAL_REACHED) != 0
        success |= live & task.success
        done |= terminated | truncated
    # Every waypoint reached within the time, the last one ending the episode as a success;
    # progress sums to the path length (at least 3 × 25 m, less the last 3 m) plus 3 bonuses.
    assert success.all() and (reached == 3).all(), (success, reached)
    assert (ret > 3 * 10.0 + 3 * 25.0 - 3.0 - 5.0).all(), ret
    envs.close()


def road_driver(obs: np.ndarray) -> np.ndarray:
    """``vk`` actions steering at the lane point 5 m ahead (the first point of the ``route``
    term, scaled by 1/20) at up to 60 % of the 15 m/s speed limit, slower where the lane bends
    within 20 m (``road`` curvatures 5, 10 and 20 m ahead) for 2 m/s² of lateral acceleration,
    and while turning onto the lane (junctions turn sharply)."""
    ahead = obs[:, 6:8]
    heading_error = np.arctan2(ahead[:, 1], ahead[:, 0])
    bend = np.abs(obs[:, 2:5]).max(axis=1)
    turning = np.maximum(2.5, 9.0 - 12.0 * np.abs(heading_error))
    speed = np.minimum(turning, np.sqrt(2.0 / np.maximum(bend, 1e-3))) / 15.0
    return np.stack([speed, np.clip(2.0 * heading_error, -1.0, 1.0)], 1).astype(np.float32)


def test_road_follow_reaches_the_yard():
    from autonomousim import Event

    n = 4
    envs = gym.make_vec(
        "autonomousim/RoadFollowRural-v0",
        num_envs=n,
        num_threads=2,
        map=RURAL,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    task = envs.unwrapped.task
    layout = envs.unwrapped.obs_layout
    assert [t[0] for t in layout[:3]] == ["road", "route", "on_road"] and layout[-1] == ("lidar_log", 25, 72)
    obs, _ = envs.reset(seed=1)
    state = envs.unwrapped.state
    assert (np.abs(state[:, STATE["road"]][:, 0]) < 0.6).all() and (state[:, STATE["road"]][:, 2] == 0.0).all()
    ret, offsets, done = np.zeros(n), [], np.zeros(n, bool)
    success = np.zeros(n, bool)
    while not done.all():
        obs, reward, terminated, truncated, info = envs.step(road_driver(obs))
        live = ~done
        ret[live] += reward[live]
        offsets.append(np.abs(envs.unwrapped.state[live, STATE["road"]][:, 0]))
        success |= live & task.success
        done |= terminated | truncated
    # The scripted driver keeps to its lane and reaches every yard; it cuts the corners where
    # the route turns from one road into another (the offsets peak there).
    assert success.all(), success
    offsets = np.concatenate(offsets)
    assert offsets.mean() < 0.3 and np.quantile(offsets, 0.9) < 0.6 and offsets.max() < 3.5, (
        offsets.mean(),
        np.quantile(offsets, 0.9),
        offsets.max(),
    )
    assert (ret > task.finish_bonus + 100.0).all(), ret
    envs.close()


def test_road_follow_fails_off_the_road():
    envs = gym.make_vec(
        "autonomousim/RoadFollowRural-v0", num_envs=2, map=RURAL, autoreset_mode=AutoresetMode.DISABLED
    )
    from autonomousim import Event

    envs.reset(seed=0)
    # Full lock to the left: off the road (or into the bank of its cutting) within a few
    # seconds.
    turn = np.tile(np.array([0.5, 1.0], np.float32), (2, 1))
    failed, off, crashed = np.zeros(2, bool), np.zeros(2), np.zeros(2, bool)
    for _ in range(200):
        _, reward, terminated, truncated, info = envs.step(turn)
        ended = terminated & ~failed
        assert not envs.unwrapped.task.success.any()
        off[ended] = envs.unwrapped.state[ended, STATE["road"]][:, 2]
        crashed[ended] = (info["events"].reshape(-1)[ended] & int(Event.CRASH_TERRAIN)) != 0
        failed |= terminated
        if failed.all():
            break
    assert failed.all() and ((off > 3.0) | crashed).all() and (off > 3.0).any(), (off, crashed)
    envs.close()


def test_trailer_reverse_scripted_driver_parks():
    n = 4
    envs = gym.make_vec(
        "autonomousim/TrailerReverse-v0", num_envs=n, num_threads=2, map=RURAL, autoreset_mode=AutoresetMode.DISABLED
    )
    task = envs.unwrapped.task
    assert [t[0] for t in envs.unwrapped.obs_layout] == [
        "trailer_goal",
        "articulation",
        "speed",
        "steering",
        "last_action",
        "lidar_log",
    ]
    envs.reset(seed=0)
    d, psi = task.bay_errors(envs.unwrapped.state)
    assert ((d > 11.0) & (d < 25.0)).all() and (np.abs(psi) < np.radians(10.5)).all(), (d, psi)
    ret, done, success = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
    final = np.zeros_like(envs.unwrapped.state)
    while not done.all():
        _, reward, terminated, truncated, _ = envs.step(task.scripted(envs.unwrapped.state))
        live = ~done
        ret[live] += reward[live]
        ended = live & (terminated | truncated)
        success |= live & task.success
        final[ended] = envs.unwrapped.state[ended]
        assert not (terminated & ~task.success).any(), "the driver never crashes or jackknifes"
        done |= terminated | truncated
    # The feedback driver parks most rigs (it may run out of room to align the last few
    # degrees); parked rigs stand in the bay, aligned.
    assert success.sum() >= 3, success
    d, psi = task.bay_errors(final[success])
    assert (d < 1.0).all() and (np.abs(psi) < np.radians(5.0)).all()
    assert (ret[success] > task.success_bonus).all(), ret
    envs.close()


def test_trailer_reverse_fails_driving_away():
    envs = gym.make_vec("autonomousim/TrailerReverse-v0", num_envs=2, map=RURAL, autoreset_mode=AutoresetMode.DISABLED)
    envs.reset(seed=0)
    task = envs.unwrapped.task
    ahead = np.tile(np.array([1.0, 0.0], np.float32), (2, 1))
    failed, last, distance = np.zeros(2, bool), np.zeros(2), np.zeros(2)
    for _ in range(400):
        _, reward, terminated, truncated, _ = envs.step(ahead)
        ended = terminated & ~failed
        assert not (task.success | truncated).any()
        last[ended] = reward[ended]
        distance[ended] = task.bay_errors(envs.unwrapped.state)[0][ended]
        failed |= terminated
        if failed.all():
            break
    # Driving forward, away from the bay, ends the episode as a failure: more than 35 m off,
    # or a crash on the way out of the yard.
    assert failed.all() and (distance > 25.0).all() and (distance > 35.0).any(), distance
    assert (last < -10.0).all(), last
    envs.close()


def test_car_waypoint_fails_when_stuck():
    envs = gym.make_vec(
        "autonomousim/CarWaypointOffroad-v0",
        num_envs=2,
        map="flat",
        stuck_time=2.0,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    envs.reset(seed=0)
    stop = np.zeros((2, 2), np.float32)
    for k in range(100):
        _, reward, terminated, truncated, info = envs.step(stop)
        if terminated.any():
            break
    # Standing still for 2 s (40 steps at 20 Hz) ends the episode as a failure.
    assert terminated.all() and not envs.unwrapped.task.success.any() and 38 <= k <= 42, k
    assert (info["events"] & autonomousim.Event.STUCK).all() and (reward < -40.0).all()
    envs.close()
