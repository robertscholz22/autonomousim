"""Gymnasium environments: API conformance, vector semantics, seeding and the tasks."""

import json
import pathlib

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.vector import AutoresetMode

import autonomousim
from autonomousim import STATE
from autonomousim.scenario import load_scenario, normalize, quat_up_z
from autonomousim.tasks import QuadHover, make_task
from autonomousim.vector_env import AutonomousimVectorEnv

IDS = [f"autonomousim/{name}" for name in autonomousim.ENVS]
# Small maps for the API tests (the waypoint task defaults to a pool of 16 generated maps).
KWARGS = {"autonomousim/QuadWaypointForest-v0": {"map": "forest"}}
OBS_DIM = {"autonomousim/QuadWaypointForest-v0": 148}

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
    d = OBS_DIM.get(env_id, 19)
    assert envs.single_observation_space.shape == (d,) and envs.single_action_space.shape == (4,)
    assert envs.observation_space.shape == (4, d) and envs.action_space.shape == (4, 4)
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
