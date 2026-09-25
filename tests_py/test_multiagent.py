"""Multi-agent vector environment: per-group arrays, per-agent stopping, per-world autoreset,
seeding, and a mixed drone + car team."""

import numpy as np
import pytest

from autonomousim.events import Event
from autonomousim.multiagent import MultiAgentVectorEnv
from autonomousim.tasks import CarWaypointOffroad, QuadHover
from autonomousim.tasks.multi import MultiAgentTask, Team


def hover_team(count=4, **kwargs):
    """Drones in velocity mode: a zero action holds position; the hover task fails an agent
    that leaves the ±5 m box round its goal."""
    return MultiAgentTask(
        teams={"drones": Team(QuadHover(action_mode="velocity"), count)},
        overrides={"groups": [{"spawn": {"min_separation": 3.0}}]},
        **kwargs,
    )


def mixed_team(**kwargs):
    return MultiAgentTask(
        teams={
            "drones": Team(QuadHover(action_mode="velocity"), 3),
            "car": Team(CarWaypointOffroad(), 1),
        },
        **kwargs,
    )


def test_single_group_shapes_and_dtypes():
    envs = MultiAgentVectorEnv(3, hover_team(), seed=1, num_threads=2)
    assert envs.groups == ["drones"] and envs.count == {"drones": 4}
    assert envs.obs_dim == {"drones": 19} and envs.act_dim == {"drones": 4}
    obs, info = envs.reset(seed=5)
    assert obs["drones"].shape == (3, 4, 19) and obs["drones"].dtype == np.float32
    assert info["active"]["drones"].all()
    for _ in range(3):
        # A single-group env also takes a bare array.
        obs, reward, terminated, truncated, info = envs.step(envs.action_space("drones").sample())
        assert envs.observation_space("drones").contains(obs["drones"])
        assert reward["drones"].shape == (3, 4) and reward["drones"].dtype == np.float64
        assert terminated["drones"].shape == (3, 4) and terminated["drones"].dtype == bool
        assert truncated.shape == (3,) and truncated.dtype == bool
        assert info["events"]["drones"].shape == (3, 4) and info["events"]["drones"].dtype == np.uint32
    with pytest.raises(ValueError):
        envs.step({"cars": np.zeros((3, 4, 4))})
    envs.close()


def test_stopped_agents_are_masked_and_frozen():
    envs = MultiAgentVectorEnv(2, hover_team(), seed=0, num_threads=1)
    envs.reset(seed=0)
    actions = np.zeros((2, 4, 4), np.float32)
    actions[0, 1, 0] = 1.0  # world 0, agent 1 flies off at 5 m/s and leaves its goal box
    stop = None
    for step in range(200):
        _, reward, terminated, truncated, info = envs.step(actions)
        assert not truncated.any() and "_episode" not in info
        if terminated["drones"][0, 1]:
            stop = step
            assert info["active"]["drones"][0, 1]
            frozen = envs.state["drones"][0, 1].copy()
            break
    assert stop is not None and stop > 10
    # Only that agent stopped; the others hold position.
    assert terminated["drones"].sum() == 1
    for _ in range(20):
        _, reward, terminated, truncated, info = envs.step(actions)
        assert not info["active"]["drones"][0, 1] and info["active"]["drones"][[0, 0, 0], [0, 2, 3]].all()
        assert reward["drones"][0, 1] == 0.0 and not terminated["drones"][0, 1]
        assert info["events"]["drones"][0, 1] & Event.DISABLED
        np.testing.assert_array_equal(envs.state["drones"][0, 1], frozen)
        assert not terminated["drones"].any()
    envs.close()


def test_world_resets_when_every_agent_stopped_or_at_the_time_limit():
    envs = MultiAgentVectorEnv(2, hover_team(count=2, episode_time=3.0), seed=3, num_threads=1)
    envs.reset(seed=3)
    actions = np.zeros((2, 2, 4), np.float32)
    actions[0, :, 0] = 1.0  # both agents of world 0 fly off
    start = np.zeros(2, dtype=int)
    ends = []
    for step in range(1, 151):
        before = envs.state["drones"].copy()
        _, _, terminated, truncated, info = envs.step(actions)
        if "_episode" not in info:
            continue
        done = info["_episode"]
        ends.append((step, done.copy(), truncated.copy()))
        np.testing.assert_array_equal(info["_final_obs"], done)
        assert info["final_obs"]["drones"].shape == (2, 2, 19)
        for w in np.flatnonzero(done):
            assert info["episode"]["l"][w] == step - start[w]
            start[w] = step
            # A fresh episode: a new spawn, everyone active again.
            assert not np.array_equal(envs.state["drones"][w, :, :3], before[w, :, :3])
    # World 0 ends early by termination (not truncation); world 1 is truncated at 3 s.
    first0 = next(e for e in ends if e[1][0])
    assert first0[0] < 150 and not first0[2][0]
    first1 = next(e for e in ends if e[1][1])
    assert first1[0] == 150 and first1[2][1]
    envs.close()


def test_seeding_reproduces_every_group():
    def run(seed):
        envs = MultiAgentVectorEnv(2, mixed_team(), seed=seed, num_threads=2)
        obs, _ = envs.reset(seed=seed)
        rng = np.random.default_rng(0)
        out = [obs]
        for _ in range(5):
            actions = {g: rng.uniform(-1, 1, envs.action_space(g).shape).astype(np.float32) for g in envs.groups}
            out.append(envs.step(actions)[0])
        envs.close()
        return out

    a, b, c = run(7), run(7), run(8)
    for x, y in zip(a, b, strict=True):
        for g in x:
            np.testing.assert_array_equal(x[g], y[g])
    assert not np.array_equal(a[0]["car"], c[0]["car"])


def test_mixed_team_of_drones_and_a_car():
    envs = MultiAgentVectorEnv(3, mixed_team(episode_time=0.5), seed=0, num_threads=2)
    assert envs.groups == ["drones", "car"]
    assert envs.count == {"drones": 3, "car": 1}
    assert envs.obs_dim == {"drones": 19, "car": 231} and envs.act_dim == {"drones": 4, "car": 2}
    assert envs.sim.dt == pytest.approx(1e-3)  # 1 kHz with a ground vehicle
    obs, info = envs.reset(seed=0)
    assert obs["car"].shape == (3, 1, 231)
    with pytest.raises(TypeError):
        envs.step(np.zeros((3, 3, 4), np.float32))
    zero = {g: np.zeros(envs.action_space(g).shape, np.float32) for g in envs.groups}
    for step in range(1, 26):
        obs, reward, terminated, truncated, info = envs.step(zero)
        assert set(reward) == {"drones", "car"} and reward["car"].shape == (3, 1)
        assert np.isfinite(reward["drones"]).all() and np.isfinite(reward["car"]).all()
    # 0.5 s at 50 Hz: every world is truncated on step 25 and reset.
    assert truncated.all() and info["_episode"].all()
    assert info["episode"]["r"]["car"].shape == (3, 1) and info["episode"]["success"]["drones"].shape == (3, 3)
    assert info["final_obs"]["car"].shape == (3, 1, 231)
    envs.close()
