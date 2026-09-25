"""Swarm tasks: formation goals, the tasks' observations and rewards, and a short IPPO run with
export, recording and replay."""

import pathlib

import numpy as np
import pytest

from autonomousim.multiagent import MultiAgentVectorEnv
from autonomousim.scenario import STATE
from autonomousim.tasks.swarm import SwarmHover


def test_formation_slots_and_observation():
    envs = MultiAgentVectorEnv(4, "swarm_hover", seed=0, num_threads=2, count=9, spacing=2.5)
    assert envs.obs_dim == {"drones": 7 * 3 + 20} and envs.act_dim == {"drones": 4}
    envs.reset(seed=0)
    s = envs.state["drones"]
    goals, pos = s[..., STATE["goal"]], s[..., STATE["position"]]
    for w in range(4):
        d = np.linalg.norm(goals[w, :, None, :2] - goals[w, None, :, :2], axis=-1)
        np.fill_diagonal(d, np.inf)
        np.testing.assert_allclose(d.min(axis=1), 2.5, atol=1e-9)  # a 3×3 grid, 2.5 m apart
        np.testing.assert_allclose(goals[w, :, :2].mean(axis=0), pos[w, :, :2].mean(axis=0), atol=1e-9)
        assert np.ptp(pos[w, :, :2], axis=0).max() <= 8.0  # spawned in an 8 m cluster
    assert not np.allclose(goals[0, :, :2].mean(0), goals[1, :, :2].mean(0))
    np.testing.assert_allclose(SwarmHover.formation_error(s), np.linalg.norm(goals - pos, axis=-1))
    envs.close()


def test_proximity_costs_reward():
    task = SwarmHover(count=2)
    agent = task.agent
    agent.bind(2, 0.02, 4)
    state = np.zeros((2, max(sl.stop for sl in STATE.values())))
    state[:, STATE["agent_clearance"]] = [[5.0], [0.25]]
    r = agent.reward(state, np.zeros((2, 4)), np.zeros((2, 4)), np.zeros(2, np.uint32))
    np.testing.assert_allclose(r[0] - r[1], agent.proximity_weight * (1.0 - 0.25 / agent.safe_distance))


def test_swarm_waypoint_forest_spawns_together_and_sees_neighbours():
    envs = MultiAgentVectorEnv(2, "swarm_waypoint_forest", seed=0, num_threads=2, map="forest", count=6)
    assert envs.obs_dim == {"drones": 148 + 7 * 3 + 1} and envs.act_dim == {"drones": 4}
    obs, _ = envs.reset(seed=3)
    s = envs.state["drones"]
    for w in range(2):
        p = s[w, :, :3]
        d = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(6) * 99
        assert d.min() >= 4.0 - 1e-9 and np.ptp(p[:, :2], axis=0).max() <= 20.0
    # Every neighbour slot is filled (6 drones within 20 m), and its presence flag is 1.
    np.testing.assert_array_equal(obs["drones"][..., 148 + 6 : 148 + 21 : 7], 1.0)
    envs.close()


def test_ippo_runs_exports_and_replays(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).parents[1] / "examples"))
    import eval_record
    import ppo_multiagent
    from test_rl import _check_export

    result = ppo_multiagent.main(
        [
            "--total-timesteps", "8192", "--num-envs", "4", "--num-steps", "32", "--sim-threads", "2",
            "--torch-threads", "1", "--eval-episodes", "3", "--no-tensorboard", "--runs-dir", str(tmp_path),
            "--task-kwargs", '{"count": 4, "episode_time": 1.0}',
        ]
    )  # fmt: skip
    assert result["episodes"] == 3 and np.isfinite(result["formation_error_mean"])
    run = next(tmp_path.iterdir())
    assert (run / "policy.pt").exists() and (run / "policy_drones.pt").exists()

    import ppo_continuous

    policy, ckpt = ppo_continuous.load_policy(run / "policy.pt")
    assert ckpt["group"] == "drones" and policy(np.zeros((2, 41), np.float32)).shape == (2, 4)
    _check_export(run / "policy.pt", "clip")

    # A tight goal box stops some drones mid-episode; the replay has to stop them too.
    out = tmp_path / "swarm.mcap"
    kwargs = '{"count": 4, "episode_time": 1.0, "bounds": 2.5}'
    eval_record.main([str(run / "policy.pt"), "--episodes", "2", "--out", str(out), "--env-kwargs", kwargs])
    text = capsys.readouterr().out
    assert "of 4 agents equal the live ones" in text and "replay: 2 episodes" in text
    rec = eval_record.read_recording(out)
    disabled = [s["disabled"] for ep in rec["episodes"] for a in ep["agents"] for s in a["states"]]
    assert any(disabled) and not all(disabled)


def test_warm_start_from_a_single_agent_policy(tmp_path, monkeypatch):
    """A policy for the single-drone forest observation (a prefix of the swarm drone's) acts
    unchanged after padding; the new inputs get statistics from a warm-up."""
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).parents[1] / "examples"))
    import ppo_continuous
    import ppo_multiagent

    torch.manual_seed(0)
    single = ppo_continuous.Agent(148, 4, hidden=16)
    norm = ppo_continuous.ObsNormalizer(148)
    norm.rms.mean[:] = np.random.default_rng(0).normal(size=148)
    ppo_continuous.save_policy(tmp_path / "single.pt", single, norm, ppo_multiagent.parse_args([]))

    args = ppo_multiagent.parse_args(["--hidden", "16", "--num-envs", "2"])
    envs = MultiAgentVectorEnv(2, "swarm_waypoint_forest", seed=0, num_threads=1, map="forest", count=4)
    grp = ppo_multiagent.Group("drones", envs, args)
    ppo_multiagent.init_group(grp, str(tmp_path / "single.pt"), envs, "drones", warmup=5)
    obs, _ = envs.reset(seed=1)
    o = obs["drones"].reshape(8, -1)
    np.testing.assert_array_equal(
        ppo_continuous.Policy(grp.agent, grp.obs_norm)(o), ppo_continuous.Policy(single, norm)(o[:, :148])
    )
    assert (grp.obs_norm.rms.var[148:] >= 0.05).all() and grp.obs_norm.rms.mean[148 + 6] == 1.0
    envs.close()
