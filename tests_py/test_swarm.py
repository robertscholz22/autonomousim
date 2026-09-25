"""SwarmHover: formation goals, the task's observation and reward, and a short IPPO run."""

import pathlib
import sys

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


def test_ippo_runs_and_evaluates(tmp_path):
    pytest.importorskip("torch")
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples"))
    import ppo_multiagent

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
