"""Training helpers (``autonomousim.rl``) and short runs of the example scripts."""

import pathlib
import sys

import numpy as np
import pytest

from autonomousim.rl import ObsNormalizer, RewardScaler, RunningMeanStd, evaluate

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    batches = [rng.normal(3.0, 2.0, (n, 4)) for n in (1, 7, 100, 33)]
    rms = RunningMeanStd((4,))
    for b in batches:
        rms.update(b)
    all_ = np.concatenate(batches)
    np.testing.assert_allclose(rms.mean, all_.mean(0), rtol=1e-5)
    np.testing.assert_allclose(rms.var, all_.var(0), rtol=1e-4)


def test_obs_normalizer_round_trip():
    rng = np.random.default_rng(1)
    norm = ObsNormalizer(3, clip=5.0)
    x = rng.normal([1.0, -2.0, 100.0], [0.1, 1.0, 50.0], (1000, 3))
    y = norm(x, update=True)
    assert y.dtype == np.float32 and np.abs(y.mean(0)).max() < 0.05 and np.abs(y.std(0) - 1).max() < 0.05
    assert norm(np.array([[1e6, 0.0, 0.0]]))[0, 0] == 5.0
    other = ObsNormalizer(3)
    other.load_state_dict(norm.state_dict())
    assert other.clip == 5.0 and np.array_equal(other(x[:5]), norm(x[:5]))


def test_reward_scaler_restarts_returns():
    scale = RewardScaler(2, gamma=0.5)
    scale(np.array([1.0, 1.0]), np.array([True, False]))
    assert scale.ret.tolist() == [0.0, 1.0]
    scale(np.array([1.0, 1.0]), np.array([False, False]))
    assert scale.ret.tolist() == [1.0, 1.5]


def test_evaluate_constant_policy():
    # Rotors off: every drone falls and crashes well before the 10 s time limit.
    r = evaluate(lambda obs: np.tile([0.0, 0.0, 0.0, -1.0], (len(obs), 1)), "autonomousim/QuadHover-v0", 8, 0)
    assert r["survived"] == 0.0 and 1 <= r["length"] < 100 and np.isnan(r["final_error_m"])


@pytest.fixture
def examples(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    yield
    for name in ("ppo_continuous", "sac_continuous", "eval_record"):
        sys.modules.pop(name, None)


def _run_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    (run,) = (tmp_path / "runs").iterdir()
    return run


def test_ppo_and_eval_record(examples, tmp_path, capsys):
    import eval_record
    import ppo_continuous

    result = ppo_continuous.main(
        ["--num-envs", "16", "--num-steps", "16", "--total-timesteps", "512", "--sim-threads", "2",
         "--torch-threads", "1", "--eval-episodes", "4", "--no-tensorboard", "--runs-dir", str(tmp_path / "runs")]
    )  # fmt: skip
    assert set(result) >= {"return", "length", "survived", "final_error_m"}
    policy = _run_dir(tmp_path) / "policy.pt"
    out = tmp_path / "rec.mcap"
    # Records three episodes and checks read-back and bit-exact replay (raises otherwise).
    eval_record.main([str(policy), "--episodes", "3", "--out", str(out)])
    text = capsys.readouterr().out
    assert "read-back:" in text and "replay: 3 episodes" in text


def test_sac_and_eval_record(examples, tmp_path, capsys):
    import eval_record
    import sac_continuous

    sac_continuous.main(
        ["--num-envs", "4", "--total-timesteps", "600", "--learning-starts", "400", "--batch-size", "32",
         "--updates-per-step", "2", "--buffer-size", "1000", "--hidden", "32", "--sim-threads", "1",
         "--torch-threads", "1", "--eval-episodes", "2", "--no-tensorboard", "--runs-dir", str(tmp_path / "runs")]
    )  # fmt: skip
    eval_record.main([str(_run_dir(tmp_path) / "policy.pt"), "--episodes", "2", "--out", str(tmp_path / "s.mcap")])
    assert "replay: 2 episodes" in capsys.readouterr().out
