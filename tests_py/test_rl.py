"""Training helpers (``autonomousim.rl``) and short runs of the example scripts."""

import json
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
    for name in ("ppo_continuous", "sac_continuous", "ppo_pixels", "eval_record", "export_policy"):
        sys.modules.pop(name, None)


def _run_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    (run,) = (tmp_path / "runs").iterdir()
    return run


def _check_export(policy: pathlib.Path, output: str) -> None:
    """Export ``policy`` and run the exported network (as the Rust loader reads it) in numpy."""
    import export_policy

    export_policy.main([str(policy)])
    data = json.loads(policy.with_suffix(".json").read_text())
    assert data["format"] == "autonomousim-policy" and data["output"] == output
    assert data["scenario"]["groups"][0]["name"] == data["group"]
    n = data["obs_norm"]
    x = np.asarray(data["check"]["obs"], np.float64)
    x = np.clip((x - n["mean"]) / np.sqrt(np.asarray(n["var"]) + n["eps"]), -n["clip"], n["clip"])
    act = {"tanh": np.tanh, "relu": lambda z: np.maximum(z, 0.0), "identity": lambda z: z}
    for layer in data["layers"]:
        w = np.asarray(layer["weight"]).reshape(layer["shape"])
        x = act[layer["activation"]](x @ w.T + layer["bias"])
    x = np.clip(x, -1.0, 1.0) if output == "clip" else np.tanh(x)
    assert len(x) >= 16
    np.testing.assert_allclose(x, data["check"]["action"], atol=1e-4)


def test_ppo_and_eval_record(examples, tmp_path, capsys):
    import eval_record
    import ppo_continuous

    result = ppo_continuous.main(
        ["--num-envs", "16", "--num-steps", "16", "--total-timesteps", "512", "--sim-threads", "2",
         "--torch-threads", "1", "--eval-episodes", "4", "--no-tensorboard", "--runs-dir", str(tmp_path / "runs")]
    )  # fmt: skip
    assert set(result) >= {"return", "length", "survived", "final_error_m"}
    policy = _run_dir(tmp_path) / "policy.pt"
    assert policy.with_suffix(".json").exists()  # exported after training
    out = tmp_path / "rec.mcap"
    # Records three episodes and checks read-back and bit-exact replay (raises otherwise).
    eval_record.main([str(policy), "--episodes", "3", "--out", str(out)])
    text = capsys.readouterr().out
    assert "read-back:" in text and "replay: 3 episodes" in text
    _check_export(policy, "clip")


def test_sac_and_eval_record(examples, tmp_path, capsys):
    import eval_record
    import sac_continuous

    sac_continuous.main(
        ["--num-envs", "4", "--total-timesteps", "600", "--learning-starts", "400", "--batch-size", "32",
         "--updates-per-step", "2", "--buffer-size", "1000", "--hidden", "32", "--sim-threads", "1",
         "--torch-threads", "1", "--eval-episodes", "2", "--no-tensorboard", "--runs-dir", str(tmp_path / "runs")]
    )  # fmt: skip
    assert (_run_dir(tmp_path) / "policy.json").exists()
    eval_record.main([str(_run_dir(tmp_path) / "policy.pt"), "--episodes", "2", "--out", str(tmp_path / "s.mcap")])
    assert "replay: 2 episodes" in capsys.readouterr().out
    _check_export(_run_dir(tmp_path) / "policy.pt", "tanh")


def test_ppo_pixels_export_runs_in_rust(examples, tmp_path):
    """A pixel policy exported after a short run: the Rust network (``_native.Policy``, which
    also checks the file's samples) matches PyTorch within 1e-5 on random states and images."""
    import ppo_pixels

    from autonomousim import _native

    ppo_pixels.main(
        ["--num-envs", "8", "--num-steps", "16", "--total-timesteps", "256", "--sim-threads", "2",
         "--torch-threads", "1", "--eval-episodes", "2", "--no-tensorboard", "--runs-dir", str(tmp_path / "runs"),
         "--env-kwargs", '{"image_size": 24, "depth": true}']
    )  # fmt: skip
    import export_policy
    import torch

    trained = _run_dir(tmp_path) / "policy.pt"
    assert trained.with_suffix(".json").exists()  # exported after training
    # The last actor layer starts at 0.01 of the usual scale: scaled up, actions are of order 1.
    ckpt = torch.load(trained, weights_only=False)
    for k in ("actor_mean.4.weight", "actor_mean.4.bias"):
        ckpt["agent"][k] *= 20.0
    path = tmp_path / "scaled.pt"
    torch.save(ckpt, path)
    export_policy.main([str(path)])
    data = json.loads(path.with_suffix(".json").read_text())
    assert data["algo"] == "ppo_pixels" and data["encoder"]["image_shape"] == [24, 24, 4]
    assert len(data["check"]["image"]) == len(data["check"]["obs"]) >= 16
    rust = _native.Policy(path.with_suffix(".json").read_text())
    torch_policy, _ = ppo_pixels.load_policy(path)
    assert rust.image_shape == (24, 24, 4) and (rust.obs_dim, rust.act_dim) == (17, 4)
    rng = np.random.default_rng(0)
    state = rng.normal(0.0, 3.0, (64, 17)).astype(np.float32)
    image = rng.integers(0, 256, (64, 24, 24, 4), dtype=np.uint8)
    expected = torch_policy({"state": state, "image": image})
    inside = np.abs(expected) < 1.0
    assert inside.mean() > 0.5 and np.abs(expected[inside]).mean() > 0.1, expected
    actions = rust.act(state, image)
    np.testing.assert_allclose(actions, expected, rtol=0, atol=1e-5)
    print(f"largest difference to PyTorch {np.abs(actions - expected).max():.1e}")
    with pytest.raises(ValueError, match="images"):
        rust.act(state)
