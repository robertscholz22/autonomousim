"""Export a trained policy for Rust programs: the viewer flies it without Python.

    uv run python examples/export_policy.py runs/<run>/policy.pt            # writes runs/<run>/policy.json
    cargo run -p autonomousim-viewer --release -- policy runs/<run>/policy.json

The training scripts export their final checkpoints this way themselves (``--no-export``
skips it); run this for intermediate checkpoints or other task options.

The JSON file (read by ``autonomousim_sim::policy``) holds the task's scenario, the
observation normalisation and the actor network as a list of dense layers:

- PPO: the action mean (two tanh layers and a linear one), clipped to [−1, 1];
- SAC: the tanh of the mean (two ReLU layers and a linear one);
- PPO from pixels (``ppo_pixels.py``): as PPO over the image features and the state, with
  the image encoder (convolutions and a dense layer) as ``encoder``.

Checkpoints of ``ppo_multiagent.py`` export the policy of their group with the multi-agent
task's scenario; the viewer flies every agent of that group with it.

The file also carries observations (and images) from a few short episodes with the actions
PyTorch computed for them; the Rust loader checks its network against them.
"""

import argparse
import base64
import json
import pathlib
import sys
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode
from torch import nn

from autonomousim.multiagent import MultiAgentVectorEnv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_record import load_policy  # noqa: E402

FORMAT = "autonomousim-policy"
ACTIVATIONS = {nn.Tanh: "tanh", nn.ReLU: "relu"}


def dense_layers(modules: list[nn.Module]) -> list[dict[str, Any]]:
    """``Linear`` modules, each with the activation that follows it (identity if none)."""
    layers: list[dict[str, Any]] = []
    for m in modules:
        if isinstance(m, nn.Linear):
            w = m.weight.detach().float().numpy()
            layers.append(
                {
                    "shape": list(w.shape),
                    "weight": w.reshape(-1).tolist(),
                    "bias": m.bias.detach().float().numpy().tolist(),
                    "activation": "identity",
                }
            )
        elif type(m) in ACTIVATIONS and layers and layers[-1]["activation"] == "identity":
            layers[-1]["activation"] = ACTIVATIONS[type(m)]
        else:
            raise ValueError(f"cannot export {m!r}")
    return layers


def conv_layers(modules: list[nn.Module]) -> list[dict[str, Any]]:
    """``Conv2d`` modules (square stride and padding, no dilation or groups), each with the
    activation that follows it; a final ``Flatten``."""
    layers: list[dict[str, Any]] = []
    for m in modules:
        if isinstance(m, nn.Conv2d):
            stride, padding = set(m.stride), set(m.padding) if isinstance(m.padding, tuple) else {m.padding}
            if len(stride) != 1 or len(padding) != 1 or set(m.dilation) != {1} or m.groups != 1:
                raise ValueError(f"cannot export {m!r}")
            w = m.weight.detach().float().numpy()
            layers.append(
                {
                    "shape": list(w.shape),
                    "stride": stride.pop(),
                    "padding": padding.pop(),
                    "weight": w.reshape(-1).tolist(),
                    "bias": m.bias.detach().float().numpy().tolist(),
                    "activation": "identity",
                }
            )
        elif type(m) in ACTIVATIONS and layers and layers[-1]["activation"] == "identity":
            layers[-1]["activation"] = ACTIVATIONS[type(m)]
        elif not isinstance(m, nn.Flatten):
            raise ValueError(f"cannot export {m!r}")
    return layers


def image_encoder(policy, algo: str) -> dict[str, Any] | None:
    """The image encoder of a pixel policy, else ``None``."""
    if algo != "ppo_pixels":
        return None
    enc = policy.agent.encoder
    (fc,) = dense_layers(list(enc.fc))
    return {"image_shape": list(policy.agent.image_shape), "convs": conv_layers(list(enc.conv)), "fc": fc}


def network(policy, algo: str) -> tuple[list[dict[str, Any]], str]:
    if algo in ("ppo", "ppo_pixels"):
        return dense_layers(list(policy.agent.actor_mean)), "clip"
    if algo == "sac":
        return dense_layers([*policy.actor.trunk, policy.actor.fc_mean]), "tanh"
    raise ValueError(f"unknown algorithm {algo!r}")


def check_samples(policy, env_id: str, env_kwargs: dict[str, Any], episodes: int = 8, steps: int = 25):
    """Observations of the first ``steps`` steps of a few episodes and the policy's actions."""
    envs = gym.make_vec(env_id, num_envs=episodes, num_threads=1, autoreset_mode=AutoresetMode.DISABLED, **env_kwargs)
    obs, _ = envs.reset(seed=12345)
    seen, images, actions = [], [], []
    for k in range(steps):
        a = policy(obs, deterministic=True)
        if k % 5 == 0:
            if isinstance(obs, dict):  # pixel tasks
                seen.append(obs["state"].copy())
                images.append(obs["image"].copy())
            else:
                seen.append(obs.copy())
            actions.append(a.copy())
        obs, *_ = envs.step(a)
    task = envs.unwrapped.task
    envs.close()
    return np.concatenate(seen), np.concatenate(actions), task, np.concatenate(images) if images else None


def check_samples_multi(policy, task: str, task_kwargs: dict[str, Any], group: str, episodes: int = 2, steps: int = 25):
    """As ``check_samples`` for a multi-agent task: the group's agents in a few worlds."""
    envs = MultiAgentVectorEnv(episodes, task, seed=12345, num_threads=1, autoreset=False, **task_kwargs)
    obs, _ = envs.reset(seed=12345)
    seen, actions = [], []
    for k in range(steps):
        acts = {g: np.zeros(envs.action_space(g).shape, np.float32) for g in envs.groups}
        o = obs[group].reshape(-1, envs.obs_dim[group])
        a = policy(o, deterministic=True)
        acts[group] = a.reshape(acts[group].shape)
        if k % 5 == 0:
            seen.append(o.copy())
            actions.append(a.copy())
        obs, *_ = envs.step(acts)
    task_obj = envs.task
    envs.close()
    return np.concatenate(seen), np.concatenate(actions), task_obj


def export(path: pathlib.Path, out: pathlib.Path | None = None, env_kwargs: dict[str, Any] | None = None) -> pathlib.Path:
    policy, ckpt = load_policy(path)
    args = ckpt["args"]
    layers, output = network(policy, ckpt["algo"])
    if "task" in args:
        env_id = args["task"]
        kwargs = args.get("task_kwargs", {}) if env_kwargs is None else env_kwargs
        group = ckpt["group"]
        obs, actions, task = check_samples_multi(policy, env_id, kwargs, group)
        images = None
    else:
        env_id = args["env_id"]
        kwargs = args.get("env_kwargs", {}) if env_kwargs is None else env_kwargs
        obs, actions, task, images = check_samples(policy, env_id, kwargs)
        # The learning group (scripted groups, driven by a ``driver``, may come first).
        group = next(g["name"] for g in task.scenario()["groups"] if "driver" not in g)
    rms = policy.obs_norm.rms
    data = {
        "format": FORMAT,
        "version": 1,
        "name": path.resolve().parent.name,
        "env_id": env_id,
        "env_kwargs": kwargs,
        "algo": ckpt["algo"],
        "global_step": ckpt.get("global_step"),
        "scenario": task.scenario(),
        "group": group,
        "episode_time": task.episode_time,
        "obs_norm": {
            "mean": rms.mean.tolist(),
            "var": rms.var.tolist(),
            "clip": policy.obs_norm.clip,
            "eps": 1e-8,
        },
        "layers": layers,
        "output": output,
        "check": {"obs": obs.tolist(), "action": actions.tolist()},
    }
    encoder = image_encoder(policy, ckpt["algo"])
    if encoder is not None:
        data["encoder"] = encoder
        data["check"]["image"] = [base64.b64encode(np.ascontiguousarray(i).tobytes()).decode() for i in images]
    out = out or path.with_suffix(".json")
    out.write_text(json.dumps(data))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("policy", type=pathlib.Path, help="policy.pt written by ppo_continuous.py, sac_continuous.py, ppo_pixels.py or ppo_multiagent.py")
    p.add_argument("--out", type=pathlib.Path, default=None, help="default: policy.json next to the checkpoint")
    p.add_argument("--env-kwargs", type=json.loads, default=None, help="task options (default: the training ones)")
    args = p.parse_args(argv)
    out = export(args.policy, args.out, args.env_kwargs)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
