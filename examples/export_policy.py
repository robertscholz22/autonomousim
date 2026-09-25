"""Export a trained policy for Rust programs: the viewer flies it without Python.

    uv run python examples/export_policy.py runs/<run>/policy.pt            # writes runs/<run>/policy.json
    cargo run -p autonomousim-viewer --release -- policy runs/<run>/policy.json

The JSON file (read by ``autonomousim_sim::policy``) holds the task's scenario, the
observation normalisation and the actor network as a list of dense layers:

- PPO: the action mean (two tanh layers and a linear one), clipped to [−1, 1];
- SAC: the tanh of the mean (two ReLU layers and a linear one).

Checkpoints of ``ppo_multiagent.py`` export the policy of their group with the multi-agent
task's scenario; the viewer flies every agent of that group with it.

The file also carries observations from a few short episodes with the actions PyTorch
computed for them; the Rust loader checks its network against them.
"""

import argparse
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


def network(policy, algo: str) -> tuple[list[dict[str, Any]], str]:
    if algo == "ppo":
        return dense_layers(list(policy.agent.actor_mean)), "clip"
    if algo == "sac":
        return dense_layers([*policy.actor.trunk, policy.actor.fc_mean]), "tanh"
    raise ValueError(f"unknown algorithm {algo!r}")


def check_samples(policy, env_id: str, env_kwargs: dict[str, Any], episodes: int = 8, steps: int = 25):
    """Observations of the first ``steps`` steps of a few episodes and the policy's actions."""
    envs = gym.make_vec(env_id, num_envs=episodes, num_threads=1, autoreset_mode=AutoresetMode.DISABLED, **env_kwargs)
    obs, _ = envs.reset(seed=12345)
    seen, actions = [], []
    for k in range(steps):
        a = policy(obs, deterministic=True)
        if k % 5 == 0:
            seen.append(obs.copy())
            actions.append(a.copy())
        obs, *_ = envs.step(a)
    task = envs.unwrapped.task
    envs.close()
    return np.concatenate(seen), np.concatenate(actions), task


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
    else:
        env_id = args["env_id"]
        kwargs = args.get("env_kwargs", {}) if env_kwargs is None else env_kwargs
        obs, actions, task = check_samples(policy, env_id, kwargs)
        group = task.scenario()["groups"][0]["name"]
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
    out = out or path.with_suffix(".json")
    out.write_text(json.dumps(data))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("policy", type=pathlib.Path, help="policy.pt written by ppo_continuous.py, sac_continuous.py or ppo_multiagent.py")
    p.add_argument("--out", type=pathlib.Path, default=None, help="default: policy.json next to the checkpoint")
    p.add_argument("--env-kwargs", type=json.loads, default=None, help="task options (default: the training ones)")
    args = p.parse_args(argv)
    out = export(args.policy, args.out, args.env_kwargs)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
