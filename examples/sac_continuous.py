"""SAC with continuous actions on autonomousim's native vector environments (CleanRL style,
one file).

    make train-deps
    uv run python examples/sac_continuous.py --env-id autonomousim/QuadHover-v0 --total-timesteps 200000
    uv run python examples/eval_record.py runs/<run>/policy.pt

Differences from CleanRL's ``sac_continuous_action.py``:

- A few worlds (``--num-envs``, default 16) step in one native call; each vector step is
  followed by ``--updates-per-step`` gradient updates.
- Transitions that end an episode store ``info["final_obs"]`` as the next observation, so
  truncated episodes bootstrap and terminated ones do not.
- Observations are stored raw and normalised with the current running statistics when a batch
  is sampled (``autonomousim.rl.ObsNormalizer``); the statistics are saved with the policy.

The checkpoint (``runs/<run>/policy.pt``) holds the actor, the observation statistics and
the arguments; ``load_policy`` rebuilds a numpy-in, numpy-out policy from it.
"""

import argparse
import json
import pathlib
import random
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import autonomousim  # noqa: F401  (registers the environments)
from autonomousim.rl import ObsNormalizer, evaluate

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a = p.add_argument
    a("--env-id", default="autonomousim/QuadHover-v0")
    a("--exp-name", default="sac")
    a("--seed", type=int, default=1)
    a("--total-timesteps", type=int, default=200_000)
    a("--num-envs", type=int, default=16)
    a("--updates-per-step", type=int, default=4, help="gradient updates per vector step")
    a("--buffer-size", type=int, default=1_000_000)
    a("--gamma", type=float, default=0.99)
    a("--tau", type=float, default=0.005)
    a("--batch-size", type=int, default=256)
    a("--learning-starts", type=int, default=10_000, help="random-action transitions before learning")
    a("--policy-lr", type=float, default=3e-4)
    a("--q-lr", type=float, default=1e-3)
    a("--policy-frequency", type=int, default=2, help="updates between (delayed) actor updates")
    a("--alpha", type=float, default=0.2, help="entropy weight (initial value with --autotune)")
    a("--autotune", action=argparse.BooleanOptionalAction, default=True)
    a("--hidden", type=int, default=128)
    a("--sim-threads", type=int, default=2)
    a("--torch-threads", type=int, default=2, help="more is slower for these small networks")
    a("--log-every", type=int, default=20_000, help="environment steps between progress lines")
    a("--eval-episodes", type=int, default=64)
    a("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    a("--runs-dir", default="runs")
    a("--env-kwargs", type=json.loads, default={}, help='task options as JSON, e.g. \'{"action_mode": "motors"}\'')
    a(
        "--eval-env-kwargs",
        type=json.loads,
        default={},
        help='task options for the final evaluation on top of --env-kwargs, e.g. unseen maps: \'{"map_seed": 1000}\'',
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------- networks


def mlp(inp: int, out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, a], 1))


class Actor(nn.Module):
    """Tanh-squashed Gaussian; actions in [−1, 1] (the action space of every task)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.fc_mean = nn.Linear(hidden, act_dim)
        self.fc_logstd = nn.Linear(hidden, act_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        log_std = torch.tanh(self.fc_logstd(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return self.fc_mean(h), log_std

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sampled action, its log-probability and the deterministic (mean) action."""
        mean, log_std = self(x)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = (normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)).sum(1, keepdim=True)
        return y_t, log_prob, torch.tanh(mean)


class Policy:
    """Deterministic (tanh of the mean) or sampled actions for raw observations, as numpy."""

    def __init__(self, actor: Actor, obs_norm: ObsNormalizer):
        self.actor = actor
        self.obs_norm = obs_norm

    @torch.no_grad()
    def __call__(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        sample, _, mean = self.actor.get_action(torch.from_numpy(self.obs_norm(obs)))
        return (mean if deterministic else sample).numpy()


def save_policy(path: pathlib.Path, actor: Actor, obs_norm: ObsNormalizer, args: argparse.Namespace, **extra) -> None:
    torch.save(
        {
            "algo": "sac",
            "actor": actor.state_dict(),
            "obs_norm": obs_norm.state_dict(),
            "obs_dim": actor.trunk[0].in_features,
            "act_dim": actor.fc_mean.out_features,
            "hidden": actor.trunk[0].out_features,
            "args": vars(args),
            **extra,
        },
        path,
    )


def load_policy(path: str | pathlib.Path) -> tuple[Policy, dict[str, Any]]:
    ckpt = torch.load(path, weights_only=False)
    actor = Actor(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    norm = ObsNormalizer(ckpt["obs_dim"])
    norm.load_state_dict(ckpt["obs_norm"])
    return Policy(actor, norm), ckpt


# ---------------------------------------------------------------------------- replay buffer


class ReplayBuffer:
    """Ring buffer of raw transitions in numpy arrays."""

    def __init__(self, size: int, obs_dim: int, act_dim: int):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.actions = np.zeros((size, act_dim), np.float32)
        self.rewards = np.zeros(size, np.float32)
        self.terminated = np.zeros(size, np.float32)
        self.size, self.pos, self.full = size, 0, False

    def add(self, obs, next_obs, actions, rewards, terminated) -> None:
        n = len(obs)
        idx = (self.pos + np.arange(n)) % self.size
        self.obs[idx], self.next_obs[idx], self.actions[idx] = obs, next_obs, actions
        self.rewards[idx], self.terminated[idx] = rewards, terminated
        self.full |= self.pos + n >= self.size
        self.pos = (self.pos + n) % self.size

    def __len__(self) -> int:
        return self.size if self.full else self.pos

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        i = rng.integers(0, len(self), n)
        return self.obs[i], self.next_obs[i], self.actions[i], self.rewards[i], self.terminated[i]


# ---------------------------------------------------------------------------- training


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(argv)
    run_name = f"{args.env_id.split('/')[-1]}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = pathlib.Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(run_dir))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    rng = np.random.default_rng(args.seed)

    envs = gym.make_vec(args.env_id, num_envs=args.num_envs, num_threads=args.sim_threads, seed=args.seed, **args.env_kwargs)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    actor = Actor(obs_dim, act_dim, args.hidden)
    qf1, qf2 = SoftQNetwork(obs_dim, act_dim, args.hidden), SoftQNetwork(obs_dim, act_dim, args.hidden)
    qf1_target, qf2_target = SoftQNetwork(obs_dim, act_dim, args.hidden), SoftQNetwork(obs_dim, act_dim, args.hidden)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.policy_lr)
    if args.autotune:
        target_entropy = -float(act_dim)
        log_alpha = torch.tensor(np.log(args.alpha), dtype=torch.float32, requires_grad=True)
        alpha = log_alpha.exp().item()
        a_optimizer = torch.optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    obs_norm = ObsNormalizer(obs_dim)
    rb = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    num_iterations = max(1, args.total_timesteps // args.num_envs)
    print(f"{run_name}: {args.num_envs} worlds, {num_iterations} vector steps × {args.updates_per_step} updates")
    obs, _ = envs.reset(seed=args.seed)
    obs_norm.rms.update(obs)
    start = time.time()
    global_step = 0
    updates = 0
    next_log = args.log_every
    episode_returns: list[float] = []
    episode_lengths: list[float] = []
    episode_success: list[bool] = []
    has_success = envs.unwrapped.task.has_success
    qf_loss = actor_loss = torch.tensor(float("nan"))
    qf1_a_values = torch.tensor(float("nan"))

    for _iteration in range(num_iterations):
        global_step += args.num_envs
        if global_step <= args.learning_starts:
            actions = rng.uniform(-1.0, 1.0, (args.num_envs, act_dim)).astype(np.float32)
        else:
            with torch.no_grad():
                actions = actor.get_action(torch.from_numpy(obs_norm(obs)))[0].numpy()

        next_obs, rewards, terminated, truncated, info = envs.step(actions)
        real_next_obs = next_obs.copy()
        done = terminated | truncated
        if done.any():
            real_next_obs[done] = info["final_obs"][done]
        rb.add(obs, real_next_obs, actions, rewards, terminated)
        obs_norm.rms.update(next_obs)
        obs = next_obs
        if "episode" in info:
            m = info["_episode"]
            episode_returns.extend(info["episode"]["r"][m].tolist())
            episode_lengths.extend(info["episode"]["l"][m].tolist())
            episode_success.extend(info["episode"]["success"][m].tolist())

        if global_step <= args.learning_starts:
            continue
        for _ in range(args.updates_per_step):
            updates += 1
            b_obs, b_next, b_act, b_rew, b_term = rb.sample(args.batch_size, rng)
            b_obs, b_next = torch.from_numpy(obs_norm(b_obs)), torch.from_numpy(obs_norm(b_next))
            b_act, b_rew, b_term = torch.from_numpy(b_act), torch.from_numpy(b_rew), torch.from_numpy(b_term)
            with torch.no_grad():
                next_actions, next_log_pi, _ = actor.get_action(b_next)
                min_q_next = torch.min(qf1_target(b_next, next_actions), qf2_target(b_next, next_actions)) - alpha * next_log_pi
                next_q = b_rew + (1 - b_term) * args.gamma * min_q_next.view(-1)
            qf1_a_values = qf1(b_obs, b_act).view(-1)
            qf2_a_values = qf2(b_obs, b_act).view(-1)
            qf_loss = F.mse_loss(qf1_a_values, next_q) + F.mse_loss(qf2_a_values, next_q)
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if updates % args.policy_frequency == 0:
                # Delayed actor updates, repeated to keep one actor update per critic update.
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(b_obs)
                    min_q_pi = torch.min(qf1(b_obs, pi), qf2(b_obs, pi))
                    actor_loss = (alpha * log_pi - min_q_pi).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    if args.autotune:
                        alpha_loss = (-log_alpha.exp() * (log_pi.detach() + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            with torch.no_grad():
                for net, target in ((qf1, qf1_target), (qf2, qf2_target)):
                    for p, tp in zip(net.parameters(), target.parameters()):
                        tp.mul_(1 - args.tau).add_(args.tau * p)

        if global_step >= next_log or _iteration == num_iterations - 1:
            next_log += args.log_every
            sps = global_step / (time.time() - start)
            recent_r = float(np.mean(episode_returns[-100:])) if episode_returns else float("nan")
            recent_l = float(np.mean(episode_lengths[-100:])) if episode_lengths else float("nan")
            recent_s = float(np.mean(episode_success[-100:])) if episode_success else float("nan")
            print(
                f"step {global_step:9,d}  return {recent_r:7.2f}  length {recent_l:6.1f}  "
                + (f"success {recent_s:4.0%}  " if has_success else "")
                + f"alpha {alpha:.4f}  q_loss {qf_loss.item():.4f}  q {qf1_a_values.mean().item():7.2f}  {sps:,.0f} SPS"
            )
            if writer is not None:
                writer.add_scalar("charts/episodic_return", recent_r, global_step)
                writer.add_scalar("charts/episodic_length", recent_l, global_step)
                if has_success:
                    writer.add_scalar("charts/success_rate", recent_s, global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
            episode_returns, episode_lengths = episode_returns[-100:], episode_lengths[-100:]
            episode_success = episode_success[-100:]

    envs.close()
    policy_path = run_dir / "policy.pt"
    save_policy(policy_path, actor, obs_norm, args)
    eval_kwargs = {**args.env_kwargs, **args.eval_env_kwargs}
    result = evaluate(Policy(actor, obs_norm), args.env_id, args.eval_episodes, args.seed + 1000, eval_kwargs)
    result["sps"] = global_step / (time.time() - start)
    result["minutes"] = (time.time() - start) / 60
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2))
    print(f"saved {policy_path}; evaluation: {json.dumps(result)}")
    if writer is not None:
        for k, v in result.items():
            writer.add_scalar(f"eval/{k}", v, global_step)
        writer.close()
    return result


if __name__ == "__main__":
    main()
