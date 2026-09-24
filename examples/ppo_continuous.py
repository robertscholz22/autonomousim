"""PPO with continuous actions on autonomousim's native vector environments (CleanRL style,
one file).

    make train-deps
    uv run python examples/ppo_continuous.py --env-id autonomousim/QuadHover-v0 --total-timesteps 10000000
    uv run python examples/eval_record.py runs/<run>/policy.pt

Differences from CleanRL's ``ppo_continuous_action.py``:

- All worlds step in one native call (``gym.make_vec``, SAME_STEP autoreset).
- Truncated episodes bootstrap from ``info["final_obs"]``: ``r += γ·V(final_obs)``.
- Observation and reward normalisation (``autonomousim.rl``) are applied in the loop, not by
  wrappers, and the observation statistics are saved with the policy.
- The simulation and torch get separate thread budgets (``--sim-threads``, ``--torch-threads``).
- Optional bounds loss on the action mean (``--bound-coef``, as in rl_games): the environment
  clips actions to [−1, 1], so a mean far outside that range turns every sample into the same
  action. PPO then sees no effect of its exploration noise and cannot learn to back off.

The checkpoint (``runs/<run>/policy.pt``) holds the network, the observation statistics and
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
from torch.distributions.normal import Normal

import autonomousim  # noqa: F401  (registers the environments)
from autonomousim.rl import ObsNormalizer, RewardScaler, evaluate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a = p.add_argument
    a("--env-id", default="autonomousim/QuadHover-v0")
    a("--exp-name", default="ppo")
    a("--seed", type=int, default=1)
    a("--total-timesteps", type=int, default=10_000_000)
    a("--num-envs", type=int, default=256)
    a("--num-steps", type=int, default=64, help="steps per world per rollout")
    a("--learning-rate", type=float, default=3e-4)
    a("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    a("--gamma", type=float, default=0.99)
    a("--gae-lambda", type=float, default=0.95)
    a("--num-minibatches", type=int, default=4)
    a("--update-epochs", type=int, default=5)
    a("--clip-coef", type=float, default=0.2)
    a("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    a("--norm-adv", action=argparse.BooleanOptionalAction, default=True)
    a("--norm-reward", action=argparse.BooleanOptionalAction, default=True)
    a("--ent-coef", type=float, default=0.0)
    a("--bound-coef", type=float, default=0.0, help="weight of the loss on action means beyond ±--mean-bound")
    a("--mean-bound", type=float, default=1.1)
    a("--vf-coef", type=float, default=0.5)
    a("--max-grad-norm", type=float, default=0.5)
    a("--target-kl", type=float, default=None)
    a("--hidden", type=int, default=128)
    a("--init-log-std", type=float, default=-0.5)
    a("--sim-threads", type=int, default=9)
    a("--torch-threads", type=int, default=3)
    a("--log-every", type=int, default=10, help="iterations between progress lines")
    a("--eval-episodes", type=int, default=64)
    a("--save-every", type=int, default=50, help="iterations between checkpoints (0: only at the end)")
    a("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    a("--runs-dir", default="runs")
    a("--env-kwargs", type=json.loads, default={}, help='task options as JSON, e.g. \'{"action_mode": "motors"}\'')
    a(
        "--eval-env-kwargs",
        type=json.loads,
        default={},
        help='task options for the final evaluation on top of --env-kwargs, e.g. unseen maps: \'{"map_seed": 1000}\'',
    )
    args = p.parse_args(argv)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = max(1, args.total_timesteps // args.batch_size)
    return args


# ---------------------------------------------------------------------------- networks


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """Separate 2-layer tanh MLPs for the value and the action mean; state-independent
    log standard deviation."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, init_log_std: float = 0.0):
        super().__init__()

        def mlp(out: int, std: float) -> nn.Sequential:
            return nn.Sequential(
                layer_init(nn.Linear(obs_dim, hidden)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden, hidden)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden, out), std=std),
            )

        self.critic = mlp(1, 1.0)
        self.actor_mean = mlp(act_dim, 0.01)
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), init_log_std))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        """Action (sampled unless given), its log-probability, the entropy, the value and the
        action mean."""
        mean = self.actor_mean(x)
        dist = Normal(mean, self.actor_logstd.expand_as(mean).exp())
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x), mean


class Policy:
    """Deterministic (mean) or sampled actions for raw observations, as numpy arrays."""

    def __init__(self, agent: Agent, obs_norm: ObsNormalizer):
        self.agent = agent
        self.obs_norm = obs_norm

    @torch.no_grad()
    def __call__(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        x = torch.from_numpy(self.obs_norm(obs))
        if deterministic:
            a = self.agent.actor_mean(x)
        else:
            a = self.agent.get_action_and_value(x)[0]
        return a.clamp(-1.0, 1.0).numpy()


def save_policy(path: pathlib.Path, agent: Agent, obs_norm: ObsNormalizer, args: argparse.Namespace, **extra) -> None:
    torch.save(
        {
            "algo": "ppo",
            "agent": agent.state_dict(),
            "obs_norm": obs_norm.state_dict(),
            "obs_dim": agent.critic[0].in_features,
            "act_dim": agent.actor_logstd.shape[1],
            "hidden": agent.critic[0].out_features,
            "args": vars(args),
            **extra,
        },
        path,
    )


def load_policy(path: str | pathlib.Path) -> tuple[Policy, dict[str, Any]]:
    ckpt = torch.load(path, weights_only=False)
    agent = Agent(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"])
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    norm = ObsNormalizer(ckpt["obs_dim"])
    norm.load_state_dict(ckpt["obs_norm"])
    return Policy(agent, norm), ckpt


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

    envs = gym.make_vec(args.env_id, num_envs=args.num_envs, num_threads=args.sim_threads, seed=args.seed, **args.env_kwargs)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    agent = Agent(obs_dim, act_dim, args.hidden, args.init_log_std)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_norm = ObsNormalizer(obs_dim)
    reward_scale = RewardScaler(args.num_envs, args.gamma) if args.norm_reward else None

    shape = (args.num_steps, args.num_envs)
    b_obs = torch.zeros(shape + (obs_dim,))
    b_actions = torch.zeros(shape + (act_dim,))
    b_logprobs = torch.zeros(shape)
    b_rewards = torch.zeros(shape)
    b_dones = torch.zeros(shape)
    b_values = torch.zeros(shape)

    print(f"{run_name}: {args.num_envs} worlds × {args.num_steps} steps, {args.num_iterations} iterations")
    global_step = 0
    start = time.time()
    raw_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.from_numpy(obs_norm(raw_obs, update=True))
    next_done = torch.zeros(args.num_envs)
    episode_returns: list[float] = []
    episode_lengths: list[float] = []
    episode_success: list[bool] = []
    has_success = envs.unwrapped.task.has_success
    policy_path = run_dir / "policy.pt"
    sim_time = 0.0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            b_obs[step] = next_obs
            b_dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
            b_values[step] = value.flatten()
            b_actions[step] = action
            b_logprobs[step] = logprob

            t = time.perf_counter()
            raw_obs, reward, terminated, truncated, info = envs.step(action.clamp(-1.0, 1.0).numpy())
            sim_time += time.perf_counter() - t
            done = terminated | truncated
            if reward_scale is not None:
                reward = reward_scale(reward, done)
            if truncated.any():
                # Bootstrap time limits from the value of the last observation.
                final = torch.from_numpy(obs_norm(info["final_obs"][truncated]))
                with torch.no_grad():
                    reward[truncated] += args.gamma * agent.get_value(final).flatten().numpy()
            b_rewards[step] = torch.from_numpy(reward.astype(np.float32))
            next_obs = torch.from_numpy(obs_norm(raw_obs, update=True))
            next_done = torch.from_numpy(done.astype(np.float32))
            if "episode" in info:
                m = info["_episode"]
                episode_returns.extend(info["episode"]["r"][m].tolist())
                episode_lengths.extend(info["episode"]["l"][m].tolist())
                episode_success.extend(info["episode"]["success"][m].tolist())

        # Generalised advantage estimation.
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(b_rewards)
            last = torch.zeros(args.num_envs)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    not_done, next_v = 1.0 - next_done, next_value
                else:
                    not_done, next_v = 1.0 - b_dones[t + 1], b_values[t + 1]
                delta = b_rewards[t] + args.gamma * next_v * not_done - b_values[t]
                last = delta + args.gamma * args.gae_lambda * not_done * last
                advantages[t] = last
            returns = advantages + b_values

        f_obs = b_obs.reshape(-1, obs_dim)
        f_actions = b_actions.reshape(-1, act_dim)
        f_logprobs = b_logprobs.reshape(-1)
        f_advantages = advantages.reshape(-1)
        f_returns = returns.reshape(-1)
        f_values = b_values.reshape(-1)

        clipfracs = []
        for _epoch in range(args.update_epochs):
            perm = torch.randperm(args.batch_size)
            for s in range(0, args.batch_size, args.minibatch_size):
                mb = perm[s : s + args.minibatch_size]
                _, newlogprob, entropy, newvalue, newmean = agent.get_action_and_value(f_obs[mb], f_actions[mb])
                logratio = newlogprob - f_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                mb_adv = f_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_clipped = f_values[mb] + (newvalue - f_values[mb]).clamp(-args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max((newvalue - f_returns[mb]) ** 2, (v_clipped - f_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - f_returns[mb]) ** 2).mean()
                excess = (newmean.abs() - args.mean_bound).clamp(min=0.0)
                b_loss = (excess**2).sum(1).mean()
                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef + args.bound_coef * b_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        sps = global_step / (time.time() - start)
        recent_r = float(np.mean(episode_returns[-200:])) if episode_returns else float("nan")
        recent_l = float(np.mean(episode_lengths[-200:])) if episode_lengths else float("nan")
        recent_s = float(np.mean(episode_success[-200:])) if episode_success else float("nan")
        if writer is not None:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/episodic_return", recent_r, global_step)
            writer.add_scalar("charts/episodic_length", recent_l, global_step)
            if has_success:
                writer.add_scalar("charts/success_rate", recent_s, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/bound_loss", b_loss.item(), global_step)
            writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
            writer.add_scalar("policy/std", agent.actor_logstd.exp().mean().item(), global_step)
        if iteration % args.log_every == 0 or iteration == args.num_iterations:
            print(
                f"it {iteration:4d}  step {global_step:9,d}  return {recent_r:7.2f}  length {recent_l:6.1f}  "
                + (f"success {recent_s:4.0%}  " if has_success else "")
                + f"std {agent.actor_logstd.exp().mean().item():.3f}  v_loss {v_loss.item():.4f}  "
                f"kl {approx_kl.item():.4f}  {sps:,.0f} SPS (sim {sim_time / (time.time() - start):.0%})"
            )
            episode_returns, episode_lengths = episode_returns[-200:], episode_lengths[-200:]
            episode_success = episode_success[-200:]
        if args.save_every and iteration % args.save_every == 0:
            save_policy(policy_path, agent, obs_norm, args, global_step=global_step)

    envs.close()
    save_policy(policy_path, agent, obs_norm, args, global_step=global_step)
    eval_kwargs = {**args.env_kwargs, **args.eval_env_kwargs}
    result = evaluate(Policy(agent, obs_norm), args.env_id, args.eval_episodes, args.seed + 1000, eval_kwargs)
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
