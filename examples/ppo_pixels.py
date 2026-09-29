"""PPO from pixels: a small CNN encoder for the camera image plus the state vector (CleanRL
style, one file).

    make train-deps
    uv run python examples/ppo_pixels.py --env-id autonomousim/QuadHoverPad-v0 --total-timesteps 1000000

For tasks with camera observations (``Dict`` spaces ``{"state", "image"}``; see
``autonomousim.vector_env``). The image (``uint8 [H, W, C]``) goes through a stack of 3×3
convolutions with stride 2 and ReLU, is flattened and projected to ``--features`` values
(ReLU), then concatenated with the normalised state and fed to separate tanh MLP heads for
the action mean and the value. The encoder is shared by both heads. Image channels are scaled
to [0, 1]; the state is normalised with running statistics as in ``ppo_continuous.py``.

Everything else follows ``ppo_continuous.py`` (bootstrapping truncated episodes from
``final_obs``, reward scaling, separate thread budgets). Cameras render on the GPU chosen by
``AUTONOMOUSIM_RENDER_ADAPTER`` (default: the best available). The checkpoint
(``runs/<run>/policy.pt``) holds the network, the state statistics and the arguments;
``load_policy`` rebuilds a numpy-in, numpy-out policy that takes the ``Dict`` observations.
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
    a("--env-id", default="autonomousim/QuadHoverPad-v0")
    a("--exp-name", default="ppo_pixels")
    a("--seed", type=int, default=1)
    a("--total-timesteps", type=int, default=1_000_000)
    a("--num-envs", type=int, default=64)
    a("--num-steps", type=int, default=64, help="steps per world per rollout")
    a("--learning-rate", type=float, default=3e-4)
    a("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    a("--gamma", type=float, default=0.99)
    a("--gae-lambda", type=float, default=0.95)
    a("--num-minibatches", type=int, default=4)
    a("--update-epochs", type=int, default=4)
    a("--clip-coef", type=float, default=0.2)
    a("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    a("--norm-adv", action=argparse.BooleanOptionalAction, default=True)
    a("--norm-reward", action=argparse.BooleanOptionalAction, default=True)
    a("--ent-coef", type=float, default=0.0)
    a("--vf-coef", type=float, default=0.5)
    a("--max-grad-norm", type=float, default=0.5)
    a("--target-kl", type=float, default=None)
    a("--channels", type=json.loads, default=[16, 32, 32], help="output channels of the 3×3 stride-2 convolutions")
    a("--features", type=int, default=128, help="size of the image features")
    a("--hidden", type=int, default=128)
    a("--init-log-std", type=float, default=-0.5)
    a("--init", default=None, help="policy.pt to continue from (weights and state statistics)")
    a("--sim-threads", type=int, default=4)
    a("--torch-threads", type=int, default=4)
    a("--log-every", type=int, default=5, help="iterations between progress lines")
    a("--eval-episodes", type=int, default=64)
    a("--save-every", type=int, default=50, help="iterations between checkpoints (0: only at the end)")
    a("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    a("--runs-dir", default="runs")
    a("--env-kwargs", type=json.loads, default={}, help='task options as JSON, e.g. \'{"depth": true}\'')
    a("--eval-env-kwargs", type=json.loads, default={}, help="task options for the final evaluation on top of --env-kwargs")
    args = p.parse_args(argv)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = max(1, args.total_timesteps // args.batch_size)
    return args


# ---------------------------------------------------------------------------- networks


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Encoder(nn.Module):
    """``uint8 [N, H, W, C]`` images → ``[N, features]``: 3×3 convolutions with stride 2 and
    padding 1, each followed by ReLU, then a linear layer and ReLU."""

    def __init__(self, image_shape: tuple[int, int, int], channels: list[int], features: int):
        super().__init__()
        h, w, c = image_shape
        layers: list[nn.Module] = []
        for out in channels:
            layers += [layer_init(nn.Conv2d(c, out, 3, stride=2, padding=1)), nn.ReLU()]
            c, h, w = out, (h + 1) // 2, (w + 1) // 2
        self.conv = nn.Sequential(*layers, nn.Flatten())
        self.fc = nn.Sequential(layer_init(nn.Linear(c * h * w, features)), nn.ReLU())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image.permute(0, 3, 1, 2).float() / 255.0
        return self.fc(self.conv(x))


class Agent(nn.Module):
    """A shared image encoder; separate 2-layer tanh MLPs over (image features, state) for the
    value and the action mean; state-independent log standard deviation."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        state_dim: int,
        act_dim: int,
        channels: list[int],
        features: int = 128,
        hidden: int = 128,
        init_log_std: float = 0.0,
    ):
        super().__init__()
        self.image_shape, self.state_dim = tuple(image_shape), state_dim
        self.encoder = Encoder(image_shape, channels, features)
        n = features + state_dim

        def mlp(out: int, std: float) -> nn.Sequential:
            return nn.Sequential(
                layer_init(nn.Linear(n, hidden)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden, hidden)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden, out), std=std),
            )

        self.critic = mlp(1, 1.0)
        self.actor_mean = mlp(act_dim, 0.01)
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), init_log_std))

    def features(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.encoder(image), state], dim=1)

    def get_value(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.critic(self.features(image, state))

    def get_action_and_value(self, image: torch.Tensor, state: torch.Tensor, action: torch.Tensor | None = None):
        """Action (sampled unless given), its log-probability, the entropy, the value and the
        action mean."""
        x = self.features(image, state)
        mean = self.actor_mean(x)
        dist = Normal(mean, self.actor_logstd.expand_as(mean).exp())
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x), mean


class Policy:
    """Deterministic (mean) or sampled actions for raw ``Dict`` observations, as numpy arrays."""

    def __init__(self, agent: Agent, obs_norm: ObsNormalizer):
        self.agent = agent
        self.obs_norm = obs_norm

    @torch.no_grad()
    def __call__(self, obs: dict[str, np.ndarray], deterministic: bool = True) -> np.ndarray:
        image = torch.from_numpy(np.ascontiguousarray(obs["image"]))
        state = torch.from_numpy(self.obs_norm(obs["state"]))
        if deterministic:
            a = self.agent.actor_mean(self.agent.features(image, state))
        else:
            a = self.agent.get_action_and_value(image, state)[0]
        return a.clamp(-1.0, 1.0).numpy()


def save_policy(path: pathlib.Path, agent: Agent, obs_norm: ObsNormalizer, args: argparse.Namespace, **extra) -> None:
    torch.save(
        {
            "algo": "ppo_pixels",
            "agent": agent.state_dict(),
            "obs_norm": obs_norm.state_dict(),
            "image_shape": list(agent.image_shape),
            "state_dim": agent.state_dim,
            "act_dim": agent.actor_logstd.shape[1],
            "channels": args.channels,
            "features": args.features,
            "hidden": args.hidden,
            "args": vars(args),
            **extra,
        },
        path,
    )


def load_policy(path: str | pathlib.Path) -> tuple[Policy, dict[str, Any]]:
    ckpt = torch.load(path, weights_only=False)
    agent = Agent(
        tuple(ckpt["image_shape"]), ckpt["state_dim"], ckpt["act_dim"], ckpt["channels"], ckpt["features"], ckpt["hidden"]
    )
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    norm = ObsNormalizer(ckpt["state_dim"])
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
    space = envs.single_observation_space
    if not isinstance(space, gym.spaces.Dict):
        raise SystemExit(f"{args.env_id} has no camera observations; use ppo_continuous.py")
    image_shape = space["image"].shape
    state_dim = space["state"].shape[0]
    act_dim = envs.single_action_space.shape[0]
    agent = Agent(image_shape, state_dim, act_dim, args.channels, args.features, args.hidden, args.init_log_std)
    obs_norm = ObsNormalizer(state_dim)
    if args.init:
        ckpt = torch.load(args.init, weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        obs_norm.load_state_dict(ckpt["obs_norm"])
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    reward_scale = RewardScaler(args.num_envs, args.gamma) if args.norm_reward else None

    shape = (args.num_steps, args.num_envs)
    b_images = torch.zeros(shape + tuple(image_shape), dtype=torch.uint8)
    b_states = torch.zeros(shape + (state_dim,))
    b_actions = torch.zeros(shape + (act_dim,))
    b_logprobs = torch.zeros(shape)
    b_rewards = torch.zeros(shape)
    b_dones = torch.zeros(shape)
    b_values = torch.zeros(shape)

    params = sum(p.numel() for p in agent.parameters())
    print(
        f"{run_name}: {args.num_envs} worlds × {args.num_steps} steps, {args.num_iterations} iterations; "
        f"image {tuple(image_shape)}, state {state_dim}, {params:,} parameters; "
        f"adapter {autonomousim.render_adapter()}"
    )
    global_step = 0
    start = time.time()
    raw, _ = envs.reset(seed=args.seed)
    next_image = torch.from_numpy(raw["image"])
    next_state = torch.from_numpy(obs_norm(raw["state"], update=True))
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
            b_images[step] = next_image
            b_states[step] = next_state
            b_dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(next_image, next_state)
            b_values[step] = value.flatten()
            b_actions[step] = action
            b_logprobs[step] = logprob

            t = time.perf_counter()
            raw, reward, terminated, truncated, info = envs.step(action.clamp(-1.0, 1.0).numpy())
            sim_time += time.perf_counter() - t
            done = terminated | truncated
            if reward_scale is not None:
                reward = reward_scale(reward, done)
            if truncated.any():
                # Bootstrap time limits from the value of the last observation.
                final = info["final_obs"]
                image = torch.from_numpy(final["image"][truncated])
                state = torch.from_numpy(obs_norm(final["state"][truncated]))
                with torch.no_grad():
                    reward[truncated] += args.gamma * agent.get_value(image, state).flatten().numpy()
            b_rewards[step] = torch.from_numpy(reward.astype(np.float32))
            next_image = torch.from_numpy(raw["image"])
            next_state = torch.from_numpy(obs_norm(raw["state"], update=True))
            next_done = torch.from_numpy(done.astype(np.float32))
            if "episode" in info:
                m = info["_episode"]
                episode_returns.extend(info["episode"]["r"][m].tolist())
                episode_lengths.extend(info["episode"]["l"][m].tolist())
                episode_success.extend(info["episode"]["success"][m].tolist())

        # Generalised advantage estimation.
        with torch.no_grad():
            next_value = agent.get_value(next_image, next_state).flatten()
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

        f_images = b_images.reshape(-1, *image_shape)
        f_states = b_states.reshape(-1, state_dim)
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
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(f_images[mb], f_states[mb], f_actions[mb])
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
                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef
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
            writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
            writer.add_scalar("policy/std", agent.actor_logstd.exp().mean().item(), global_step)
        if iteration % args.log_every == 0 or iteration == args.num_iterations:
            print(
                f"it {iteration:4d}  step {global_step:9,d}  return {recent_r:7.2f}  length {recent_l:6.1f}  "
                + (f"success {recent_s:4.0%}  " if has_success else "")
                + f"std {agent.actor_logstd.exp().mean().item():.3f}  v_loss {v_loss.item():.4f}  "
                f"kl {approx_kl.item():.4f}  {sps:,.0f} SPS (sim {sim_time / (time.time() - start):.0%})",
                flush=True,
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
