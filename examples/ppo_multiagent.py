"""Independent PPO with parameter sharing (IPPO) on autonomousim's multi-agent vector
environment (CleanRL style, one file).

    make train-deps
    uv run python examples/ppo_multiagent.py --task swarm_hover --total-timesteps 20000000

Every agent group has one policy shared by its agents; every (world, agent) slot is a
separate sample stream. Differences from ``ppo_continuous.py``, whose network, observation
normalisation and checkpoint format are reused (so a group's policy runs wherever a
single-agent policy does):

- An agent that stops (see ``autonomousim.tasks.multi``) stays out of the batch until its
  world starts a new episode: its slot's samples are masked out of the loss and the
  normalisation statistics.
- GAE runs per slot. A slot's chain ends when its agent stops or its world's episode ends;
  agents cut off by the time limit bootstrap from ``info["final_obs"]``.
- ``--total-timesteps`` counts agent steps (the samples PPO trains on).
- The evaluation reports per-agent returns, success, and collision rates, plus the
  formation error for tasks that define ``formation_error`` (``swarm_hover``).

Checkpoints: ``runs/<run>/policy_<group>.pt`` per group, and ``policy.pt`` too when there
is only one group.
"""

import argparse
import json
import pathlib
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ppo_continuous import Agent, Policy, save_policy  # noqa: E402

from autonomousim.events import Event  # noqa: E402
from autonomousim.multiagent import MultiAgentVectorEnv  # noqa: E402
from autonomousim.rl import ObsNormalizer, RewardScaler  # noqa: E402
from autonomousim.scenario import STATE  # noqa: E402
from autonomousim.tasks.multi import make_multi_task  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a = p.add_argument
    a("--task", default="swarm_hover", help="registered multi-agent task")
    a("--exp-name", default="ippo")
    a("--seed", type=int, default=1)
    a("--total-timesteps", type=int, default=20_000_000, help="agent steps")
    a("--num-envs", type=int, default=64, help="worlds")
    a("--num-steps", type=int, default=64, help="steps per world per rollout")
    a("--learning-rate", type=float, default=3e-4)
    a("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    a("--gamma", type=float, default=0.99)
    a("--gae-lambda", type=float, default=0.95)
    a("--num-minibatches", type=int, default=8)
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
    a("--eval-episodes", type=int, default=64, help="evaluation episodes (worlds)")
    a("--save-every", type=int, default=50, help="iterations between checkpoints (0: only at the end)")
    a("--time-limit", type=float, default=0.0, help="stop training after this many minutes (0: none)")
    a("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    a("--runs-dir", default="runs")
    a("--task-kwargs", type=json.loads, default={}, help='task options as JSON, e.g. \'{"count": 16}\'')
    a("--eval-task-kwargs", type=json.loads, default={}, help="task options for the evaluation on top of --task-kwargs")
    return p.parse_args(argv)


class Group:
    """Policy, normalisation and rollout buffers of one agent group (``slots`` = worlds ×
    agents)."""

    def __init__(self, name: str, envs: MultiAgentVectorEnv, args: argparse.Namespace):
        self.name = name
        self.slots = envs.num_envs * envs.count[name]
        self.obs_dim, self.act_dim = envs.obs_dim[name], envs.act_dim[name]
        self.agent = Agent(self.obs_dim, self.act_dim, args.hidden, args.init_log_std)
        self.obs_norm = ObsNormalizer(self.obs_dim)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)
        self.reward_scale = RewardScaler(self.slots, args.gamma) if args.norm_reward else None
        shape = (args.num_steps, self.slots)
        self.obs = torch.zeros(shape + (self.obs_dim,))
        self.actions = torch.zeros(shape + (self.act_dim,))
        self.logprobs = torch.zeros(shape)
        self.rewards = torch.zeros(shape)
        self.values = torch.zeros(shape)
        self.dones = torch.zeros(shape)  # the slot's chain ends after this step
        self.valid = torch.zeros(shape, dtype=torch.bool)  # the agent was active
        self.returns: list[float] = []
        self.success: list[bool] = []

    def normalise(self, obs: np.ndarray, active: np.ndarray) -> torch.Tensor:
        flat = obs.reshape(self.slots, -1)
        self.obs_norm.rms.update(flat[active.reshape(-1)])
        return torch.from_numpy(self.obs_norm(flat))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    run_name = f"{args.task}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    envs = MultiAgentVectorEnv(args.num_envs, args.task, seed=args.seed, num_threads=args.sim_threads, **args.task_kwargs)
    groups = {g: Group(g, envs, args) for g in envs.groups}
    has_success = envs.task.has_success
    n = args.num_envs
    # Iterations as if every agent stayed active throughout.
    args.num_iterations = max(1, args.total_timesteps // (n * sum(envs.count.values()) * args.num_steps))
    print(f"{run_name}: {n} worlds × {envs.count} agents × {args.num_steps} steps, {args.num_iterations} iterations", flush=True)

    agent_steps = 0
    start = time.time()
    sim_time = 0.0
    raw_obs, info = envs.reset(seed=args.seed)
    active = info["active"]
    next_obs = {g: grp.normalise(raw_obs[g], active[g]) for g, grp in groups.items()}
    lengths: list[float] = []
    no_world_done = np.zeros(n, dtype=bool)

    for iteration in range(1, args.num_iterations + 1):
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        for grp in groups.values():
            if args.anneal_lr:
                grp.optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            actions = {}
            for g, grp in groups.items():
                grp.obs[step] = next_obs[g]
                grp.valid[step] = torch.from_numpy(active[g].reshape(-1))
                with torch.no_grad():
                    action, logprob, _, value, _ = grp.agent.get_action_and_value(next_obs[g])
                grp.values[step] = value.flatten()
                grp.actions[step] = action
                grp.logprobs[step] = logprob
                actions[g] = action.clamp(-1.0, 1.0).numpy().reshape(n, envs.count[g], -1)
                agent_steps += int(active[g].sum())

            t = time.perf_counter()
            raw_obs, reward, terminated, truncated, info = envs.step(actions)
            sim_time += time.perf_counter() - t
            world_done = info.get("_episode", no_world_done)
            for g, grp in groups.items():
                was = active[g]
                done = terminated[g] | world_done[:, None]
                r = reward[g].reshape(-1)
                if grp.reward_scale is not None:
                    r = grp.reward_scale(r, done.reshape(-1), was.reshape(-1))
                cut = (truncated[:, None] & was & ~terminated[g]).reshape(-1)
                if cut.any():
                    # Bootstrap agents stopped by the time limit from their last observation.
                    final = info["final_obs"][g].reshape(grp.slots, -1)[cut]
                    with torch.no_grad():
                        r[cut] += args.gamma * grp.agent.get_value(torch.from_numpy(grp.obs_norm(final))).flatten().numpy()
                grp.rewards[step] = torch.from_numpy(np.where(was.reshape(-1), r, 0.0).astype(np.float32))
                grp.dones[step] = torch.from_numpy(done.reshape(-1).astype(np.float32))
            active = {g: (info["active"][g] & ~terminated[g]) | world_done[:, None] for g in groups}
            next_obs = {g: grp.normalise(raw_obs[g], active[g]) for g, grp in groups.items()}
            if world_done.any():
                lengths.extend(info["episode"]["l"][world_done].tolist())
                for g, grp in groups.items():
                    grp.returns.extend(info["episode"]["r"][g][world_done].ravel().tolist())
                    grp.success.extend(info["episode"]["success"][g][world_done].ravel().tolist())

        stats = {}
        for g, grp in groups.items():
            stats[g] = update(grp, next_obs[g], args)

        elapsed = time.time() - start
        sps = agent_steps / elapsed
        recent_l = float(np.mean(lengths[-200:])) if lengths else float("nan")
        lengths = lengths[-200:]
        line = [f"it {iteration:4d}  agent steps {agent_steps:11,d}  length {recent_l:6.1f}"]
        for g, grp in groups.items():
            recent_r = float(np.mean(grp.returns[-1000:])) if grp.returns else float("nan")
            recent_s = float(np.mean(grp.success[-1000:])) if grp.success else float("nan")
            grp.returns, grp.success = grp.returns[-1000:], grp.success[-1000:]
            s = stats[g]
            line.append(
                f"{g}: return {recent_r:7.2f}  " + (f"success {recent_s:4.0%}  " if has_success else "")
                + f"std {s['std']:.3f}  v_loss {s['value_loss']:.4f}  kl {s['approx_kl']:.4f}"
            )
            if writer is not None:
                writer.add_scalar(f"{g}/episodic_return", recent_r, agent_steps)
                if has_success:
                    writer.add_scalar(f"{g}/success_rate", recent_s, agent_steps)
                for k, v in s.items():
                    writer.add_scalar(f"{g}/{k}", v, agent_steps)
        if writer is not None:
            writer.add_scalar("charts/episodic_length", recent_l, agent_steps)
            writer.add_scalar("charts/SPS", sps, agent_steps)
        out_of_time = args.time_limit > 0 and elapsed > 60 * args.time_limit
        if iteration % args.log_every == 0 or iteration == args.num_iterations or out_of_time:
            line.append(f"{sps:,.0f} agent SPS (sim {sim_time / elapsed:.0%})")
            print("  ".join(line), flush=True)
        if args.save_every and iteration % args.save_every == 0:
            save_all(run_dir, groups, args, agent_steps)
        if out_of_time:
            print(f"time limit of {args.time_limit} min reached")
            break

    envs.close()
    save_all(run_dir, groups, args, agent_steps)
    policies = {g: Policy(grp.agent, grp.obs_norm) for g, grp in groups.items()}
    task_kwargs = {**args.task_kwargs, **args.eval_task_kwargs}
    result = evaluate_multi(policies, args.task, args.eval_episodes, args.seed + 1000, task_kwargs)
    result["agent_sps"] = agent_steps / (time.time() - start)
    result["minutes"] = (time.time() - start) / 60
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2))
    print(f"saved {run_dir}; evaluation: {json.dumps(result)}")
    if writer is not None:
        for k, v in result.items():
            if isinstance(v, int | float):
                writer.add_scalar(f"eval/{k}", v, agent_steps)
        writer.close()
    return result


def update(grp: Group, next_obs: torch.Tensor, args: argparse.Namespace) -> dict[str, float]:
    """GAE over the group's slots, then PPO epochs over its valid samples."""
    with torch.no_grad():
        next_value = grp.agent.get_value(next_obs).flatten()
        advantages = torch.zeros_like(grp.rewards)
        last = torch.zeros(grp.slots)
        for t in reversed(range(args.num_steps)):
            next_v = next_value if t == args.num_steps - 1 else grp.values[t + 1]
            not_done = 1.0 - grp.dones[t]
            delta = grp.rewards[t] + args.gamma * next_v * not_done - grp.values[t]
            last = delta + args.gamma * args.gae_lambda * not_done * last
            advantages[t] = last
        returns = advantages + grp.values

    valid = grp.valid.reshape(-1)
    f_obs = grp.obs.reshape(-1, grp.obs_dim)[valid]
    f_actions = grp.actions.reshape(-1, grp.act_dim)[valid]
    f_logprobs = grp.logprobs.reshape(-1)[valid]
    f_advantages = advantages.reshape(-1)[valid]
    f_returns = returns.reshape(-1)[valid]
    f_values = grp.values.reshape(-1)[valid]
    batch = len(f_obs)
    minibatch = max(1, batch // args.num_minibatches)

    agent, optimizer = grp.agent, grp.optimizer
    clipfracs = []
    approx_kl = pg_loss = v_loss = b_loss = entropy = torch.zeros(())
    for _epoch in range(args.update_epochs):
        perm = torch.randperm(batch)
        for s in range(0, batch, minibatch):
            mb = perm[s : s + minibatch]
            _, newlogprob, entropy, newvalue, newmean = agent.get_action_and_value(f_obs[mb], f_actions[mb])
            logratio = newlogprob - f_logprobs[mb]
            ratio = logratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
            mb_adv = f_advantages[mb]
            if args.norm_adv and len(mb) > 1:
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
    return {
        "value_loss": v_loss.item(),
        "policy_loss": pg_loss.item(),
        "entropy": entropy.mean().item(),
        "approx_kl": approx_kl.item(),
        "bound_loss": b_loss.item(),
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        "std": agent.actor_logstd.exp().mean().item(),
        "samples": float(batch),
    }


def save_all(run_dir: pathlib.Path, groups: dict[str, Group], args: argparse.Namespace, agent_steps: int) -> None:
    for g, grp in groups.items():
        save_policy(run_dir / f"policy_{g}.pt", grp.agent, grp.obs_norm, args, global_step=agent_steps, group=g)
    if len(groups) == 1:
        grp = next(iter(groups.values()))
        save_policy(run_dir / "policy.pt", grp.agent, grp.obs_norm, args, global_step=agent_steps, group=grp.name)


def evaluate_multi(
    policies: dict[str, Any], task: str, episodes: int = 64, seed: int = 1000, task_kwargs: dict[str, Any] | None = None,
    num_threads: int = 4,
) -> dict[str, Any]:
    """Run one deterministic episode in each of ``episodes`` worlds. Per group: mean return
    and success per agent, and the fraction of agents that touched another agent (a
    ``CRASH_AGENT`` event or zero ``agent_clearance``). For tasks with ``formation_error``:
    the mean and largest slot error at the end of each episode, and the fraction of episodes
    that end with a mean error below 0.3 m and no contact (``formation_success``)."""
    envs = MultiAgentVectorEnv(episodes, make_multi_task(task, **(task_kwargs or {})), seed=seed, num_threads=num_threads, autoreset=False)
    obs, info = envs.reset(seed=seed)
    running = np.ones(episodes, dtype=bool)
    ret = {g: np.zeros((episodes, envs.count[g])) for g in envs.groups}
    success = {g: np.zeros((episodes, envs.count[g]), dtype=bool) for g in envs.groups}
    touched = {g: np.zeros((episodes, envs.count[g]), dtype=bool) for g in envs.groups}
    final_state = {g: np.zeros_like(envs.state[g]) for g in envs.groups}
    while running.any():
        actions = {g: policies[g](obs[g].reshape(-1, envs.obs_dim[g])).reshape(episodes, envs.count[g], -1) for g in envs.groups}
        obs, reward, terminated, truncated, info = envs.step(actions)
        for g in envs.groups:
            live = info["active"][g] & running[:, None]
            ret[g] += np.where(live, reward[g], 0.0)
            contact = ((info["events"][g] & Event.CRASH_AGENT) != 0) | (envs.state[g][..., STATE["agent_clearance"]][..., 0] <= 0.0)
            touched[g] |= contact & live
        ended = running & info.get("_episode", np.zeros(episodes, dtype=bool))
        if ended.any():
            for g in envs.groups:
                final_state[g][ended] = envs.state[g][ended]
                success[g][ended] = info["episode"]["success"][g][ended]
            running &= ~ended
    envs.close()
    out: dict[str, Any] = {"episodes": episodes}
    for g in envs.groups:
        prefix = f"{g}/" if len(envs.groups) > 1 else ""
        out[f"{prefix}return"] = float(ret[g].mean())
        out[f"{prefix}success"] = float(success[g].mean())
        out[f"{prefix}agent_contact_rate"] = float(touched[g].mean())
        out[f"{prefix}episodes_without_contact"] = float((~touched[g].any(axis=1)).mean())
        if hasattr(envs.task, "formation_error"):
            err = envs.task.formation_error(final_state[g])
            out[f"{prefix}formation_error_mean"] = float(err.mean())
            out[f"{prefix}formation_error_max"] = float(err.max())
            ok = (err.mean(axis=1) < 0.3) & ~touched[g].any(axis=1)
            out[f"{prefix}formation_success"] = float(ok.mean())
    return out


if __name__ == "__main__":
    main()
