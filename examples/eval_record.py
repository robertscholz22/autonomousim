"""Evaluate a trained policy, record its episodes to MCAP and check that the recording plays
back.

    uv run python examples/eval_record.py runs/<run>/policy.pt --episodes 5
    cargo run -p autonomousim-viewer --release -- replay recordings/<run>.mcap

The policy flies ``--episodes`` episodes in one world with a recorder attached. The file is
then read back and checked in two ways:

1. **Read-back**: the recorded states equal the states seen live, bit for bit.
2. **Replay**: a new simulation built only from the file (the scenario in ``/meta``, the map
   hashes, the episode seeds and the recorded actions) reproduces every recorded state bit
   for bit. This is what the viewer's replay relies on.

Recordings open in Foxglove Studio as well (``/agent/<id>/pose`` is a ``foxglove.PoseInFrame``).
"""

import argparse
import importlib
import json
import pathlib
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode
from mcap.reader import make_reader

from autonomousim import STATE, BatchSim, events

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

SCRIPTS = {"ppo": "ppo_continuous", "sac": "sac_continuous"}


def load_policy(path: pathlib.Path):
    """The policy of a checkpoint, rebuilt by the training script that wrote it."""
    algo = torch.load(path, weights_only=False)["algo"]
    return importlib.import_module(SCRIPTS[algo]).load_policy(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("policy", type=pathlib.Path, help="policy.pt written by ppo_continuous.py or sac_continuous.py")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="episode k uses seed + k")
    p.add_argument("--out", type=pathlib.Path, default=None, help="default: recordings/<run>.mcap")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    p.add_argument("--lidar", action="store_true", help="also record LiDAR scans")
    p.add_argument("--env-kwargs", type=json.loads, default=None, help="task options (default: the training ones)")
    return p.parse_args(argv)


def run_episodes(policy, env_id: str, env_kwargs: dict[str, Any], args: argparse.Namespace, path: pathlib.Path):
    """Fly the episodes with a recorder on world 0; returns per-episode results and the live
    state rows (one after each reset and one per step)."""
    envs = gym.make_vec(env_id, num_envs=1, num_threads=1, autoreset_mode=AutoresetMode.DISABLED, **env_kwargs)
    sim = envs.unwrapped.sim
    results, live = [], []
    for k in range(args.episodes):
        obs, _ = envs.reset(seed=args.seed + k)
        if k == 0:
            # One state per policy step (writes /meta and the first state).
            sim.attach_recorder(0, str(path), state_hz=round(1.0 / sim.policy_dt), lidar=args.lidar)
        states = [envs.unwrapped.state[0].copy()]
        ret, seen = 0.0, 0
        while True:
            obs, reward, terminated, truncated, info = envs.step(policy(obs, deterministic=not args.stochastic))
            states.append(envs.unwrapped.state[0].copy())
            ret += float(reward[0])
            seen |= int(info["events"][0])
            if terminated[0] or truncated[0]:
                break
        s = states[-1]
        success = bool(envs.unwrapped.task.success[0])
        results.append(
            {
                "seed": args.seed + k,
                "return": ret,
                "steps": len(states) - 1,
                "outcome": "success" if success else "truncated" if truncated[0] else "terminated",
                "final_error_m": float(np.linalg.norm(s[STATE["goal"]] - s[STATE["position"]])),
                "events": events.names(seen),
            }
        )
        live.append(np.array(states))
    if not sim.detach_recorder(0):
        raise RuntimeError("no recorder was attached")
    map_hashes = sim.map_hashes
    envs.close()
    return results, live, map_hashes


def read_recording(path: pathlib.Path) -> dict[str, Any]:
    """Messages in file (write) order, split into episodes at each ``/episode`` message."""
    rec: dict[str, Any] = {"meta": None, "episodes": [], "counts": {}}
    with open(path, "rb") as f:
        for _schema, channel, message in make_reader(f).iter_messages(log_time_order=False):
            rec["counts"][channel.topic] = rec["counts"].get(channel.topic, 0) + 1
            data = json.loads(message.data)
            if channel.topic == "/meta":
                rec["meta"] = data
            elif channel.topic == "/episode":
                rec["episodes"].append({"info": data, "states": [], "actions": []})
            elif channel.topic == "/agent/0/state":
                rec["episodes"][-1]["states"].append(data)
            elif channel.topic == "/agent/0/action":
                rec["episodes"][-1]["actions"].append(data["action"])
    return rec


def state_rows(msgs: list[dict[str, Any]]) -> np.ndarray:
    """Position, orientation, velocity and rates of recorded state messages."""
    return np.array([m["position"] + m["orientation"] + m["velocity"] + m["rates"] for m in msgs])


def live_rows(states: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [states[:, STATE["position"]], states[:, STATE["orientation"]], states[:, STATE["velocity"]], states[:, STATE["rates"]]],
        axis=1,
    )


def replay(rec: dict[str, Any], seeds: list[int]) -> list[np.ndarray]:
    """Re-simulate every episode from the recording alone: scenario, seeds and actions."""
    meta = rec["meta"]
    sim = BatchSim(json.dumps(meta["scenario"]), 1, num_threads=1)
    hashes = [m["hash"] for m in meta["maps"]]
    if sim.map_hashes != hashes:
        raise AssertionError(f"rebuilt maps differ from the recording: {sim.map_hashes} != {hashes}")
    out = []
    for ep, seed in zip(rec["episodes"], seeds):
        sim.reset(seeds=[seed])
        rows = [sim.state(0)[0, 0].copy()]
        for a in ep["actions"]:
            sim.step(np.asarray(a, np.float64).reshape(1, 1, -1))
            rows.append(sim.state(0)[0, 0].copy())
        out.append(np.array(rows))
    sim.close()
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    policy, ckpt = load_policy(args.policy)
    train_args = ckpt["args"]
    env_id = train_args["env_id"]
    env_kwargs = train_args.get("env_kwargs", {}) if args.env_kwargs is None else args.env_kwargs
    path = args.out or pathlib.Path("recordings") / f"{args.policy.resolve().parent.name}.mcap"
    path.parent.mkdir(parents=True, exist_ok=True)

    results, live, map_hashes = run_episodes(policy, env_id, env_kwargs, args, path)
    options = f" {json.dumps(env_kwargs)}" if env_kwargs else ""
    print(f"{env_id}{options}, {'stochastic' if args.stochastic else 'deterministic'} policy")
    print(f"{'seed':>5} {'return':>8} {'steps':>6} {'outcome':>11} {'error m':>8}  events")
    for r in results:
        print(
            f"{r['seed']:5d} {r['return']:8.2f} {r['steps']:6d} {r['outcome']:>11} "
            f"{r['final_error_m']:8.3f}  {','.join(r['events']) or '-'}"
        )

    rec = read_recording(path)
    size = path.stat().st_size
    seconds = sum(r["steps"] for r in results) / rec["meta"]["policy_hz"]
    print(f"\nwrote {path} ({size / 1024:.0f} KiB, {seconds:.1f} s simulated, {size / 1024 / seconds:.1f} KiB/s)")
    print("  " + ", ".join(f"{t} {n}" for t, n in sorted(rec["counts"].items())))

    # 1. Read-back: the file holds exactly what was simulated.
    assert [m["hash"] for m in rec["meta"]["maps"]] == map_hashes, "map hashes"
    assert len(rec["episodes"]) == args.episodes, f"{len(rec['episodes'])} episodes recorded"
    for k, (ep, states) in enumerate(zip(rec["episodes"], live)):
        recorded = state_rows(ep["states"])
        if recorded.shape != (len(states), 13) or not np.array_equal(recorded, live_rows(states)):
            raise AssertionError(f"episode {k}: recorded states differ from the live ones")
        if len(ep["actions"]) != len(states) - 1:
            raise AssertionError(f"episode {k}: {len(ep['actions'])} actions for {len(states) - 1} steps")
    print(f"read-back: {sum(len(s) for s in live)} recorded states equal the live ones")

    # 2. Replay: the recording alone reproduces the episodes.
    for k, (ep, rows) in enumerate(zip(rec["episodes"], replay(rec, [r["seed"] for r in results]))):
        if not np.array_equal(live_rows(rows), state_rows(ep["states"])):
            diff = np.abs(live_rows(rows) - state_rows(ep["states"])).max()
            raise AssertionError(f"episode {k}: replay differs from the recording (max {diff:.3g})")
    print(f"replay: {args.episodes} episodes re-simulated from the file match it bit for bit")


if __name__ == "__main__":
    main()
