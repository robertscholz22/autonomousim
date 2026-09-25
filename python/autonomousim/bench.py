"""Throughput of the Python API: ``python -m autonomousim.bench``.

Steps the Gymnasium vector environment with random actions and reports environment steps
per second (one policy step of one world), split into the native step, native resets and
Python (task reward and termination, bookkeeping, copies). Random actions end episodes
quickly, so resets are frequent. Reports the median of ``--repeat`` runs and can append the
results to ``benchmarks/results/<date>-<host>.json``.
"""

import argparse
import datetime
import json
import pathlib
import platform
import time

import numpy as np

from autonomousim.vector_env import AutonomousimVectorEnv


class _Timed:
    """Forwards ``step`` and ``reset`` to the native simulation and times them."""

    def __init__(self, sim):
        self.sim = sim
        self.step_s = 0.0
        self.reset_s = 0.0

    def step(self, actions):
        t = time.perf_counter()
        self.sim.step(actions)
        self.step_s += time.perf_counter() - t

    def reset(self, *args):
        t = time.perf_counter()
        self.sim.reset(*args)
        self.reset_s += time.perf_counter() - t

    def __getattr__(self, name):
        return getattr(self.sim, name)


def bench(task: str, num_envs: int, threads: int, steps: int, repeat: int, **task_kwargs) -> dict:
    """Step the vector environment with random actions (episodes end often and are reset
    in the same step) and split the time into the native step, native resets and Python
    (task, bookkeeping, copies)."""
    envs = AutonomousimVectorEnv(num_envs, task, num_threads=threads, **task_kwargs)
    rng = np.random.default_rng(0)
    pool = rng.uniform(-1, 1, (64, num_envs, envs.act_dim)).astype(np.float32)
    envs.reset(seed=0)
    for i in range(20):  # warm up
        envs.step(pool[i % 64])
    runs = []
    for _ in range(repeat):
        timed = envs.sim = _Timed(envs.sim.sim if isinstance(envs.sim, _Timed) else envs.sim)
        episodes = 0
        t = time.perf_counter()
        for i in range(steps):
            _, _, terminated, truncated, _ = envs.step(pool[i % 64])
            episodes += int(terminated.sum() + truncated.sum())
        total = time.perf_counter() - t
        runs.append((total, timed.step_s, timed.reset_s, episodes))
    envs.close()
    total, step_s, reset_s, episodes = sorted(runs)[len(runs) // 2]
    return {
        "task": task,
        "num_envs": num_envs,
        "threads": envs.sim.num_threads,
        "vector_env_steps_per_s": num_envs * steps / total,
        "native_steps_per_s": num_envs * steps / step_s,
        "native_step_us": 1e6 * step_s / steps,
        "native_reset_us": 1e6 * reset_s / steps,
        "python_us": 1e6 * (total - step_s - reset_s) / steps,
        "mean_episode_steps": num_envs * steps / max(episodes, 1),
        **{k: v for k, v in task_kwargs.items() if isinstance(v, (int, float, str))},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="hover", choices=["hover", "recover", "waypoint_forest", "car_waypoint"])
    p.add_argument("--num-envs", type=int, nargs="+", default=[256])
    p.add_argument("--threads", type=int, nargs="+", default=[10])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--map", default=None, help="default: flat for hover and recover, the task's own otherwise")
    p.add_argument("--vehicle", default=None, help="default: the task's")
    p.add_argument("--action-mode", default=None, help="default: the task's")
    p.add_argument("--save", action="store_true", help="append to benchmarks/results/<date>-<host>.json")
    args = p.parse_args()

    rows = []
    print(
        f"{'task':12} {'envs':>5} {'thr':>4} {'vector env/s':>13} {'native/s':>10} "
        f"{'step µs':>8} {'reset µs':>9} {'python µs':>10} {'ep len':>7}"
    )
    for n in args.num_envs:
        for t in args.threads:
            options = {"map": args.map, "vehicle": args.vehicle, "action_mode": args.action_mode}
            if args.task in ("hover", "recover"):
                options["map"] = options["map"] or "flat"
            r = bench(args.task, n, t, args.steps, args.repeat, **{k: v for k, v in options.items() if v is not None})
            rows.append(r)
            print(
                f"{r['task']:12} {n:5d} {r['threads']:4d} {r['vector_env_steps_per_s']:13,.0f} "
                f"{r['native_steps_per_s']:10,.0f} {r['native_step_us']:8.0f} {r['native_reset_us']:9.0f} "
                f"{r['python_us']:10.0f} {r['mean_episode_steps']:7.1f}"
            )
    if args.save:
        root = pathlib.Path(__file__).resolve().parents[2]
        path = root / "benchmarks" / "results" / f"{datetime.date.today().isoformat()}-{platform.node()}.json"
        data = json.loads(path.read_text()) if path.exists() else {"date": datetime.date.today().isoformat()}
        data.setdefault("python", []).extend(rows)
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"appended to {path.relative_to(root)}")


if __name__ == "__main__":
    main()
