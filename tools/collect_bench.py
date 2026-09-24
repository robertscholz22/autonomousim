"""Collect criterion medians from target/criterion into benchmarks/results/<date>-<host>.json."""

import datetime
import json
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def cpu_model() -> str:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main() -> None:
    crit = ROOT / "target" / "criterion"
    results = {}
    for est in sorted(crit.rglob("new/estimates.json")):
        bench_json = est.parent / "benchmark.json"
        name = json.loads(bench_json.read_text())["full_id"] if bench_json.exists() else est.parent.parent.name
        results[name] = {"median_ns": json.loads(est.read_text())["median"]["point_estimate"]}
    if not results:
        sys.exit("no criterion results found; run `cargo bench` first")
    governor = pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    out = {
        "date": datetime.date.today().isoformat(),
        "host": platform.node(),
        "cpu": cpu_model(),
        "governor": governor.read_text().strip() if governor.exists() else None,
        "rustc": rustc,
        "results": results,
    }
    path = ROOT / "benchmarks" / "results" / f"{out['date']}-{out['host']}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({len(results)} benchmarks)")


if __name__ == "__main__":
    main()
