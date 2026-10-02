#!/usr/bin/env bash
# The ROS 2 example end to end (examples/ros/README.md): the bridge runs the scenario on the
# host in real time, an rclpy node in the ROS 2 container flies the drone through its
# waypoints, and `ros2 bag record` (also in the container) records the run as a rosbag2 bag.
#   examples/ros/demo.sh [output dir, default runs/ros_example]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="${1:-runs/ros_example}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

cargo build --release -p autonomousim-ros
rm -rf "$OUT/bag"
mkdir -p "$OUT"
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT

# The recorder first, so that it has discovered the bridge's topics when they start.
tools/ros/run.sh timeout -s INT 70 ros2 bag record -s mcap --use-sim-time --all-topics -o "$OUT/bag" \
    > "$OUT/record.log" 2>&1 &
pids+=($!)
sleep 3
target/release/autonomousim-ros run --scenario examples/ros/drone.toml --duration 60 > "$OUT/bridge.log" 2>&1 &
pids+=($!)
tools/ros/run.sh python3 examples/ros/waypoints.py | tee "$OUT/waypoints.json"
wait "${pids[@]}"
trap - EXIT
tools/ros/run.sh ros2 bag info "$OUT/bag"
