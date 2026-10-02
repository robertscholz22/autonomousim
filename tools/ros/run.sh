#!/usr/bin/env bash
# Runs a command in ROS 2 Lyrical (Docker, `ros:lyrical-ros-base`) with the repository mounted at
# /ws, host networking (DDS discovery with processes on the host) and the caller's user id.
#   tools/ros/run.sh ros2 topic echo /agent0/odom
set -euo pipefail
IMAGE="${AUTONOMOUSIM_ROS_IMAGE:-ros:lyrical-ros-base}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TTY=()
[ -t 0 ] && [ -t 1 ] && TTY=(-it)
exec docker run --rm "${TTY[@]}" --network host --ipc host \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e ROS_LOG_DIR=/tmp/ros_log \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}" \
    -v "$ROOT:/ws" -w /ws "$IMAGE" \
    bash -c 'source /opt/ros/lyrical/setup.bash && exec "$@"' bash "$@"
