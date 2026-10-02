"""Listens to a bridge's LiDAR, camera, marker and route topics from ROS 2 (rclpy) and prints a
JSON summary.

    tools/ros/run.sh python3 tools/ros/street_probe.py --seconds 6 --agent 0 --lidar lidar --camera front

For each topic: message count, rate from the header stamps, largest stamp gap, and the shape of
the last message (cloud width and fields, image size and encoding, camera matrix, marker count
and the first marker's scale, route length). Also the `/clock` range against the wall clock.
"""

import argparse
import json
import time

import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from visualization_msgs.msg import MarkerArray

RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def shape(m):
    if isinstance(m, PointCloud2):
        return {"width": m.width, "height": m.height, "point_step": m.point_step, "fields": [f.name for f in m.fields],
                "bytes": len(m.data), "frame": m.header.frame_id}
    if isinstance(m, Image):
        return {"width": m.width, "height": m.height, "encoding": m.encoding, "step": m.step, "bytes": len(m.data),
                "frame": m.header.frame_id}
    if isinstance(m, CameraInfo):
        return {"width": m.width, "height": m.height, "k": list(m.k), "frame": m.header.frame_id}
    if isinstance(m, MarkerArray):
        first = m.markers[0] if m.markers else None
        return {"markers": len(m.markers), "frame": first.header.frame_id if first else None,
                "scale": [first.scale.x, first.scale.y, first.scale.z] if first else None}
    if isinstance(m, Path):
        return {"poses": len(m.poses), "frame": m.header.frame_id}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0, help="listening time (wall clock)")
    ap.add_argument("--agent", type=int, default=0)
    ap.add_argument("--lidar", default="lidar")
    ap.add_argument("--camera", default="front")
    args = ap.parse_args()

    rclpy.init()
    node = Node("autonomousim_street_probe")
    ns = f"/agent{args.agent}"
    topics = {}

    def listen(typ, topic, qos):
        t = topics[topic] = {"stamps": [], "last": None, "first_wall": None}

        def cb(m):
            t["stamps"].append(stamp(m) if not isinstance(m, MarkerArray) else (stamp(m.markers[0]) if m.markers else 0.0))
            t["last"] = m
            t["first_wall"] = t["first_wall"] or time.monotonic()

        node.create_subscription(typ, topic, cb, qos)

    listen(PointCloud2, f"{ns}/{args.lidar}", qos_profile_sensor_data)
    cam = f"{ns}/{args.camera}"
    for kind in ("image", "depth", "semantic"):
        listen(Image, f"{cam}/{kind}", qos_profile_sensor_data)
    listen(CameraInfo, f"{cam}/camera_info", qos_profile_sensor_data)
    for name in ("npcs", "pedestrians", "signals"):
        listen(MarkerArray, f"/autonomousim/{name}", RELIABLE)
    listen(Path, f"{ns}/route", LATCHED)
    clocks = []
    node.create_subscription(Clock, "/clock", lambda m: clocks.append((time.monotonic(), m.clock.sec + m.clock.nanosec * 1e-9)), 10)

    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.02)

    out = {"topics": {}}
    for topic, t in topics.items():
        s = sorted(t["stamps"])
        summary = {"count": len(s)}
        if len(s) >= 2 and s[-1] > s[0]:
            summary["rate"] = (len(s) - 1) / (s[-1] - s[0])
            summary["max_gap"] = max(b - a for a, b in zip(s, s[1:]))
        if t["last"] is not None:
            summary["last"] = shape(t["last"])
        out["topics"][topic] = summary
    if len(clocks) >= 2:
        (w0, c0), (w1, c1) = clocks[0], clocks[-1]
        out["clock"] = {"count": len(clocks), "sim": c1 - c0, "wall": w1 - w0}
    print(json.dumps(out))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
