"""Listens to a running bridge from ROS 2 (rclpy, `use_sim_time`) and prints a JSON summary.

    tools/ros/run.sh python3 tools/ros/probe.py --seconds 4 --agent 0 --sensors imu:Imu,gps:NavSatFix

For each topic: message count, the rate from the header stamps, the largest stamp gap; the
`/clock` range; the latched `/tf_static` and `/autonomousim/meta`; and `map` → `agent<id>/base_link`
looked up through tf2 at the stamp of the last odometry message, next to that message's pose.
"""

import argparse
import json
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs import msg as sensor_msgs
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def summary(stamps):
    if len(stamps) < 2:
        return {"count": len(stamps)}
    stamps = sorted(stamps)
    span = stamps[-1] - stamps[0]
    gap = max(b - a for a, b in zip(stamps, stamps[1:]))
    return {"count": len(stamps), "first": stamps[0], "last": stamps[-1], "rate": (len(stamps) - 1) / span, "max_gap": gap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0, help="listening time (wall clock)")
    ap.add_argument("--agent", type=int, default=0)
    ap.add_argument("--sensors", default="", help="name:Type,... (sensor_msgs types)")
    args = ap.parse_args()

    rclpy.init()
    node = Node("autonomousim_probe", parameter_overrides=[Parameter("use_sim_time", value=True)])
    ns = f"/agent{args.agent}"
    stamps = {}

    def record(topic):
        stamps[topic] = []
        return lambda m: stamps[topic].append(stamp(m))

    last_odom = []
    clocks = []
    latched = {}
    node.create_subscription(Clock, "/clock", lambda m: clocks.append(m.clock.sec + m.clock.nanosec * 1e-9), 10)
    odom_cb = record(f"{ns}/odom")

    def on_odom(m):
        odom_cb(m)
        last_odom[:] = [m]

    node.create_subscription(Odometry, f"{ns}/odom", on_odom, 10)
    for item in filter(None, args.sensors.split(",")):
        name, typ = item.split(":")
        node.create_subscription(getattr(sensor_msgs, typ), f"{ns}/{name}", record(f"{ns}/{name}"), qos_profile_sensor_data)
    node.create_subscription(TFMessage, "/tf_static", lambda m: latched.setdefault("tf_static", m), LATCHED)
    node.create_subscription(String, "/autonomousim/meta", lambda m: latched.setdefault("meta", m), LATCHED)
    buffer = Buffer(cache_time=Duration(seconds=30))
    TransformListener(buffer, node)

    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)

    out = {"topics": {t: summary(s) for t, s in stamps.items()}, "clock": summary(clocks)}
    out["tf_static"] = [
        {
            "parent": t.header.frame_id,
            "child": t.child_frame_id,
            "translation": [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z],
            "rotation": [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w],
        }
        for t in getattr(latched.get("tf_static"), "transforms", [])
    ]
    out["meta"] = json.loads(latched["meta"].data) if "meta" in latched else None
    if last_odom:
        o = last_odom[0]
        p = o.pose.pose.position
        out["odom_last"] = {"stamp": stamp(o), "position": [p.x, p.y, p.z]}
        try:
            tf = buffer.lookup_transform("map", f"agent{args.agent}/base_link", Time.from_msg(o.header.stamp))
            t = tf.transform.translation
            out["tf_lookup"] = [t.x, t.y, t.z]
        except Exception as e:  # noqa: BLE001 (reported, the test fails on it)
            out["tf_lookup"] = str(e)
    out["sim_time_now"] = node.get_clock().now().nanoseconds * 1e-9
    print(json.dumps(out))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
