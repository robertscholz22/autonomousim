"""A waypoint follower for the autonomousim ROS 2 bridge (rclpy), the example's controller.

    tools/ros/run.sh python3 examples/ros/waypoints.py [--agent 0] [--speed 3]

Subscribes to `/agent<id>/odom` (pose in `map`, ENU) and publishes `/agent<id>/cmd_vel`: a
velocity setpoint in the drone's heading frame (x forward, y left, z up, m/s) for every
odometry message, towards the next waypoint. The waypoints are offsets from where the drone
starts (east, north, up): climb to 8 m, a 24 m square over the pillars, back and down to 1 m. Runs on simulated time (`use_sim_time`), prints a JSON report and exits with 0 once
every waypoint was reached (1 on `--timeout`).
"""

import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy

WAYPOINTS = [(0, 0, 8), (24, 0, 8), (24, 24, 8), (0, 24, 8), (0, 0, 8), (0, 0, 1)]


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Follower(Node):
    def __init__(self, agent: int, speed: float, tolerance: float):
        super().__init__("waypoint_follower", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.speed, self.tolerance = speed, tolerance
        self.start = None  # (x, y, z)
        self.next = 0
        self.reached = []  # (waypoint, simulated time)
        self.last = None
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.cmd = self.create_publisher(Twist, f"/agent{agent}/cmd_vel", qos)
        self.create_subscription(Odometry, f"/agent{agent}/odom", self.on_odom, qos)

    def targets(self):
        x0, y0, z0 = self.start
        return [(x0 + x, y0 + y, z0 + z) for x, y, z in WAYPOINTS]

    @property
    def done(self):
        return self.next >= len(WAYPOINTS)

    def on_odom(self, m: Odometry):
        p, yaw = m.pose.pose.position, yaw_of(m.pose.pose.orientation)
        self.last = (p.x, p.y, p.z)
        if self.start is None:
            self.start = (p.x, p.y, p.z)
            self.get_logger().info(f"start at ({p.x:.1f}, {p.y:.1f}, {p.z:.1f})")
        t = Twist()
        while not self.done:
            tx, ty, tz = self.targets()[self.next]
            ex, ey, ez = tx - p.x, ty - p.y, tz - p.z
            if math.sqrt(ex * ex + ey * ey + ez * ez) > self.tolerance:
                # The error in the heading frame, at most `speed`.
                c, s = math.cos(yaw), math.sin(yaw)
                v = (c * ex + s * ey, -s * ex + c * ey, ez)
                g = min(1.0, self.speed / math.sqrt(sum(x * x for x in v)))
                t.linear.x, t.linear.y, t.linear.z = (g * x for x in v)
                break
            stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            self.reached.append((self.next, round(stamp, 2)))
            self.get_logger().info(f"waypoint {self.next} reached at t = {stamp:.2f} s")
            self.next += 1
        self.cmd.publish(t)  # zero once done: hover


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", type=int, default=0)
    ap.add_argument("--speed", type=float, default=3.0, help="m/s")
    ap.add_argument("--tolerance", type=float, default=0.7, help="m")
    ap.add_argument("--timeout", type=float, default=120.0, help="wall-clock limit (s)")
    args = ap.parse_args()
    rclpy.init()
    node = Follower(args.agent, args.speed, args.tolerance)
    end = time.monotonic() + args.timeout
    while not node.done and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    for _ in range(10):  # let the last command go out
        rclpy.spin_once(node, timeout_sec=0.01)
    print(json.dumps({"reached": node.reached, "waypoints": len(WAYPOINTS), "start": node.start, "position": node.last}))
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if node.done else 1)


if __name__ == "__main__":
    main()
