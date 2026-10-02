"""An external controller for the bridge (rclpy): answers every `/clock` tick with a `cmd_vel`.

    tools/ros/run.sh python3 tools/ros/drive.py square --car 1 --drone 0
    tools/ros/run.sh python3 tools/ros/drive.py pattern --car 1 --drone 0 --steps 300

Meant for a bridge in lockstep (`autonomousim-ros run --lockstep`): it publishes one command per
agent for each new clock value, so the bridge steps once it has them all.

- `square`: closed loop on odometry. The car drives the corners of a square (side `--side`)
  ahead and to the left of where it starts; the drone flies through waypoints around its start.
  Prints a JSON report: corners and waypoints reached, final positions.
- `pattern`: open loop, commands a function of the policy step `k` only (exactly representable
  values, so a replay elsewhere gives the same commands); exits after `--steps` steps.
"""

import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock

RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


def pattern(k, kind):
    """The open-loop commands at step k: (linear x, y, z, angular z)."""
    if kind == "car":
        return (0.5 * ((k // 25) % 4), 0.0, 0.0, 0.25 * ((k // 40) % 3 - 1))
    return (0.25 * ((k // 30) % 3), 0.125 * ((k // 20) % 3 - 1), 0.25 * ((k // 50) % 2), 0.125 * ((k // 35) % 2))


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Driver(Node):
    def __init__(self, args):
        super().__init__("autonomousim_driver")
        self.args = args
        self.agents = {}
        for kind in ("car", "drone"):
            agent = getattr(args, kind)
            if agent is None:
                continue
            a = {"kind": kind, "id": agent, "odom": None, "start": None, "target": 0, "reached_at": []}
            a["pub"] = self.create_publisher(Twist, f"/agent{agent}/cmd_vel", RELIABLE)
            self.create_subscription(Odometry, f"/agent{agent}/odom", lambda m, a=a: self.on_odom(a, m), RELIABLE)
            self.agents[kind] = a
        self.create_subscription(Clock, "/clock", self.on_clock, RELIABLE)
        self.last_k = -1
        self.done = False

    def on_odom(self, a, m):
        a["odom"] = m
        if a["start"] is None:
            p = m.pose.pose.position
            a["start"] = (p.x, p.y, p.z, yaw_of(m.pose.pose.orientation))

    def on_clock(self, m):
        k = round((m.clock.sec + m.clock.nanosec * 1e-9) * 50.0)
        if k <= self.last_k:
            return  # a repeat of a tick already answered
        if self.args.mode == "square" and any(a["odom"] is None for a in self.agents.values()):
            return  # no state to act on yet: the bridge sends the tick again
        self.last_k = k
        for a in self.agents.values():
            if self.args.mode == "pattern":
                cmd = pattern(k, a["kind"])
            else:
                cmd = self.steer(a)
            t = Twist()
            t.linear.x, t.linear.y, t.linear.z, t.angular.z = (float(c) for c in cmd)
            a["pub"].publish(t)
        if self.args.mode == "pattern" and k >= self.args.steps:
            self.done = True
        if self.args.mode == "square" and all(
            a["start"] is not None and a["target"] >= len(self.targets(a)) for a in self.agents.values()
        ):
            self.done = True

    def targets(self, a):
        x0, y0, z0, h = a["start"]
        s = self.args.side
        if a["kind"] == "car":
            local = [(s, 0.0, 0.0), (s, s, 0.0), (0.0, s, 0.0), (0.0, 0.0, 0.0)]
        else:
            local = [(4.0, 0.0, 1.0), (4.0, 4.0, 2.0), (0.0, 4.0, 1.0), (0.0, 0.0, 0.0)]
        c, sn = math.cos(h), math.sin(h)
        return [(x0 + c * x - sn * y, y0 + sn * x + c * y, z0 + z) for x, y, z in local]

    def steer(self, a):
        if a["odom"] is None:
            return (0.0, 0.0, 0.0, 0.0)
        targets = self.targets(a)
        if a["target"] >= len(targets):
            return (0.0, 0.0, 0.0, 0.0)
        p = a["odom"].pose.pose.position
        tx, ty, tz = targets[a["target"]]
        ex, ey, ez = tx - p.x, ty - p.y, tz - p.z
        yaw = yaw_of(a["odom"].pose.pose.orientation)
        if a["kind"] == "car":
            if math.hypot(ex, ey) < 2.0:
                a["target"] += 1
                a["reached_at"].append(self.last_k / 50.0)
                return self.steer(a)
            err = wrap(math.atan2(ey, ex) - yaw)
            v = 3.0 * max(0.3, math.cos(err))
            return (v, 0.0, 0.0, max(-1.0, min(1.0, 2.0 * err)) * v / 3.0)
        if math.sqrt(ex * ex + ey * ey + ez * ez) < 0.5:
            a["target"] += 1
            a["reached_at"].append(self.last_k / 50.0)
            return self.steer(a)
        # Velocity in the heading frame (x forward, y left, z up), at most 2 m/s.
        c, s = math.cos(yaw), math.sin(yaw)
        vx, vy, vz = c * ex + s * ey, -s * ex + c * ey, ez
        n = math.sqrt(vx * vx + vy * vy + vz * vz)
        g = min(1.0, 2.0 / n) if n > 0 else 0.0
        return (g * vx, g * vy, g * vz, 0.0)

    def report(self):
        out = {}
        for kind, a in self.agents.items():
            p = a["odom"].pose.pose.position if a["odom"] else None
            out[kind] = {
                "reached": a["target"],
                "reached_at": a["reached_at"],
                "targets": len(self.targets(a)) if a["start"] else None,
                "start": a["start"],
                "position": [p.x, p.y, p.z] if p else None,
            }
        out["last_step"] = self.last_k
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["square", "pattern"])
    ap.add_argument("--car", type=int)
    ap.add_argument("--drone", type=int)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--side", type=float, default=12.0)
    ap.add_argument("--timeout", type=float, default=120.0, help="wall-clock limit (s)")
    args = ap.parse_args()
    rclpy.init()
    node = Driver(args)
    end = time.monotonic() + args.timeout
    while not node.done and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    # Let the last commands go out before the node goes away.
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.01)
    print(json.dumps(node.report()))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
