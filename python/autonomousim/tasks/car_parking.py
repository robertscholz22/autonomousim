"""CarParking-v0: park a car in a free bay of a city, reversing into a parking lot's bay or
parallel parking between cars on the street, with ``(v, κ)`` actions; a scripted pilot (a
Reeds–Shepp path through the free space and a path tracker) proves the task solvable."""

import math
from typing import Any

import numpy as np

from autonomousim import reeds_shepp
from autonomousim.scenario import STATE
from autonomousim.tasks.base import Task

#: A horizontal ring of 72 beams, 20 m, 10 Hz, at the chassis frame's origin (about the
#: bumpers' height): the parked cars, the buildings and the lot's edges.
LIDAR = {
    "pattern": {"type": "rings", "elevations": [0.0], "azimuths": 72, "azimuth_fov": 360.0},
    "max_range": 20.0,
    "mount": {"position": [0.0, 0.0, 0.0]},
}

#: ``sedan_like``: tail to rear axle, rear axle to the front, half the width (m), wheelbase
#: (m) and steering lock (rad), for the scripted pilot.
SEDAN_REAR = 0.912
SEDAN_FRONT = 3.688
SEDAN_HALF_WIDTH = 0.92
SEDAN_WHEELBASE = 2.776
SEDAN_LOCK = 0.61
SEDAN_STEER_RATE = 1.2


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class CarParking(Task):
    """A car (``sedan_like``) in ``vk`` mode (speed up to ``max_speed`` either way, path
    curvature up to ``max_curvature``, the steering lock) parks in a marked bay of a generated
    ``urban`` map (a pool of ``map_count`` 512 m cities from ``map_seed``), with ``parked``
    cars parked in the bays nearest to it (``spawn.near_bay_goals``). ``bay`` goals of the
    ``kinds`` asked for:

    - in a parking lot (2.5 × 5 m bays either side of 6 m aisles) it reverses in: it starts in
      the middle of the aisle, ``distance`` m past the bay along it (either side), heading
      along the aisle (±10°); the goal has the tail 0.3 m from the bay's inner end, facing
      out;
    - on the street (6 m bays along the parking lanes) it parallel parks: it starts in the
      lane beside, its tail ``distance`` m ahead of the bay, heading with the traffic; the
      goal has the tail 0.3 m from the bay's rear end.

    Success: the tail within ``success_radius`` of the goal, the heading within
    ``success_heading`` degrees of it, stopped (below 0.3 m/s).

    Observation (80 values): ``trailer_goal`` (the goal relative to the tail in the car's
    heading frame, and sin, cos of the heading error; all scaled 1/10), speed, steering angle,
    last action and a horizontal LiDAR ring (``LIDAR``; ``lidar`` replaces its
    settings).

    Reward per step (``d``: distance of the tail from the goal; ``ψ``: heading error):

    - ``progress_weight·(d_before − d_after)``;
    - ``−heading_weight·|ψ|·min(1, 3/d)`` (alignment counts near the goal);
    - ``−smooth_weight·‖Δa‖²``;
    - ``success_bonus`` on success, ``−terminal_penalty`` on a crash (a parked car, a
      building), or the tail more than ``max_distance`` m from the goal.

    Episodes are truncated after ``episode_time`` (60 s). The policy runs at 20 Hz, the
    physics at 1 kHz. ``scripted(state)`` gives the actions of a parking pilot for
    comparison (``ParkingPilot``).
    """

    name = "car_parking"
    default_episode_time = 60.0
    has_success = True

    def __init__(
        self,
        *,
        kinds: tuple[str, ...] = ("lot", "street"),
        distance: tuple[float, float] = (4.0, 10.0),
        parked: int = 8,
        max_speed: float = 2.5,
        max_curvature: float = math.tan(SEDAN_LOCK) / SEDAN_WHEELBASE,
        success_radius: float = 0.5,
        success_heading: float = 5.0,
        max_distance: float = 25.0,
        lidar: dict[str, Any] | None = None,
        progress_weight: float = 1.0,
        heading_weight: float = 0.05,
        smooth_weight: float = 0.02,
        success_bonus: float = 20.0,
        terminal_penalty: float = 20.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("vehicle", "sedan_like")
        kwargs.setdefault("action_mode", "vk")
        kwargs.setdefault("map", "urban")
        kwargs.setdefault("policy_hz", 20)
        super().__init__(**kwargs)
        self.kinds = tuple(kinds)
        self.distance = distance
        self.parked = parked
        self.max_speed = max_speed
        self.max_curvature = max_curvature
        self.success_radius = success_radius
        self.success_heading = success_heading
        self.max_distance = max_distance
        self.lidar = LIDAR if lidar is None else lidar
        self.progress_weight = progress_weight
        self.heading_weight = heading_weight
        self.smooth_weight = smooth_weight
        self.success_bonus = success_bonus
        self.terminal_penalty = terminal_penalty
        self.pilot: ParkingPilot | None = None

    def group(self) -> dict[str, Any]:
        return {
            "goals": {"kind": "bay", "distance": list(self.distance), "bay": {"kinds": list(self.kinds)}},
            "ground_action_limits": {
                "speed": self.max_speed,
                "reverse": self.max_speed,
                "curvature": self.max_curvature,
            },
            "sensors": [{"name": "lidar", "type": "lidar", **self.lidar}],
            "obs": [
                {"term": "trailer_goal", "scale": 0.1},
                {"term": "speed", "scale": 0.3},
                {"term": "steering"},
                {"term": "last_action"},
                {"term": "lidar_log", "sensor": "lidar"},
            ],
        }

    def scenario(self) -> dict[str, Any]:
        sc = super().scenario()
        if self.parked > 0:
            # After the learning group, whose goal bays it keeps clear of.
            sc["groups"].append(
                {
                    "name": "parked",
                    "count": self.parked,
                    "vehicle": "sedan_like",
                    "physics": "kinematic",
                    "driver": {"type": "parked"},
                    "spawn": {"on_ground": True, "in_bays": True, "near_bay_goals": True},
                    "disable_on_terminal": False,
                }
            )
        return sc

    # ------------------------------------------------------------------ episodes

    def bind(self, num_envs: int, policy_dt: float, act_dim: int) -> None:
        super().bind(num_envs, policy_dt, act_dim)
        self.prev_distance = np.zeros(num_envs)

    def reset(self, mask: np.ndarray | None = None, state: np.ndarray | None = None) -> None:
        super().reset(mask, state)
        if state is not None:
            m = slice(None) if mask is None else mask
            self.prev_distance[m] = self.goal_errors(state[m])[0]

    @staticmethod
    def goal_errors(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance of the tail from the goal (m) and the heading error (rad)."""
        tail = state[:, STATE["tail"]]
        d = np.linalg.norm(tail[:, :2] - state[:, STATE["goal"]][:, :2], axis=1)
        return d, wrap(tail[:, 2] - state[:, STATE["goal_yaw"]][:, 0])

    def failed(self, state: np.ndarray) -> np.ndarray:
        return self.goal_errors(state)[0] > self.max_distance

    def succeeded(self, state: np.ndarray, events: np.ndarray) -> np.ndarray:
        d, psi = self.goal_errors(state)
        speed = np.linalg.norm(state[:, STATE["velocity"]], axis=1)
        return (d < self.success_radius) & (np.abs(psi) < np.radians(self.success_heading)) & (speed < 0.3)

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        d, psi = self.goal_errors(state)
        progress = self.prev_distance - d
        self.prev_distance[:] = d
        da = action - prev_action
        return (
            self.progress_weight * progress
            - self.heading_weight * np.abs(psi) * np.minimum(1.0, 3.0 / np.maximum(d, 1e-3))
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
            + self.success_bonus * self.succeeded(state, events)
        )

    # ------------------------------------------------------------------ scripted pilot

    def scripted(self, state: np.ndarray) -> np.ndarray:
        """Normalised actions of the parking pilot (``ParkingPilot``, one per world, planning
        anew at each episode's first step or for a new goal) for the state rows ``state``."""
        if self.pilot is None or self.pilot.num_envs != len(state):
            self.pilot = ParkingPilot(self, len(state))
        return self.pilot(state, self.steps)


class ParkingPilot:
    """A parking pilot for ``CarParking``: it plans a Reeds–Shepp path of the rear axle's
    centre from the start to the goal through the free space and tracks it.

    The free space is privileged knowledge of the bay's surroundings, in the goal's frame
    (x along the parked car, y to its left): the bay, its neighbours taken as occupied, and
    the aisle (6 m, in a lot) or the lane beside (on the street); the kind of bay shows in the
    start's heading (across the bay's in a lot, along it on the street). Paths of the radii in
    ``radii`` (times the turning radius at the lock) keep the whole car (plus ``margin``) in
    it; the cheapest wins (length plus ``cusp_cost`` per change of direction). Failing a
    direct path, the pilot goes through a pose in between, on a grid over the free space.

    The tracker drives each segment of one direction with pure pursuit (``lookahead`` m,
    past a segment's end along its last heading), at up to ``speed`` m/s and slowing at
    ``decel`` m/s² for the segment's end; at a change of direction it stops and waits for the
    steering to reach the next segment's curvature. Straying more than ``replan`` m off the
    path, or stopping short of the goal, it plans again from where it stands.
    """

    def __init__(
        self,
        task: CarParking,
        num_envs: int,
        radii: tuple[float, ...] = (1.1, 1.4),
        margin: float = 0.1,
        cusp_cost: float = 3.0,
        lookahead: float = 1.0,
        speed: float = 1.2,
        decel: float = 0.6,
        replan: float = 0.5,
    ):
        self.task = task
        self.num_envs = num_envs
        self.turn_radius = SEDAN_WHEELBASE / math.tan(SEDAN_LOCK)
        self.radii = radii
        self.margin = margin
        self.cusp_cost = cusp_cost
        self.lookahead = lookahead
        self.speed = speed
        self.decel = decel
        self.replan_distance = replan
        self.plans: list[list[np.ndarray] | None] = [None] * num_envs
        self.segment = np.zeros(num_envs, dtype=np.int64)
        self.wait = np.zeros(num_envs)
        self.steer = np.zeros(num_envs)  # estimated steering angle (rad)
        self.replans = np.zeros(num_envs, dtype=np.int64)
        self.goals = np.full((num_envs, 2), np.nan)
        self.free: list[list[tuple[float, float, float, float]]] = [[] for _ in range(num_envs)]
        x = np.linspace(-SEDAN_REAR - margin, SEDAN_FRONT + margin, 16)
        y = np.linspace(-SEDAN_HALF_WIDTH - margin, SEDAN_HALF_WIDTH + margin, 6)
        self.outline = np.concatenate(
            [
                np.stack([x, np.full_like(x, y[0])], 1),
                np.stack([x, np.full_like(x, y[-1])], 1),
                np.stack([np.full_like(y, x[0]), y], 1),
                np.stack([np.full_like(y, x[-1]), y], 1),
            ]
        )

    # -------------------------------------------------------------- geometry

    @staticmethod
    def rear_axle_in_goal(row: np.ndarray) -> tuple[float, float, float]:
        """The rear axle's pose in the goal's frame (whose origin is the goal's rear axle)."""
        tail = row[STATE["tail"]]
        gx, gy = row[STATE["goal"]][:2]
        gyaw = float(row[STATE["goal_yaw"]][0])
        c, s = math.cos(gyaw), math.sin(gyaw)
        th = float(tail[2])
        ax = tail[0] + SEDAN_REAR * math.cos(th) - gx
        ay = tail[1] + SEDAN_REAR * math.sin(th) - gy
        return c * ax + s * ay - SEDAN_REAR, -s * ax + c * ay, float(wrap(np.array(th - gyaw)))

    @staticmethod
    def free_space(start: tuple[float, float, float]) -> list[tuple[float, float, float, float]]:
        """Rectangles (x0, x1, y0, y1) of free space in the goal's frame (origin at the goal's
        rear axle) for a start pose there."""
        r = SEDAN_REAR
        sx, sy, sth = start
        if abs(math.cos(sth)) < 0.5:
            # Lot: the bay (the tail 0.3 m from its inner end; the neighbours' cars, centred
            # in theirs, end 0.2 m short of the aisle) and the 6 m aisle, around the start.
            lo, hi = min(0.0, sy) - 6.0, max(0.0, sy) + 6.0
            return [(-0.25 - r, 4.7 - r, -1.5, 1.5), (4.55 - r, 10.6 - r, lo, hi)]
        # Street: the bay (2.2 m wide, the sidewalk beyond; the body may overhang it by
        # 0.2 m, short of the lamp posts) with 0.7 m at each end up to the neighbours' cars
        # (centred in their 6 m bays), and beside them, the lane and half the next (the front
        # swings out), around the start.
        lo, hi = min(0.0, sx) - 10.0, max(0.0, sx) + 10.0
        return [(-0.9 - r, 6.3 - r, -1.3, 1.1), (lo, hi, 0.95, 6.0)]

    def clear(self, poses: np.ndarray, rects: list[tuple[float, float, float, float]]) -> bool:
        """Whether the car at every pose (rows x, y, heading, ...) lies in the free space."""
        c, s = np.cos(poses[:, 2:3]), np.sin(poses[:, 2:3])
        px = poses[:, 0:1] + c * self.outline[:, 0] - s * self.outline[:, 1]
        py = poses[:, 1:2] + s * self.outline[:, 0] + c * self.outline[:, 1]
        inside = np.zeros(px.shape, dtype=bool)
        for x0, x1, y0, y1 in rects:
            inside |= (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
        return bool(inside.all())

    def best_path(self, start, goal, rects, budget: float = math.inf):
        """The cheapest clear path from ``start`` to ``goal``: (cost, samples), or None."""
        best = None
        for k in self.radii:
            radius = k * self.turn_radius
            for p in reeds_shepp.paths(start, goal, radius):
                cost = reeds_shepp.length(p) * radius + self.cusp_cost * reeds_shepp.cusps(p)
                if cost >= min(budget, best[0] if best else math.inf):
                    continue
                poses = reeds_shepp.sample(start, p, radius, 0.1)
                if self.clear(poses[::2], rects) and self.clear(poses[-1:], rects):
                    best = (cost, poses)
        return best

    def plan(self, start: tuple[float, float, float], rects) -> list[np.ndarray] | None:
        """Segments (sampled poses of one direction each) from ``start`` to the goal."""
        goal = (0.0, 0.0, 0.0)
        best = self.best_path(start, goal, rects)
        if best is None:
            # Through a pose in between: a grid over the free space.
            for x0, x1, y0, y1 in rects:
                for x in np.arange(x0 + 1.0, x1, 1.0):
                    for y in np.arange(y0 + 0.5, y1, 1.0):
                        for th in np.linspace(-math.pi, math.pi, 8, endpoint=False):
                            mid = (float(x), float(y), float(th))
                            if not self.clear(np.array([mid]), rects):
                                continue
                            budget = best[0] if best else math.inf
                            a = self.best_path(start, mid, rects, budget)
                            if a is None:
                                continue
                            b = self.best_path(mid, goal, rects, budget - a[0])
                            if b is not None:
                                best = (a[0] + b[0], np.concatenate([a[1], b[1][1:]]))
        if best is None:
            return None
        poses = best[1]
        # Split where the direction changes.
        cuts = np.flatnonzero(np.diff(poses[1:, 3]) != 0) + 2
        return [s for s in np.split(poses, cuts) if len(s) > 1] or [poses]

    # -------------------------------------------------------------- tracking

    def __call__(self, state: np.ndarray, steps: np.ndarray) -> np.ndarray:
        dt = self.task.policy_dt
        out = np.zeros((self.num_envs, 2), dtype=np.float32)
        for i, row in enumerate(state):
            pose = self.rear_axle_in_goal(row)
            goal = row[STATE["goal"]][:2]
            if steps[i] == 0 or not np.array_equal(goal, self.goals[i]):
                self.goals[i] = goal
                self.free[i] = self.free_space(pose)
                self.plans[i] = self.plan(pose, self.free[i])
                self.segment[i] = 0
                self.wait[i] = 1.0
                self.replans[i] = 0
                self.steer[i] = 0.0
            v, kappa = self.track(i, pose, float(np.linalg.norm(row[STATE["velocity"]])))
            # Steering estimate: toward the commanded angle at the steering rate.
            target = math.atan(kappa * SEDAN_WHEELBASE)
            self.steer[i] += float(np.clip(target - self.steer[i], -SEDAN_STEER_RATE * dt, SEDAN_STEER_RATE * dt))
            out[i] = (v / self.task.max_speed, np.clip(kappa / self.task.max_curvature, -1.0, 1.0))
        return out

    def track(self, i: int, pose: tuple[float, float, float], speed: float) -> tuple[float, float]:
        plan = self.plans[i]
        if plan is None:
            return 0.0, 0.0
        x, y, th = pose
        if self.segment[i] >= len(plan):
            # At the end: plan again if short of the goal.
            if speed < 0.1 and (math.hypot(x, y) > 0.3 or abs(th) > math.radians(3.0)) and self.replans[i] < 3:
                self.replans[i] += 1
                self.plans[i] = self.plan(pose, self.free[i])
                self.segment[i] = 0
                self.wait[i] = 1.0
            return 0.0, 0.0
        seg = plan[self.segment[i]]
        d = seg[1, 3]
        kappa_start = seg[1, 4]
        if self.wait[i] > 0.0:
            # At a segment's start: stopped, steering to its first curvature.
            aligned = abs(self.steer[i] - math.atan(kappa_start * SEDAN_WHEELBASE)) < 0.05
            if aligned and speed < 0.1:
                self.wait[i] = 0.0
            return 0.0, float(kappa_start)
        # Nearest point and the arc length still to go.
        dist = np.hypot(seg[:, 0] - x, seg[:, 1] - y)
        k = int(np.argmin(dist))
        steps = np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1]))
        arc = np.concatenate([[0.0], np.cumsum(steps)])
        # Past the end along the last heading (in the direction of travel).
        ex, ey, eth = seg[-1, :3]
        beyond = d * ((x - ex) * math.cos(eth) + (y - ey) * math.sin(eth))
        remaining = max(0.0, arc[-1] - arc[k]) if beyond < 0.0 else 0.0
        if dist[k] > self.replan_distance and self.replans[i] < 3:
            self.replans[i] += 1
            self.plans[i] = self.plan(pose, self.free[i])
            self.segment[i] = 0
            self.wait[i] = 1.0
            return 0.0, 0.0
        if remaining < 0.03:
            if speed < 0.1:
                self.segment[i] += 1
                self.wait[i] = 1.0
            return 0.0, float(seg[-1, 4])
        # Pure pursuit on the look-ahead point.
        ahead = arc[k] + self.lookahead
        if ahead <= arc[-1]:
            j = int(np.searchsorted(arc, ahead))
            px, py = seg[j, 0], seg[j, 1]
        else:
            extra = ahead - arc[-1]
            px, py = ex + d * extra * math.cos(eth), ey + d * extra * math.sin(eth)
        lx = math.cos(th) * (px - x) + math.sin(th) * (py - y)
        ly = -math.sin(th) * (px - x) + math.cos(th) * (py - y)
        kappa = 2.0 * ly / max(lx * lx + ly * ly, 1e-6)
        v = d * max(0.25, min(self.speed, math.sqrt(2.0 * self.decel * remaining)))
        return v, kappa
