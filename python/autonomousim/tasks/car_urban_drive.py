"""CarUrbanDrive-v0: drive a car along a lane-level route of 0.5–1.5 km through a generated
city in full traffic (NPC cars, buses, cyclists, pedestrians, traffic lights, roundabouts);
the traffic driver flying the car is the scripted baseline."""

import json
from typing import Any

import numpy as np

from autonomousim._native import TERMINAL_EVENTS, BatchSim
from autonomousim.events import Event
from autonomousim.scenario import STATE
from autonomousim.tasks.base import Task

FINISHED = int(Event.FINISHED)
STUCK = int(Event.STUCK)
RED_LIGHT = int(Event.RED_LIGHT)
WRONG_WAY = int(Event.WRONG_WAY)
OFF_ROAD = int(Event.OFF_ROAD)

#: Two rings of 90 beams (−6° and level), 40 m, 10 Hz, above the car's centre: the sensor-only
#: configuration's view of the traffic, the pedestrians, the kerbs' surroundings and the
#: buildings.
LIDAR = {
    "pattern": {"type": "rings", "elevations": [-6.0, 0.0], "azimuths": 90, "azimuth_fov": 360.0},
    "max_range": 40.0,
    "mount": {"position": [0.5, 0.0, 0.9]},
}

#: Privileged observation terms: the route (goals, lane line), the lane and the movement
#: ahead, the light, the car's own motion, and the road users nearby.
PRIVILEGED = "privileged"
#: Sensor-only: the LiDAR plus a route hint (the next goal, the route's line and the light).
SENSORS = "sensors"


class CarUrbanDrive(Task):
    """A car (``sedan_like``) in ``vk`` mode (speed up to ``max_speed``) drives a lane-level
    route through a generated city (``urban``: a pool of ``map_count`` 512 m maps from
    ``map_seed``; evaluate on another ``map_seed`` for unseen maps). The route (``route``
    goals to ``lanes``) starts on a random lane and follows the lane graph's movements, without
    lane changes, for a length drawn from ``distance``; goals lie every ``goal_step`` m along
    it, the last at its end (reaching it: success).

    The city is busy: ``npcs`` traffic cars (simulated in full within 40 m of the car, else
    kinematically), ``buses`` city buses on their loops, ``cyclists`` (hybrid like the cars),
    ``parked`` cars in bays and ``pedestrians`` (jaywalking with probability ``jaywalk`` per
    crossing). NPC drivers notice the car with probability ``attention``.

    Observation (``observation``):

    - ``privileged`` (211 values): the next goal in the heading frame (scaled 1/20),
      ``route`` (1/20), ``road``, ``lanes`` (1/20), ``lane_route``, ``signal`` (distance
      1/50), speed (1/10), steering angle, last action, ``traffic`` (8 nearest road users
      within 50 m, scaled 1/10, 15 values each) and ``pedestrians`` (8 within 30 m, 1/10, 6
      each);
    - ``sensors`` (199 values): the next goal, ``route``, ``signal`` (the light is not seen by
      the LiDAR), speed, steering, last action and the LiDAR (``LIDAR``, ``lidar_log``).

    Reward per step (``d``: horizontal distance to the next goal, measured before and after
    from the same goal):

    - ``progress_weight·(d_before − d_after)``;
    - ``−time_weight`` (every step);
    - ``−lane_weight·|lateral offset|`` from the lane centre;
    - ``−smooth_weight·‖Δa‖²`` (jerk);
    - ``−red_light_penalty`` on running a red light (``RED_LIGHT``), ``−wrong_way_penalty``
      and ``−off_road_penalty`` per step with ``WRONG_WAY`` / ``OFF_ROAD``;
    - ``finish_bonus`` at the route's end (within ``goal_radius``: success);
    - ``−terminal_penalty`` on a collision (another vehicle, a building or a pole), hitting
      a pedestrian, rollover, leaving the map or getting stuck (moving less than 0.5 m in
      ``stuck_time`` s; long enough for a red light).

    Episodes are truncated after ``episode_time`` (600 s). The policy runs at 20 Hz, the
    physics at 1 kHz. ``scripted_scenario()`` is the same scenario with the traffic driver
    flying the car along its route (``scripted_baseline``).
    """

    name = "car_urban_drive"
    default_episode_time = 600.0
    has_success = True
    failure_events = STUCK

    def __init__(
        self,
        *,
        distance: tuple[float, float] = (500.0, 1500.0),
        goal_step: float = 25.0,
        goal_radius: float = 4.0,
        max_speed: float = 14.0,
        observation: str = PRIVILEGED,
        lidar: dict[str, Any] | None = None,
        npcs: int = 30,
        buses: int = 2,
        cyclists: int = 6,
        parked: int = 20,
        pedestrians: int = 100,
        jaywalk: float = 0.02,
        attention: float = 0.9,
        stuck_time: float = 120.0,
        progress_weight: float = 0.2,
        time_weight: float = 0.01,
        lane_weight: float = 0.05,
        smooth_weight: float = 0.02,
        red_light_penalty: float = 5.0,
        wrong_way_penalty: float = 0.2,
        off_road_penalty: float = 0.2,
        finish_bonus: float = 20.0,
        terminal_penalty: float = 20.0,
        **kwargs: Any,
    ):
        if observation not in (PRIVILEGED, SENSORS):
            raise ValueError(f"observation must be {PRIVILEGED!r} or {SENSORS!r}, got {observation!r}")
        kwargs.setdefault("vehicle", "sedan_like")
        kwargs.setdefault("action_mode", "vk")
        kwargs.setdefault("map", "urban")
        kwargs.setdefault("policy_hz", 20)
        super().__init__(**kwargs)
        self.distance = distance
        self.goal_step = goal_step
        self.goal_radius = goal_radius
        self.max_speed = max_speed
        self.observation = observation
        self.lidar = {**LIDAR, **(lidar or {})}
        self.npcs = npcs
        self.buses = buses
        self.cyclists = cyclists
        self.parked = parked
        self.pedestrians = pedestrians
        self.jaywalk = jaywalk
        self.attention = attention
        self.stuck_time = stuck_time
        self.progress_weight = progress_weight
        self.time_weight = time_weight
        self.lane_weight = lane_weight
        self.smooth_weight = smooth_weight
        self.red_light_penalty = red_light_penalty
        self.wrong_way_penalty = wrong_way_penalty
        self.off_road_penalty = off_road_penalty
        self.finish_bonus = finish_bonus
        self.terminal_penalty = terminal_penalty

    # ------------------------------------------------------------------ scenario

    def group(self) -> dict[str, Any]:
        route_hint = [
            {"term": "goal_rel_heading", "scale": 0.05},
            {"term": "route", "scale": 0.05},
        ]
        own = [
            {"term": "speed", "scale": 0.1},
            {"term": "steering"},
            {"term": "last_action"},
        ]
        signal = {"term": "signal", "scale": 0.02}
        group: dict[str, Any] = {
            "goals": {
                "kind": "route",
                "distance": list(self.distance),
                "radius": self.goal_radius,
                "route": {"destination": "lanes", "step": self.goal_step},
            },
            "spawn": {"on_ground": True},
            "ground_action_limits": {"speed": self.max_speed},
        }
        if self.observation == PRIVILEGED:
            group["obs"] = [
                *route_hint,
                {"term": "road"},
                {"term": "lanes", "scale": 0.05},
                {"term": "lane_route"},
                signal,
                *own,
                {"term": "traffic", "count": 8, "range": 50.0, "scale": 0.1},
                {"term": "pedestrians", "count": 8, "range": 30.0, "scale": 0.1},
            ]
        else:
            group["sensors"] = [{"name": "lidar", "type": "lidar", **self.lidar}]
            group["obs"] = [*route_hint, signal, *own, {"term": "lidar_log", "sensor": "lidar"}]
        return group

    def settings(self) -> dict[str, Any]:
        sc: dict[str, Any] = {"events": {"bounds_margin": 5.0, "ground": {"stuck_time": self.stuck_time}}}
        if self.pedestrians > 0:
            sc["pedestrians"] = {"count": self.pedestrians, "jaywalk": self.jaywalk}
        return sc

    def scenario(self) -> dict[str, Any]:
        sc = super().scenario()
        traffic = {"type": "traffic", "attention": self.attention}
        npc = {"spawn": {"on_ground": True, "on_road": True, "min_separation": 20.0}, "disable_on_terminal": False}
        groups = [
            ("traffic", self.npcs, {"vehicle": "sedan_like", "physics": "hybrid", "driver": traffic}),
            (
                "buses",
                self.buses,
                {
                    "vehicle": "bus_city",
                    "physics": "kinematic",
                    "driver": {
                        **traffic,
                        "speed_factor": [0.8, 0.9],
                        "headway": [1.5, 2.5],
                        "accel": [0.5, 0.8],
                        "decel": [1.5, 2.0],
                        "lateral_accel": 1.5,
                        "safe_decel": 3.0,
                        "bus": {"length": [1500.0, 3000.0], "stop_spacing": 300.0, "dwell": [10.0, 30.0]},
                    },
                    "spawn": {"on_ground": True, "on_road": True, "min_separation": 40.0},
                },
            ),
            (
                "cyclists",
                self.cyclists,
                {
                    "vehicle": "bicycle_city",
                    "physics": "hybrid",
                    "driver": {**traffic, "speed": [4.0, 6.0], "lateral_accel": 1.5},
                },
            ),
            (
                "parked",
                self.parked,
                {
                    "vehicle": "sedan_like",
                    "physics": "kinematic",
                    "driver": {"type": "parked"},
                    "spawn": {"on_ground": True, "in_bays": True},
                },
            ),
        ]
        for name, count, spec in groups:
            if count > 0:
                sc["groups"].append({"name": name, "count": count, **npc, **spec})
        return sc

    def scripted_scenario(self) -> dict[str, Any]:
        """The scenario with the traffic driver flying the car along its route (no respawns)."""
        sc = self.scenario()
        sc["groups"][0] = {**sc["groups"][0], "driver": {"type": "traffic", "respawn": 0.0}}
        return sc

    # ------------------------------------------------------------------ episodes

    def bind(self, num_envs: int, policy_dt: float, act_dim: int) -> None:
        super().bind(num_envs, policy_dt, act_dim)
        self.prev_position = np.zeros((num_envs, 2))

    def reset(self, mask: np.ndarray | None = None, state: np.ndarray | None = None) -> None:
        super().reset(mask, state)
        if state is not None:
            m = slice(None) if mask is None else mask
            self.prev_position[m] = state[m, STATE["position"]][:, :2]

    def succeeded(self, state: np.ndarray, events: np.ndarray) -> np.ndarray:
        return (events & FINISHED) != 0

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        position = state[:, STATE["position"]][:, :2]
        goal = state[:, STATE["goal"]][:, :2]
        before = np.linalg.norm(goal - self.prev_position, axis=1)
        after = np.linalg.norm(goal - position, axis=1)
        self.prev_position[:] = position
        da = action - prev_action
        return (
            self.progress_weight * (before - after)
            - self.time_weight
            - self.lane_weight * np.abs(state[:, STATE["road"]][:, 0])
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
            - self.red_light_penalty * ((events & RED_LIGHT) != 0)
            - self.wrong_way_penalty * ((events & WRONG_WAY) != 0)
            - self.off_road_penalty * ((events & OFF_ROAD) != 0)
            + self.finish_bonus * ((events & FINISHED) != 0)
        )


def scripted_baseline(task: CarUrbanDrive, num_envs: int, seed: int = 0, num_threads: int = 0) -> dict[str, float]:
    """One episode per world of ``task.scripted_scenario()``: the share of cars that reached the
    route's end before any crash, the share that crashed (a terminal event) first, the share
    that ran a red light, and the mean time to the end and the route length per world of those
    that reached it (s, m)."""
    sim = BatchSim(json.dumps(task.scripted_scenario()), num_envs, seed, num_threads)
    sim.reset()
    done = np.zeros(num_envs, dtype=bool)
    success = np.zeros(num_envs, dtype=bool)
    crashed = np.zeros(num_envs, dtype=bool)
    red = np.zeros(num_envs, dtype=bool)
    time = np.zeros(num_envs)
    steps = max(1, round(task.episode_time / sim.policy_dt))
    for k in range(steps):
        sim.step([])
        ev = sim.events("agent").reshape(num_envs)
        red |= ~done & ((ev & RED_LIGHT) != 0)
        crash = ~done & ((ev & TERMINAL_EVENTS) != 0)
        reached = ~done & ~crash & ((ev & FINISHED) != 0)
        crashed |= crash
        success |= reached
        time[reached] = (k + 1) * sim.policy_dt
        done |= crash | reached
        if done.all():
            break
    return {
        "success": float(success.mean()),
        "crashed": float(crashed.mean()),
        "red_lights": float(red.mean()),
        "time": float(time[success].mean()) if success.any() else float("nan"),
    }
