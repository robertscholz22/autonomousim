"""TrailerReverse-v0: back a tractor's semitrailer into a bay at the far side of a farm yard,
with ``(v, κ)`` actions, and a scripted reversing driver that proves the task solvable."""

from typing import Any

import numpy as np

from autonomousim.scenario import STATE
from autonomousim.tasks.base import Task

#: One ring 1° down with 19 azimuths over the 180° behind the trailer, 30 m, 10 Hz, at the
#: trailer's tail (its frame: the kingpin is the origin; the tail lies 11.84 m behind it): the
#: buildings behind the bay and the yard's edges.
LIDAR = {
    "unit": 1,
    "pattern": {"type": "rings", "elevations": [-1.0], "azimuths": 19, "azimuth_fov": 180.0},
    "max_range": 30.0,
    "mount": {"position": [-11.9, 0.0, 0.5], "rotation": [0.0, 0.0, np.pi]},
}

#: Kingpin to the trailer's axle-group centre (m), for the scripted driver.
TRAILER_WHEELBASE = 7.6


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class TrailerReverse(Task):
    """The ``truck_6x4`` tractor with the ``semitrailer_3axle`` trailer in ``vk`` mode (speed up
    to ``max_speed`` either way, path curvature up to ``max_curvature``) starts in the yard of a
    farm on a generated ``rural`` map (a pool of ``map_count`` 512 m maps from ``map_seed``),
    in line, facing the yard's road, with the trailer's tail ``distance`` m ahead of a bay at the
    far side of the yard (``bay`` goals: the bay lies anywhere within ±4 m across the yard, the
    tail within ±2 m of the bay's line, the heading within ±10° of the yard's). It is to reverse
    the trailer into the bay: the tail within ``success_radius`` of the bay, the trailer's
    heading within ``success_heading`` degrees of the yard's, and the tractor stopped (below
    0.5 m/s).

    Observation (31 values): ``trailer_goal`` (the bay relative to the tail in the trailer's
    heading frame, and sin, cos of the heading error; all scaled 1/10), ``articulation`` (angles
    and rates), speed, steering angle, last action and a LiDAR scan from the trailer's tail
    (``LIDAR``; ``lidar`` replaces its settings).

    Reward per step (``d``: distance of the tail from the bay; ``ψ``: the trailer's heading
    error; ``φ``: the articulation):

    - ``progress_weight·(d_before − d_after)``;
    - ``−heading_weight·|ψ|·min(1, 5/d)`` (alignment counts near the bay);
    - ``−articulation_weight·φ²``;
    - ``−smooth_weight·‖Δa‖²``;
    - ``success_bonus`` on success, ``−terminal_penalty`` on a crash (the buildings behind
      the bay), a jackknife (70°), a rollover or the tail more than ``max_distance`` m from
      the bay.

    Episodes are truncated after ``episode_time`` (60 s). The policy runs at 20 Hz, the
    physics at 1 kHz. ``scripted(state)`` gives the actions of a reversing driver (a feedback
    law on the articulation angle) for comparison.
    """

    name = "trailer_reverse"
    default_episode_time = 60.0
    has_success = True

    def __init__(
        self,
        *,
        distance: tuple[float, float] = (12.0, 24.0),
        max_speed: float = 3.0,
        max_curvature: float = 0.09,
        success_radius: float = 1.0,
        success_heading: float = 5.0,
        max_distance: float = 35.0,
        lidar: dict[str, Any] | None = None,
        progress_weight: float = 1.0,
        heading_weight: float = 0.05,
        articulation_weight: float = 0.05,
        smooth_weight: float = 0.02,
        success_bonus: float = 20.0,
        terminal_penalty: float = 20.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("vehicle", "truck_6x4")
        kwargs.setdefault("action_mode", "vk")
        kwargs.setdefault("map", "rural")
        kwargs.setdefault("policy_hz", 20)
        super().__init__(**kwargs)
        self.distance = distance
        self.max_speed = max_speed
        self.max_curvature = max_curvature
        self.success_radius = success_radius
        self.success_heading = success_heading
        self.max_distance = max_distance
        self.lidar = LIDAR if lidar is None else lidar
        self.progress_weight = progress_weight
        self.heading_weight = heading_weight
        self.articulation_weight = articulation_weight
        self.smooth_weight = smooth_weight
        self.success_bonus = success_bonus
        self.terminal_penalty = terminal_penalty

    def group(self) -> dict[str, Any]:
        return {
            "trailers": ["semitrailer_3axle"],
            "goals": {"kind": "bay", "distance": list(self.distance)},
            "ground_action_limits": {
                "speed": self.max_speed,
                "reverse": self.max_speed,
                "curvature": self.max_curvature,
            },
            "sensors": [{"name": "lidar", "type": "lidar", **self.lidar}],
            "obs": [
                {"term": "trailer_goal", "scale": 0.1},
                {"term": "articulation"},
                {"term": "speed", "scale": 0.3},
                {"term": "steering"},
                {"term": "last_action"},
                {"term": "lidar_log", "sensor": "lidar"},
            ],
        }

    # ------------------------------------------------------------------ episodes

    def bind(self, num_envs: int, policy_dt: float, act_dim: int) -> None:
        super().bind(num_envs, policy_dt, act_dim)
        self.prev_distance = np.zeros(num_envs)

    def reset(self, mask: np.ndarray | None = None, state: np.ndarray | None = None) -> None:
        super().reset(mask, state)
        if state is not None:
            m = slice(None) if mask is None else mask
            self.prev_distance[m] = self.bay_errors(state[m])[0]

    @staticmethod
    def bay_errors(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance of the tail from the bay (m) and the trailer's heading error (rad)."""
        tail = state[:, STATE["tail"]]
        d = np.linalg.norm(tail[:, :2] - state[:, STATE["goal"]][:, :2], axis=1)
        return d, wrap(tail[:, 2] - state[:, STATE["goal_yaw"]][:, 0])

    def failed(self, state: np.ndarray) -> np.ndarray:
        return self.bay_errors(state)[0] > self.max_distance

    def succeeded(self, state: np.ndarray, events: np.ndarray) -> np.ndarray:
        d, psi = self.bay_errors(state)
        speed = np.linalg.norm(state[:, STATE["velocity"]], axis=1)
        return (d < self.success_radius) & (np.abs(psi) < np.radians(self.success_heading)) & (speed < 0.5)

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        d, psi = self.bay_errors(state)
        progress = self.prev_distance - d
        self.prev_distance[:] = d
        phi = state[:, STATE["articulation"]][:, 0]
        da = action - prev_action
        return (
            self.progress_weight * progress
            - self.heading_weight * np.abs(psi) * np.minimum(1.0, 5.0 / np.maximum(d, 1e-3))
            - self.articulation_weight * phi**2
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
            + self.success_bonus * self.succeeded(state, events)
        )

    # ------------------------------------------------------------------ scripted driver

    def scripted(
        self,
        state: np.ndarray,
        lookahead: float = 10.0,
        heading_gain: float = 0.15,
        max_articulation: float = 0.25,
        articulation_gain: float = 1.0,
        speed: float = 1.5,
    ) -> np.ndarray:
        """Normalised actions of a reversing driver for the state rows ``state``.

        The tail follows the bay's centre line backwards: it aims at the point ``lookahead``
        m behind its own projection on the line, which gives the trailer a heading to take; the trailer turns
        toward it at ``heading_gain`` rad per metre reversed per rad of error, which with the
        kinematic trailer (``θ₂' = (s/L)·sin φ`` reversing at speed ``s``) asks for the
        articulation ``φ* = asin(L·heading_gain·e)``; the tractor's curvature holds the
        articulation there, ``κ = −sin φ / L − articulation_gain·(φ − φ*)`` (which makes
        ``φ' = −s·articulation_gain·(φ − φ*)``). The speed falls from ``speed`` toward the bay
        and the driver stops on it.
        """
        L = TRAILER_WHEELBASE
        tail = state[:, STATE["tail"]]
        goal, yaw = state[:, STATE["goal"]][:, :2], state[:, STATE["goal_yaw"]][:, 0]
        c, s = np.cos(yaw), np.sin(yaw)
        d = tail[:, :2] - goal
        along, lateral = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        # The tail moves backward (heading + π) toward the look-ahead point.
        wanted = yaw + np.arctan2(-lateral, -lookahead) - np.pi
        e = wrap(wanted - tail[:, 2])
        phi_star = np.arcsin(np.clip(L * heading_gain * e, -1.0, 1.0)).clip(-max_articulation, max_articulation)
        phi = state[:, STATE["articulation"]][:, 0]
        kappa = -np.sin(phi) / L - articulation_gain * (phi - phi_star)
        v = -np.clip(0.3 * along, 0.3, speed)
        v[along < 0.1] = 0.0
        return np.stack([v / self.max_speed, np.clip(kappa / self.max_curvature, -1.0, 1.0)], 1).astype(np.float32)
