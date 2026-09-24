"""QuadWaypointForest-v0: fly through a chain of waypoints in generated forests with LiDAR."""

from typing import Any

import numpy as np

from autonomousim.events import Event
from autonomousim.scenario import STATE
from autonomousim.tasks.base import Task

GOAL_REACHED = int(Event.GOAL_REACHED)
FINISHED = int(Event.FINISHED)
FOLIAGE = int(Event.FOLIAGE)

#: 4 rings (−24°, −8°, 8°, 24°) × 32 azimuths, 40 m, 10 Hz: twice the azimuth resolution of
#: ``rl64``, so that trunks a few metres ahead fall between fewer beams.
LIDAR = {"pattern": {"type": "rings", "elevations": [-24.0, -8.0, 8.0, 24.0], "azimuths": 32, "azimuth_fov": 360.0}}


class QuadWaypointForest(Task):
    """An iris-like quadrotor in ``velocity`` mode spawns 2–4 m above the ground of a
    generated wild map (a pool of ``map_count`` 512 m maps from ``map_seed``; evaluate on
    another ``map_seed`` for unseen maps) and flies through ``goals`` waypoints, each
    ``goal_distance`` from the previous one at ``goal_agl`` above the ground. A waypoint
    counts once the vehicle's centre is within ``goal_radius``; the next one then becomes
    the goal. Flying higher than ``max_agl`` above the ground ends the episode
    (``OUT_OF_BOUNDS``), so the forest has to be crossed below the treetops.

    Observation: goal in the body frame (scaled 1/20, clipped to ±3), body velocity and
    rates, rot6d, height above ground, last action and a LiDAR scan (``LIDAR``: 4 rings × 32
    azimuths, 40 m, log ranges; ``lidar`` replaces its settings): 148 values.

    Reward per step (``d``: distance to the current goal, ``c``: clearance to the nearest
    terrain or solid obstacle, from the state row):

    - ``progress_weight·(d_before − d_after)``, measured to the current goal from the
      positions before and after the step (so switching goals does not jump);
    - ``goal_bonus`` per waypoint reached;
    - ``−proximity_weight·max(0, 1 − c/proximity_distance)²``;
    - ``−closing_weight·max(0, −ċ)·max(0, 1 − c/closing_distance)``: the speed at which the
      clearance shrinks (the approach toward the nearest obstacle or the ground, not flying
      past it), near obstacles;
    - ``−foliage_weight`` on steps inside tree canopies;
    - ``−smooth_weight·‖Δa‖²``;
    - ``−terminal_penalty`` on a crash, water, leaving the map or the height limit.

    The episode succeeds when the last waypoint is reached and is truncated after
    ``episode_time`` (60 s). The policy runs at 25 Hz.
    """

    name = "waypoint_forest"
    default_episode_time = 60.0
    has_success = True

    def __init__(
        self,
        *,
        goals: int = 3,
        goal_distance: tuple[float, float] = (20.0, 50.0),
        goal_agl: tuple[float, float] = (2.0, 5.0),
        goal_radius: float = 2.0,
        spawn_agl: tuple[float, float] = (2.0, 4.0),
        max_agl: float = 10.0,
        lidar: dict[str, Any] | None = None,
        progress_weight: float = 1.0,
        goal_bonus: float = 10.0,
        proximity_weight: float = 0.2,
        proximity_distance: float = 1.5,
        closing_weight: float = 0.0,
        closing_distance: float = 3.0,
        foliage_weight: float = 0.1,
        smooth_weight: float = 0.02,
        terminal_penalty: float = 50.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("vehicle", "iris_like")
        kwargs.setdefault("action_mode", "velocity")
        kwargs.setdefault("map", "wild")
        kwargs.setdefault("policy_hz", 25)
        kwargs.setdefault("wind", (0.0, 3.0))
        super().__init__(**kwargs)
        self.goals = goals
        self.goal_distance = goal_distance
        self.goal_agl = goal_agl
        self.goal_radius = goal_radius
        self.spawn_agl = spawn_agl
        self.max_agl = max_agl
        self.lidar = LIDAR if lidar is None else lidar
        self.progress_weight = progress_weight
        self.goal_bonus = goal_bonus
        self.proximity_weight = proximity_weight
        self.proximity_distance = proximity_distance
        self.closing_weight = closing_weight
        self.closing_distance = closing_distance
        self.foliage_weight = foliage_weight
        self.smooth_weight = smooth_weight
        self.terminal_penalty = terminal_penalty

    def group(self) -> dict[str, Any]:
        return {
            "spawn": {"agl": list(self.spawn_agl), "clearance": 3.0, "margin": 40.0},
            "goals": {
                "kind": "random",
                "count": self.goals,
                "distance": list(self.goal_distance),
                "agl": list(self.goal_agl),
                "clearance": 3.0,
                "margin": 40.0,
                "radius": self.goal_radius,
            },
            "sensors": [{"name": "lidar", "type": "lidar", **self.lidar}],
            "obs": [
                {"term": "goal_rel_body", "scale": 0.05, "clip": 3.0},
                {"term": "lin_vel_body", "scale": 0.2},
                {"term": "ang_vel_body", "scale": 0.2},
                {"term": "rot6d"},
                {"term": "agl", "scale": 0.1, "clip": 3.0},
                {"term": "last_action"},
                {"term": "lidar_log", "sensor": "lidar"},
            ],
        }

    def settings(self) -> dict[str, Any]:
        return {"events": {"crash_speed": 2.0, "bounds_margin": 5.0, "max_agl": self.max_agl}}

    # ------------------------------------------------------------------ episodes

    def bind(self, num_envs: int, policy_dt: float, act_dim: int) -> None:
        super().bind(num_envs, policy_dt, act_dim)
        self.prev_position = np.zeros((num_envs, 3))
        self.prev_clearance = np.zeros(num_envs)

    def reset(self, mask: np.ndarray | None = None, state: np.ndarray | None = None) -> None:
        super().reset(mask, state)
        if state is not None:
            m = slice(None) if mask is None else mask
            self.prev_position[m] = state[m, STATE["position"]]
            self.prev_clearance[m] = state[m, STATE["clearance"]][:, 0]

    def succeeded(self, state: np.ndarray, events: np.ndarray) -> np.ndarray:
        return (events & FINISHED) != 0

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        position = state[:, STATE["position"]]
        goal = state[:, STATE["goal"]]
        before = np.linalg.norm(goal - self.prev_position, axis=1)
        after = np.linalg.norm(goal - position, axis=1)
        self.prev_position[:] = position
        clearance = state[:, STATE["clearance"]][:, 0]
        near = np.clip(1.0 - clearance / self.proximity_distance, 0.0, 1.0)
        closing = np.maximum(0.0, self.prev_clearance - clearance) / self.policy_dt
        closing *= np.clip(1.0 - clearance / self.closing_distance, 0.0, 1.0)
        self.prev_clearance[:] = clearance
        da = action - prev_action
        return (
            self.progress_weight * (before - after)
            + self.goal_bonus * ((events & GOAL_REACHED) != 0)
            - self.proximity_weight * near**2
            - self.closing_weight * closing
            - self.foliage_weight * ((events & FOLIAGE) != 0)
            - self.smooth_weight * np.einsum("ij,ij->i", da, da)
        )
