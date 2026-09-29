"""DroneLandOnCar-v0: find a car driving on rural roads with a downward camera, follow it and
land on its roof; and a scripted pilot that does so from the semantic image alone."""

from typing import Any

import numpy as np

from autonomousim._native import SEMANTIC_CLASSES
from autonomousim.events import Event
from autonomousim.scenario import STATE
from autonomousim.tasks.base import Task

LANDED = int(Event.LANDED)
#: Semantic class of other agents' vehicles in the camera image.
VEHICLE = SEMANTIC_CLASSES.index("vehicle")
#: Height of the drone's centre of mass above the car's chassis frame when it rests on the
#: roof (m; the ``sedan_like`` roof at 0.764 m plus the drone's feet).
ROOF = 0.88
#: ``velocity`` action full scale: horizontal and vertical speed (m/s).
SPEED_XY = 15.0
SPEED_Z = 3.0
# Scripted pilot (module constants, so that they can be tuned):
# - perception: the roof is the vehicle pixels within ROOF_BAND depth steps below the highest
#   one; it is the target once it has ROOF_PIXELS pixels;
# - tracking: α from ALPHA_HIGH (TRACK_HIGH m above the roof and higher) to ALPHA_LOW
#   (TRACK_LOW m and lower), β = BETA_RATIO·α²/(2 − α) (Kalata's steady-state relation times
#   a ratio that favours smoothing); the tracking is good while the mean residual is under
#   GOOD + GOOD_SLOPE·height (m);
# - horizontal: the car's velocity plus the offset times a gain (1/s) from CLOSE_HIGH to
#   CLOSE_LOW over the same heights, at most CLOSE_MAX (m/s); the command changes by at most
#   ACCEL (m/s²);
# - vertical: descends while the offset is under CONE + CONE_SLOPE·height (m), at DESCENT
#   times the height (1/s) within DESCENT_MIN–DESCENT_MAX (m/s), holds the height up to twice
#   that offset and climbs back beyond it (at 0.5 m/s, up to FOLLOW m above the roof);
# - search: climbs at SEARCH_CLIMB (m/s) while the car is out of sight, its velocity estimate
#   decaying at SEARCH_DECAY (1/s).
ROOF_BAND = 1.5
ROOF_PIXELS = 12
ALPHA_HIGH = 0.3
ALPHA_LOW = 0.7
TRACK_HIGH = 8.0
TRACK_LOW = 3.0
BETA_RATIO = 0.4
GOOD = 0.1
GOOD_SLOPE = 0.02
CLOSE_HIGH = 0.5
CLOSE_LOW = 1.0
CLOSE_MAX = 5.0
ACCEL = 6.0
CONE = 0.1
CONE_SLOPE = 0.15
DESCENT = 0.35
DESCENT_MIN = 0.8
DESCENT_MAX = 2.0
FOLLOW = 5.0
SEARCH_CLIMB = 1.5
SEARCH_DECAY = 1.0


def _rotation(q: np.ndarray) -> np.ndarray:
    """Rotation matrices ``[N, 3, 3]`` (body → world) of quaternions ``[N, 4]`` (x, y, z, w)."""
    x, y, z, w = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


class DroneLandOnCar(Task):
    """An ``iris_like`` drone in ``velocity`` mode (horizontal speed in the heading frame up to
    15 m/s, vertical up to 3 m/s, yaw rate) lands on the roof of a scripted car
    (``sedan_like``, the ``road`` driver) that drives random routes on ``rural`` training maps
    at a cruise speed drawn from ``car_speed`` per leg (``car_stops``: with random stops; the
    drone flies at up to 12 m/s).
    The drone starts ``spawn_agl`` above the ground within ``spawn_offset`` (horizontally, each
    axis) of the car, so the car is in the camera's view at the start (trees along the roads
    may hide it), in a mean wind drawn from ``wind`` (m/s).

    Observation (``Dict``): ``image`` is the ``image_size``² downward camera (``fov_deg``,
    ``camera_hz``) as RGB plus depth (over ``depth_range``; ``semantic=True`` adds the class
    image as a fifth channel, which the scripted pilot needs); ``state`` is the drone's
    attitude (rot6d), body velocity, body rates, height above ground and last action. The car's
    position is not observed.

    Reward per step, from the privileged relative position (the car's state rows): the
    decrease of the potential ``φ = √(d² + (2h)²)`` (``d``: horizontal distance to the car,
    ``h``: height above its roof) times ``progress_weight``, ``−time_weight``,
    ``−smooth_weight·‖Δa‖²``, ``landing_bonus`` on success and ``−terminal_penalty`` on
    failure.

    Success: ``LANDED`` with the car as ``support`` (at rest relative to it). Failure: a crash
    (terrain, obstacles or the car: touching it with the airframe or faster than 2 m/s), water,
    leaving the map, landing anywhere else, or losing the car: its centre outside the camera's
    field of view (or more than ``depth_range`` below) for ``lost_time`` s. Episodes are
    truncated after ``episode_time`` (40 s). The policy and the camera run at 25 Hz.

    ``scripted(obs)`` flies from the camera image only (needs ``semantic=True``): the vehicle
    pixels and their depth give the roof's position relative to the drone, whose change gives
    the car's velocity; the pilot follows the car, closing in, and descends while above the
    roof. It lands in about two thirds of the episodes.
    """

    name = "land_on_car"
    default_episode_time = 40.0
    has_success = True

    def __init__(
        self,
        *,
        car_speed: tuple[float, float] = (3.0, 8.0),
        car_stops: bool = False,
        spawn_agl: tuple[float, float] = (15.0, 30.0),
        spawn_offset: float = 10.0,
        image_size: int = 64,
        fov_deg: float = 90.0,
        camera_hz: int = 25,
        depth_range: float = 40.0,
        semantic: bool = False,
        lost_time: float = 5.0,
        progress_weight: float = 0.1,
        time_weight: float = 0.01,
        smooth_weight: float = 0.01,
        landing_bonus: float = 20.0,
        terminal_penalty: float = 10.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("vehicle", "iris_like")
        kwargs.setdefault("action_mode", "velocity")
        kwargs.setdefault("map", "rural")
        kwargs.setdefault("map_count", 8)
        kwargs.setdefault("policy_hz", 25)
        kwargs.setdefault("wind", (0.0, 3.0))
        super().__init__(**kwargs)
        self.car_speed = car_speed
        self.car_stops = car_stops
        self.spawn_agl = spawn_agl
        self.spawn_offset = spawn_offset
        self.image_size = image_size
        self.fov_deg = fov_deg
        self.camera_hz = camera_hz
        self.depth_range = depth_range
        self.semantic = semantic
        self.lost_time = lost_time
        self.progress_weight = progress_weight
        self.time_weight = time_weight
        self.smooth_weight = smooth_weight
        self.landing_bonus = landing_bonus
        self.terminal_penalty = terminal_penalty
        if self.obs is None:
            self.obs = [
                {"term": "camera", "sensor": "down", "output": "rgb"},
                {"term": "camera", "sensor": "down", "output": "depth", "range": depth_range},
                {"term": "rot6d"},
                {"term": "lin_vel_body"},
                {"term": "ang_vel_body"},
                {"term": "agl"},
                {"term": "last_action"},
            ]
            if semantic:
                self.obs.insert(2, {"term": "camera", "sensor": "down", "output": "semantic"})
        self.car = np.zeros((0, 0))

    # ------------------------------------------------------------------ scenario

    def group(self) -> dict[str, Any]:
        camera = {
            "name": "down",
            "type": "camera",
            "width": self.image_size,
            "height": self.image_size,
            "fov_deg": self.fov_deg,
            "rate_hz": self.camera_hz,
            # Under the hub, just above the feet, looking straight down.
            "mount": {"position": [0.0, 0.0, -0.08], "rotation": [0.0, np.pi / 2, 0.0]},
        }
        return {
            "spawn": {
                "agl": list(self.spawn_agl),
                "near": {"group": "car", "offset": self.spawn_offset},
                "min_separation": 0.0,
                "clearance": 2.0,
            },
            "action_limits": {"speed_xy": SPEED_XY, "speed_z": SPEED_Z},
            "sensors": [camera],
        }

    def scenario(self) -> dict[str, Any]:
        sc = super().scenario()
        driver: dict[str, Any] = {"type": "road", "speed": list(self.car_speed)}
        if self.car_stops:
            driver["stop_rate"] = 0.02
        car = {
            "name": "car",
            "count": 1,
            "vehicle": "sedan_like",
            "spawn": {"on_road": True, "on_ground": True},
            "driver": driver,
        }
        # The car comes first: the drone spawns around it.
        sc["groups"] = [car, *[g for g in sc["groups"] if g["name"] != "car"]]
        return sc

    # ------------------------------------------------------------------ episodes

    def attach(self, sim: Any) -> None:
        g = next(g for g in range(sim.num_groups) if sim.group_info(g)["name"] == "car")
        self.car = sim.state(g)[:, 0, :]
        self.car_index = int(sim.group_info(g)["first_agent"])
        learning = next(g for g in range(sim.num_groups) if not sim.group_info(g)["scripted"])
        info = sim.group_info(learning)
        self.obs_slices = {name: slice(o, o + n) for name, o, n in info["obs_layout"]}
        self.channels = {name: o for name, o, _ in info["image_layout"]}

    def bind(self, num_envs: int, policy_dt: float, act_dim: int) -> None:
        super().bind(num_envs, policy_dt, act_dim)
        self.potential = np.zeros(num_envs)
        self.lost = np.zeros(num_envs, dtype=np.int64)
        self.events = np.zeros(num_envs, dtype=np.uint32)
        # Scripted pilot: the roof's tracked horizontal position relative to the drone, the
        # car's velocity and the pilot's command (world frame), the height above the roof.
        self.tracking = np.zeros(num_envs, dtype=bool)
        self.rel = np.zeros((num_envs, 2))
        self.car_vel = np.zeros((num_envs, 2))
        self.command = np.zeros((num_envs, 2))
        self.height = np.zeros(num_envs)
        self.car_vz = np.zeros(num_envs)
        self.seen = np.zeros(num_envs, dtype=bool)
        self.kind = np.zeros(num_envs, dtype=np.int64)
        self.residual = np.ones(num_envs)

    def reset(self, mask: np.ndarray | None = None, state: np.ndarray | None = None) -> None:
        super().reset(mask, state)
        m = slice(None) if mask is None else mask
        self.lost[m] = 0
        self.tracking[m] = False
        self.seen[m] = False
        self.residual[m] = 1.0
        self.command[m] = 0.0
        if state is not None:
            self.potential[m] = self._potential(state)[m]

    def relative(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal distance from the drone to the car and height above its roof (m)."""
        d = self.car[:, STATE["position"]] - state[:, STATE["position"]]
        return np.hypot(d[:, 0], d[:, 1]), -d[:, 2] - ROOF

    def _potential(self, state: np.ndarray) -> np.ndarray:
        d, h = self.relative(state)
        return np.hypot(d, 2.0 * np.maximum(h, 0.0))

    def in_view(self, state: np.ndarray) -> np.ndarray:
        """Whether the car's centre lies in the camera's field of view and depth range."""
        rel = self.car[:, STATE["position"]] - state[:, STATE["position"]]
        body = np.einsum("nji,nj->ni", _rotation(state[:, STATE["orientation"]]), rel)
        depth = -body[:, 2]
        t = np.tan(np.radians(self.fov_deg) / 2)
        lateral = np.maximum(np.abs(body[:, 0]), np.abs(body[:, 1]))
        return (depth > 0) & (depth < self.depth_range) & (lateral <= t * depth)

    def compute(
        self, state: np.ndarray, events: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.events = events
        return super().compute(state, events, actions)

    def _on_car(self, state: np.ndarray) -> np.ndarray:
        return state[:, STATE["support"]][:, 0] == self.car_index

    def failed(self, state: np.ndarray) -> np.ndarray:
        self.lost = np.where(self.in_view(state), 0, self.lost + 1)
        lost = self.lost * self.policy_dt > self.lost_time
        landed_elsewhere = ((self.events & LANDED) != 0) & ~self._on_car(state)
        return lost | landed_elsewhere

    def succeeded(self, state: np.ndarray, events: np.ndarray) -> np.ndarray:
        return ((events & LANDED) != 0) & self._on_car(state)

    def reward(
        self, state: np.ndarray, action: np.ndarray, prev_action: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        phi = self._potential(state)
        r = self.progress_weight * (self.potential - phi)
        self.potential = phi
        r -= self.time_weight + self.smooth_weight * np.sum((action - prev_action) ** 2, axis=1)
        r[self.succeeded(state, events)] += self.landing_bonus
        return r

    # ------------------------------------------------------------------ scripted pilot

    def scripted(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Normalised ``velocity`` actions of a pilot that sees the car only in the camera
        image (``semantic=True``; the constants are this module's).

        Every pixel's depth, with the drone's attitude, places its surface point relative to
        the camera. The target is the car's roof (the vehicle pixels close below the highest
        one) once it is resolved or the silhouette overflows the image, else the silhouette;
        its mean point measures the car's horizontal position relative to the drone, the
        highest point the height above the roof. Alpha-beta trackers smooth both and estimate
        the car's horizontal and vertical velocity, with gains that grow closer to the roof,
        where the measurements get more precise (the horizontal velocity from consecutive frames
        of the same target, at half the gain while it touches the image's edge). Out of sight,
        they predict, the velocity estimates decaying.

        The pilot flies the car's velocity plus the offset times a gain (0.5/s high up, 1/s
        close above the roof). It descends, relative to the car, while the offset is within a
        cone over the roof (0.1 m plus 0.15 m per metre of height) and the tracking is good,
        faster when higher; it holds the height up to twice the cone's radius and climbs back
        to ``FOLLOW`` m above the roof beyond, and climbs while the car is out of sight.
        """
        if "down/semantic" not in self.channels:
            raise ValueError("the scripted pilot needs the semantic image (semantic=True)")
        image, s = obs["image"], obs["state"]
        n, size = len(image), self.image_size
        dt = self.policy_dt
        vehicle = image[..., self.channels["down/semantic"]] == VEHICLE
        step = self.depth_range / 255.0
        depth = image[..., self.channels["down/depth"]].astype(np.float64) * step

        def touches(m: np.ndarray) -> np.ndarray:
            return m[:, 0].any(1) | m[:, -1].any(1) | m[:, :, 0].any(1) | m[:, :, -1].any(1)

        r6 = s[:, self.obs_slices["rot6d"]]
        c0, c1 = r6[:, 0:3], r6[:, 3:6]
        rot = np.stack([c0, c1, np.cross(c0, c1)], axis=2)
        # Every pixel's surface point relative to the camera (world frame, m). The camera
        # (optical axis x, image right −y, image down −z) looks down: image up is body forward,
        # image right body right; pixel centres relative to the principal point in focal
        # lengths, times the depth along the axis.
        f = size / 2 / np.tan(np.radians(self.fov_deg) / 2)
        grid = (np.arange(size) + 0.5 - size / 2) / f
        u, v = np.broadcast_arrays(grid[None, None, :], grid[:, None][None])
        body = np.stack([-v * depth, -u * depth, -depth], axis=-1)
        points = np.einsum("nij,nhwj->nhwi", rot, body)
        # The roof: the vehicle pixels within ROOF_BAND of the highest one.
        drop = np.where(vehicle, -points[..., 2], np.inf)
        top = drop.min(axis=(1, 2))
        roof = vehicle & (drop <= top[:, None, None] + ROOF_BAND * step)
        # 0: silhouette, 1: roof (once it is resolved, or the silhouette overflows the image).
        kind = (touches(vehicle) | (roof.sum(axis=(1, 2)) >= ROOF_PIXELS)).astype(np.int64)
        target = np.where(kind[:, None, None] == 1, roof, vehicle)
        count = target.sum(axis=(1, 2))
        seen = count > 0
        same = seen & (kind == self.kind) & self.seen
        fresh = same & ~touches(target)
        self.seen, self.kind = seen, kind
        # The target's mean surface point, at the roof's height.
        rel = (target[..., None] * points).sum(axis=(1, 2)) / np.maximum(count, 1)[:, None]
        rel[:, 2] = np.where(seen, -top, 0.0)
        own3 = np.einsum("nij,nj->ni", rot, s[:, self.obs_slices["lin_vel_body"]])
        own = own3[:, :2]

        # Alpha-beta tracker of the target's horizontal position relative to the drone and the
        # car's velocity (world frame).
        start = seen & ~self.tracking
        self.rel[start] = rel[start, :2]
        self.car_vel[start] = 0.0
        self.height[start] = -rel[start, 2]
        self.car_vz[start] = 0.0
        self.tracking |= seen
        predicted = self.rel + (self.car_vel - own) * dt
        residual = np.where(seen[:, None], rel[:, :2] - predicted, 0.0)
        # Measurements get more precise closer to the roof (pixel size ∝ height).
        t = np.clip((self.height - TRACK_LOW) / (TRACK_HIGH - TRACK_LOW), 0.0, 1.0)
        alpha = ALPHA_LOW + (ALPHA_HIGH - ALPHA_LOW) * t
        self.rel = predicted + alpha[:, None] * residual
        # Centroids of targets cut by the image's edge are biased, but still show a car that
        # brakes or turns under a low drone: they count at half the gain.
        beta = BETA_RATIO * alpha**2 / (2.0 - alpha) * np.where(fresh, 1.0, np.where(same, 0.5, 0.0))
        self.car_vel += beta[:, None] / dt * residual
        # The same for the height above the roof and the car's vertical velocity.
        predicted_h = self.height + (own3[:, 2] - self.car_vz) * dt
        residual_h = np.where(seen, -rel[:, 2] - predicted_h, 0.0)
        self.height = predicted_h + 0.3 * residual_h
        self.car_vz -= np.where(seen, 0.02, 0.0) / dt * residual_h
        self.residual += 0.2 * (np.where(seen, np.linalg.norm(residual, axis=1), 1.0) - self.residual)

        self.car_vel[~seen] *= np.exp(-SEARCH_DECAY * dt)
        self.car_vz[~seen] *= np.exp(-SEARCH_DECAY * dt)
        gain = CLOSE_LOW + (CLOSE_HIGH - CLOSE_LOW) * t
        closing = gain[:, None] * self.rel
        closing *= np.minimum(1.0, CLOSE_MAX / np.maximum(np.linalg.norm(closing, axis=1), 1e-9))[:, None]
        change = self.car_vel + closing - self.command
        change *= np.minimum(1.0, ACCEL * dt / np.maximum(np.linalg.norm(change, axis=1), 1e-9))[:, None]
        self.command += change

        # Descend while inside the cone over the roof (and tracking well), hold the height
        # up to twice its radius, climb back (to FOLLOW at most) beyond.
        offset = np.linalg.norm(self.rel, axis=1)
        cone = CONE + CONE_SLOPE * self.height
        down = -np.clip(DESCENT * self.height, DESCENT_MIN, DESCENT_MAX)
        up = np.where(self.height < FOLLOW, 0.5, 0.0)
        good = self.residual < GOOD + GOOD_SLOPE * self.height
        vz = np.where((offset < cone) & good, down, np.where(offset < 2.0 * cone, 0.0, up))
        vz = np.where(seen, self.car_vz + vz, SEARCH_CLIMB)
        # World → heading frame.
        yaw = np.arctan2(c0[:, 1], c0[:, 0])
        cy, sy = np.cos(yaw), np.sin(yaw)
        a = np.zeros((n, 4), dtype=np.float32)
        a[:, 0] = (cy * self.command[:, 0] + sy * self.command[:, 1]) / SPEED_XY
        a[:, 1] = (-sy * self.command[:, 0] + cy * self.command[:, 1]) / SPEED_XY
        a[:, 2] = vz / SPEED_Z
        return np.clip(a, -1.0, 1.0)
