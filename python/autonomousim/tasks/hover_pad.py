"""QuadHoverPad-v0: find a landing pad with the down camera and hover over it."""

from typing import Any

import numpy as np

from autonomousim.tasks.hover import QuadHover

#: Observation: the down camera's colour image plus the drone's own state, without the goal.
DEFAULT_OBS = [
    {"term": "camera", "sensor": "down", "output": "rgb"},
    {"term": "rot6d"},
    {"term": "lin_vel_body"},
    {"term": "ang_vel_body"},
    {"term": "agl"},
    {"term": "last_action"},
]


class QuadHoverPad(QuadHover):
    """A pixel task: a landing pad of ``pad_radius`` lies on flat grass up to
    ``goal_distance`` from the spawn point, and the drone must hold ``hover_agl`` above it.
    The drone does not observe the pad's position: it sees the ground through a downward
    camera (``image_size`` pixels square, ``fov_deg``, ``camera_hz``) and knows its own
    attitude, velocity, rates, height above ground and last action (the ``state`` part of the
    ``Dict`` observation). ``depth=True`` adds the depth image (over ``depth_range``) as a
    fourth channel.

    Reward, end conditions and the other arguments are those of ``QuadHover``, measured from
    the point ``hover_agl`` above the pad; actions are ``velocity`` commands by default.
    """

    name = "hover_pad"
    default_episode_time = 10.0

    def __init__(
        self,
        *,
        goal_distance: tuple[float, float] = (0.0, 1.5),
        agl: tuple[float, float] = (2.0, 3.5),
        hover_agl: float = 2.5,
        pad_radius: float = 0.75,
        image_size: int = 32,
        fov_deg: float = 90.0,
        camera_hz: int = 50,
        depth: bool = False,
        depth_range: float = 10.0,
        tilt_deg: float = 10.0,
        speed: float = 0.5,
        rates: float = 0.5,
        action_mode: str = "velocity",
        **kwargs: Any,
    ):
        super().__init__(
            goal_distance=goal_distance,
            agl=agl,
            tilt_deg=tilt_deg,
            speed=speed,
            rates=rates,
            action_mode=action_mode,
            **kwargs,
        )
        self.hover_agl = hover_agl
        self.pad_radius = pad_radius
        self.image_size = image_size
        self.fov_deg = fov_deg
        self.camera_hz = camera_hz
        self.depth = depth
        self.depth_range = depth_range
        if self.obs is None:
            self.obs = list(DEFAULT_OBS)
            if depth:
                self.obs.insert(1, {"term": "camera", "sensor": "down", "output": "depth", "range": depth_range})

    def group(self) -> dict[str, Any]:
        g = super().group()
        g["goals"] = {**g["goals"], "agl": [self.hover_agl, self.hover_agl], "pad": self.pad_radius}
        camera = {
            "name": "down",
            "type": "camera",
            "width": self.image_size,
            "height": self.image_size,
            "fov_deg": self.fov_deg,
            "rate_hz": self.camera_hz,
            # Under the hub, looking straight down.
            "mount": {"position": [0.0, 0.0, -0.02], "rotation": [0.0, np.pi / 2, 0.0]},
        }
        g["sensors"] = [camera]
        return g
