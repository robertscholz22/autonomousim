"""Camera observations: ``Dict`` spaces, in-place image buffers, vector semantics, seeding and
the multi-agent and PettingZoo wrappers (rendered on lavapipe, see ``conftest.py``)."""

import base64
import json

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.vector import AutoresetMode
from mcap.reader import make_reader
from pettingzoo.test import parallel_api_test

import autonomousim
from autonomousim import SEMANTIC_CLASSES, _native
from autonomousim.multiagent import MultiAgentVectorEnv
from autonomousim.pettingzoo import parallel_env
from autonomousim.scenario import STATE
from autonomousim.tasks import QuadHover, QuadHoverPad, make_task
from autonomousim.tasks.land_on_car import ROOF
from autonomousim.tasks.hover_pad import DEFAULT_OBS
from autonomousim.tasks.multi import MultiAgentTask, Team
from autonomousim.vector_env import AutonomousimVectorEnv

SEMANTIC = {"term": "camera", "sensor": "down", "output": "semantic"}
DEPTH = {"term": "camera", "sensor": "down", "output": "depth", "range": 10.0}
MARKER = SEMANTIC_CLASSES.index("marker")
VEHICLE = SEMANTIC_CLASSES.index("vehicle")
RURAL = {"type": "rural", "seed": 0, "count": 2, "cache": False}


def test_the_software_adapter_renders():
    assert "llvmpipe" in autonomousim.render_adapter()
    # The context exists now: another choice is refused and the adapter kept.
    assert not autonomousim.set_render_adapter("auto")
    assert SEMANTIC_CLASSES[:2] == ["sky", "grass"] and MARKER == 14


def test_native_image_buffers():
    task = QuadHoverPad(obs=[SEMANTIC, *DEFAULT_OBS, DEPTH], image_size=24)
    sim = _native.BatchSim(json.dumps(task.scenario()), 3, 0, 2)
    info = sim.group_info(0)
    assert info["image_shape"] == (24, 24, 5)
    assert info["image_layout"] == [("down/semantic", 0, 1), ("down/rgb", 1, 3), ("down/depth", 4, 1)]
    assert info["obs_dim"] == 17  # camera terms are not in the flat vector
    images = sim.images(0)
    assert images.shape == (3, 1, 24, 24, 5) and images.dtype == np.uint8
    # Rendered at the reset, overwritten in place by every step.
    assert (images[..., 0] == MARKER).any()
    before = images.copy()
    sim.step(np.full((3, 1, 4), 0.5, np.float32))
    assert sim.images(0) is not None and not np.array_equal(images, before)
    # Groups without camera terms have no images.
    plain = _native.BatchSim(json.dumps(QuadHover().scenario()), 1, 0, 1)
    assert plain.images(0) is None
    assert plain.group_info(0)["image_shape"] is None and plain.group_info(0)["image_layout"] == []


def test_check_env():
    env = gym.make("autonomousim/QuadHoverPad-v0", image_size=16)
    check_env(env.unwrapped, skip_render_check=True)
    obs, _ = env.reset(seed=3)
    assert env.observation_space.contains(obs)
    assert set(obs) == {"state", "image"} and obs["image"].shape == (16, 16, 3)
    env.close()


def test_vector_spaces_and_views():
    envs = gym.make_vec("autonomousim/QuadHoverPad-v0", num_envs=4, image_size=16, obs=[*DEFAULT_OBS, SEMANTIC])
    space = envs.single_observation_space
    assert isinstance(space, gym.spaces.Dict)
    assert space["image"] == gym.spaces.Box(0, 255, (16, 16, 4), np.uint8)
    assert envs.observation_space["image"].shape == (4, 16, 16, 4)
    obs, _ = envs.reset(seed=0)
    assert envs.observation_space.contains(obs)
    # Every world sees its pad.
    assert ((obs["image"][..., 3] == MARKER).reshape(4, -1).any(axis=1)).all()
    obs2, *_ = envs.step(np.zeros((4, 4), np.float32))
    assert obs2["image"] is not obs["image"] and not np.shares_memory(obs2["image"], obs["image"])
    envs.close()

    # copy=False returns the buffers themselves.
    envs = AutonomousimVectorEnv(2, "hover_pad", image_size=16, copy=False)
    a, _ = envs.reset(seed=0)
    b, *_ = envs.step(np.zeros((2, 4), np.float32))
    assert a["image"] is b["image"] and a["state"] is b["state"]
    envs.close()


def test_seeding_reproduces_images():
    def run(seed):
        envs = AutonomousimVectorEnv(3, "hover_pad", image_size=16, obs=[*DEFAULT_OBS, DEPTH], num_threads=2)
        out = [envs.reset(seed=seed)[0]["image"]]
        for k in range(5):
            out.append(envs.step(np.full((3, 4), 0.1 * k, np.float32))[0]["image"])
        envs.close()
        return np.stack(out)

    a, b, c = run(7), run(7), run(8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("mode", [AutoresetMode.SAME_STEP, AutoresetMode.DISABLED])
def test_final_obs_and_reset_masks(mode):
    envs = AutonomousimVectorEnv(3, "hover_pad", image_size=16, episode_time=0.1, autoreset_mode=mode)  # 5 steps
    envs.reset(seed=0)
    act = np.zeros((3, 4), np.float32)
    for _ in range(5):
        obs, _, terminated, truncated, info = envs.step(act)
    assert truncated.all()
    if mode == AutoresetMode.SAME_STEP:
        final = info["final_obs"]
        assert set(final) == {"state", "image"} and final["image"].shape == (3, 16, 16, 3)
        assert info["_final_obs"].all()
        # The new episodes start elsewhere, so they see something else.
        assert not np.array_equal(final["image"], obs["image"])
    else:
        assert "final_obs" not in info
        mask = np.array([True, False, False])
        before = obs["image"].copy()
        obs, _ = envs.reset(options={"reset_mask": mask})
        assert not np.array_equal(obs["image"][0], before[0])
        assert np.array_equal(obs["image"][1:], before[1:])
    envs.close()


def pad_team():
    return MultiAgentTask(
        teams={"drones": Team(QuadHoverPad(image_size=16), 2), "plain": Team(QuadHover(action_mode="velocity"), 1)},
        overrides={"groups": [{"spawn": {"min_separation": 3.0}}]},
        episode_time=2.0,
    )


def test_multi_agent_dict_observations():
    envs = MultiAgentVectorEnv(2, pad_team(), seed=0)
    space = envs.observation_space("drones")
    assert isinstance(space, gym.spaces.Dict) and space["image"].shape == (2, 2, 16, 16, 3)
    assert isinstance(envs.observation_space("plain"), gym.spaces.Box)
    obs, _ = envs.reset(seed=1)
    assert set(obs["drones"]) == {"state", "image"} and obs["drones"]["image"].shape == (2, 2, 16, 16, 3)
    assert obs["plain"].shape == (2, 1, envs.obs_dim["plain"])
    assert space.contains(obs["drones"])
    # Each drone sees its own pad (their goals differ).
    img = obs["drones"]["image"]
    assert not np.array_equal(img[:, 0], img[:, 1])
    actions = {g: np.zeros(envs.action_space(g).shape, np.float32) for g in envs.groups}
    done = False
    while not done:
        obs, _, _, truncated, info = envs.step(actions)
        done = bool(truncated.all())
    assert info["final_obs"]["drones"]["image"].shape == (2, 2, 16, 16, 3)
    envs.close()


def test_pettingzoo_api_with_images():
    env = parallel_env(pad_team())
    parallel_api_test(env, num_cycles=50)
    obs, _ = env.reset(seed=0)
    assert set(obs["drones_0"]) == {"state", "image"} and obs["drones_0"]["image"].shape == (16, 16, 3)
    assert env.observation_space("drones_1").contains(obs["drones_1"])
    assert isinstance(obs["plain_0"], np.ndarray)
    env.close()


def test_recorded_images(tmp_path):
    sim = _native.BatchSim(json.dumps(QuadHoverPad(image_size=24).scenario()), 1, 0, 1)
    assert sim.group_info(0)["image_layout"] == [("down/rgb", 0, 3)]
    path = tmp_path / "cam.mcap"
    sim.attach_recorder(0, str(path), camera_hz=50)
    for _ in range(10):
        sim.step(np.full((1, 1, 4), 0.5, np.float32))
    shown = sim.images(0)[0, 0].copy()
    assert sim.detach_recorder(0)
    frames = []
    with open(path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages(topics=["/agent/0/camera/down"]):
            assert schema.name == "foxglove.RawImage"
            frames.append(json.loads(message.data))
    # The reset's and one per step (the camera's 50 Hz); the last one is the observed one.
    assert len(frames) == 11, len(frames)
    last = frames[-1]
    assert (last["width"], last["height"], last["encoding"], last["step"]) == (24, 24, "rgb8", 72)
    rgb = np.frombuffer(base64.b64decode(last["data"]), np.uint8).reshape(24, 24, 3)
    assert np.array_equal(rgb, shown)


def test_image_only_observations():
    task = QuadHoverPad(obs=[SEMANTIC], image_size=16)
    sim = _native.BatchSim(json.dumps(task.scenario()), 2, 0, 1)
    assert sim.group_info(0)["obs_dim"] == 0 and sim.obs(0).shape == (2, 1, 0)
    sim.step(np.zeros((2, 1, 4), np.float32))
    assert (sim.images(0)[..., 0] == MARKER).any()


def test_drone_land_on_car_scripted_pilot():
    n = 8
    envs = gym.make_vec(
        "autonomousim/DroneLandOnCar-v0",
        num_envs=n,
        num_threads=4,
        semantic=True,
        map=RURAL,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    task = envs.unwrapped.task
    assert envs.single_observation_space["image"] == gym.spaces.Box(0, 255, (64, 64, 5), np.uint8)
    assert [t[0] for t in envs.unwrapped.obs_layout] == ["rot6d", "lin_vel_body", "ang_vel_body", "agl", "last_action"]
    assert envs.single_observation_space["state"].shape == (17,) and envs.single_action_space.shape == (4,)
    obs, _ = envs.reset(seed=0)
    # Every drone starts 15–30 m up within 10 m (each axis) of its car, which is in its view
    # (and visible, unless trees along the road hide it).
    state = envs.unwrapped.state
    d, h = task.relative(state)
    assert (d < 10.0 * np.sqrt(2.0)).all() and (h > 10.0).all() and (h < 30.0).all()
    assert task.in_view(state).all()
    assert (obs["image"][..., 4] == VEHICLE).reshape(n, -1).any(axis=1).sum() >= n // 2
    ret, done, success = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
    while not done.all():
        obs, reward, terminated, truncated, info = envs.step(task.scripted(obs))
        live = ~done
        ret[live] += reward[live]
        success |= live & task.success
        done |= terminated | truncated
    # From the image alone, the pilot lands on the moving car's roof in some of the worlds
    # (about two thirds on the training maps): at rest on it, earning the bonus; failures
    # cost the terminal penalty.
    assert success.sum() >= 2, success
    assert (state[success, STATE["support"]][:, 0] == task.car_index).all()
    assert (ret[success] > task.landing_bonus / 2).all() and (ret[~success] < 0.0).all(), ret
    envs.close()

    envs = gym.make_vec("autonomousim/DroneLandOnCar-v0", num_envs=1, map=RURAL, image_size=16)
    obs, _ = envs.reset(seed=0)
    with pytest.raises(ValueError, match="semantic"):
        envs.unwrapped.task.scripted(obs)
    envs.close()


def test_drone_land_on_car_rewards_and_ends():
    task = make_task("land_on_car", map=RURAL)
    task.bind(4, 0.04, 4)
    car = np.zeros((4, autonomousim.STATE_DIM))
    car[:, STATE["orientation"]] = [0.0, 0.0, 0.0, 1.0]
    task.car, task.car_index = car, 0
    state = np.zeros((4, autonomousim.STATE_DIM))
    state[:, STATE["orientation"]] = [0.0, 0.0, 0.0, 1.0]
    state[:, STATE["support"]] = -1
    state[:, STATE["position"]] = [[5.0, 0.0, 10.0], [0.0, 0.0, 3.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]
    task.reset(None, state)
    # Closer in; lower down; on the roof, at rest on the car; landed on something else.
    state[:, STATE["position"]] = [[4.0, 0.0, 10.0], [0.0, 0.0, 2.0], [0.0, 0.0, ROOF], [3.0, 0.0, 0.3]]
    state[2:, STATE["support"]] = [[0], [-1]]
    landed = int(autonomousim.Event.LANDED)
    events = np.array([0, 0, landed, landed], np.uint32)
    action = np.zeros((4, 4), np.float32)
    r = task.reward(state, action, action, events)
    h = 10.0 - ROOF
    progress = [np.hypot(5.0, 2 * h) - np.hypot(4.0, 2 * h), 2.0, 2 * (2.0 - ROOF), 2 * (2.0 - ROOF) - 3.0]
    np.testing.assert_allclose(r, 0.1 * np.array(progress) - 0.01 + [0.0, 0.0, 20.0, 0.0])
    assert task.succeeded(state, events).tolist() == [False, False, True, False]
    task.events = events
    assert task.failed(state).tolist() == [False, False, False, True]
    # Losing sight of the car (its centre out of the 90° field of view) for 5 s fails.
    state[:, STATE["position"]] = [[11.0, 0.0, 10.0], [9.0, 0.0, 10.0], [0.0, 0.0, 50.0], [0.0, 0.0, 30.0]]
    assert task.in_view(state).tolist() == [False, True, False, True]
    task.events = np.zeros(4, np.uint32)
    for _ in range(125):
        assert not task.failed(state).any()
    assert task.failed(state).tolist() == [True, False, True, False]

