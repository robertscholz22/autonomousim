"""Camera observations: ``Dict`` spaces, in-place image buffers, vector semantics, seeding and
the multi-agent and PettingZoo wrappers (rendered on lavapipe, see ``conftest.py``)."""

import json

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.vector import AutoresetMode
from pettingzoo.test import parallel_api_test

import autonomousim
from autonomousim import SEMANTIC_CLASSES, _native
from autonomousim.multiagent import MultiAgentVectorEnv
from autonomousim.pettingzoo import parallel_env
from autonomousim.tasks import QuadHover, QuadHoverPad
from autonomousim.tasks.hover_pad import DEFAULT_OBS
from autonomousim.tasks.multi import MultiAgentTask, Team
from autonomousim.vector_env import AutonomousimVectorEnv

SEMANTIC = {"term": "camera", "sensor": "down", "output": "semantic"}
DEPTH = {"term": "camera", "sensor": "down", "output": "depth", "range": 10.0}
MARKER = SEMANTIC_CLASSES.index("marker")


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
