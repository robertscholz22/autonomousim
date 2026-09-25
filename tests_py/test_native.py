"""The native BatchSim: layout, in-place outputs, actions, resets, seeding, recording."""

import json
import pathlib

import numpy as np
import pytest
from mcap.reader import make_reader

import autonomousim
from autonomousim import STATE, STATE_DIM, TERMINAL, BatchSim, Event
from autonomousim._native import normalize_scenario
from autonomousim.scenario import deep_merge, default_scenario, load_scenario

ROOT = pathlib.Path(__file__).resolve().parent.parent

TWO_GROUPS = {
    "name": "two",
    "map": {"type": "testworld", "kind": "flat", "size": 100.0},
    "groups": [
        {"name": "small", "count": 2, "vehicle": "cf2x", "action_mode": "ctbr", "spawn": {"agl": [2.0, 3.0]}},
        {
            "name": "big",
            "vehicle": "iris_like",
            "action_mode": "velocity",
            "spawn": {"agl": [2.0, 3.0]},
            "obs": [{"term": "position"}, {"term": "quat"}],
        },
    ],
}


def make(num_envs=3, scenario=TWO_GROUPS, **kw):
    return BatchSim(json.dumps(scenario), num_envs, **kw)


def zeros(sim):
    return [np.zeros((sim.num_envs, i["count"], i["act_dim"]), np.float32) for i in map(sim.group_info, range(2))]


def test_layout_and_group_info():
    sim = make()
    assert (sim.num_envs, sim.num_groups, sim.group_names) == (3, 2, ["small", "big"])
    assert sim.decimation == 10 and sim.dt == pytest.approx(0.002) and sim.policy_dt == pytest.approx(0.02)
    small, big = sim.group_info("small"), sim.group_info(1)
    assert (small["count"], small["obs_dim"], small["act_dim"], small["action_mode"]) == (2, 19, 4, "ctbr")
    assert (big["vehicle"], big["obs_dim"], big["action_mode"], big["mass"]) == ("iris_like", 7, "velocity", 1.5)
    assert big["obs_layout"] == [("position", 0, 3), ("quat", 3, 4)]
    assert sim.obs(0).shape == (3, 2, 19) and sim.obs(0).dtype == np.float32
    assert sim.state("big").shape == (3, 1, STATE_DIM) and sim.state(1).dtype == np.float64
    assert sim.events(0).shape == (3, 2) and sim.events(0).dtype == np.uint32
    assert len(sim.map_hashes) == 1 and len(sim.map_hashes[0]) == 64
    assert json.loads(sim.scenario_json)["groups"][1]["vehicle"] == "iris_like"
    with pytest.raises(KeyError):
        sim.obs("nope")
    with pytest.raises(IndexError):
        sim.state(2)
    with pytest.raises(IndexError):
        sim.time(3)
    # The position observation and the state row agree.
    np.testing.assert_allclose(sim.obs(1)[:, 0, :3], sim.state(1)[:, 0, STATE["position"]], rtol=1e-6)


def test_outputs_are_overwritten_in_place():
    sim = make()
    obs, state, events = sim.obs(0), sim.state(0), sim.events(0)
    before = state.copy()
    actions = zeros(sim)
    actions[0][..., 3] = 1.0  # ctbr (roll, pitch, yaw rate, thrust): full thrust, the drones climb
    for _ in range(10):
        sim.step(actions)
    assert sim.obs(0) is obs
    assert np.shares_memory(sim.state(0), state) and np.shares_memory(sim.events(0), events)
    assert (state[..., STATE["position"]][..., 2] > before[..., STATE["position"]][..., 2]).all()
    assert sim.time(0) == pytest.approx(0.2)


def test_actions_are_validated():
    sim = make()
    a = zeros(sim)
    sim.step([a[0].astype(np.float64), a[1].reshape(3, 4)])  # float64 and any shape with num_envs rows
    sim.step(tuple(a))
    with pytest.raises(ValueError, match="one per group"):
        sim.step([a[0]])
    with pytest.raises(TypeError, match="per group"):
        sim.step(a[0])
    with pytest.raises(ValueError, match="expected 24 values"):
        sim.step([a[0][:, :1], a[1]])
    with pytest.raises(ValueError, match="num_envs"):
        sim.step([a[0].reshape(2, 12), a[1]])
    with pytest.raises(TypeError, match="float32 or float64"):
        sim.step([a[0].astype(np.int32), a[1]])
    one = BatchSim(json.dumps({"groups": [{}]}), 2)
    one.step(np.zeros((2, 4), np.float32))  # a single array for a single group
    # Out-of-range and non-finite actions are clipped and zeroed.
    one.step(np.array([[np.nan, 5.0, -5.0, np.inf]] * 2, np.float32))
    assert np.isfinite(one.state(0)).all()


def test_reset_masks_and_seeds():
    sim = make(num_envs=4, num_threads=2)
    sim.step(zeros(sim))
    before = sim.state(0).copy()
    sim.reset(np.array([True, False, True, False]))
    after = sim.state(0)
    assert not np.array_equal(after[0], before[0]) and np.array_equal(after[1], before[1])
    assert sim.time(0) == 0.0 and sim.time(1) == pytest.approx(0.02)
    # One seed for every world: identical episodes.
    sim.reset(seeds=np.full(4, 7, np.uint64))
    assert all(np.array_equal(sim.state(0)[i], sim.state(0)[0]) for i in range(4))
    first = sim.state_hash(0)
    for _ in range(5):
        sim.step(zeros(sim))
    assert len({sim.state_hash(i) for i in range(4)}) == 1
    sim.reset(mask=[True, True, False, False], seeds=[7, 7, 0, 0])  # lists work too
    assert sim.state_hash(0) == first
    with pytest.raises(ValueError, match="one value per world"):
        sim.reset(np.ones(3, bool))
    with pytest.raises(ValueError, match="non-negative"):
        sim.reset(seeds=np.array([-1, 0, 0, 0]))


def test_world_does_not_depend_on_batch_size_or_threads():
    rng = np.random.default_rng(0)
    steps = [[rng.uniform(-1, 1, (5, n, 4)).astype(np.float32) for n in (2, 1)] for _ in range(30)]

    def run(num_envs, threads):
        sim = make(num_envs=num_envs, seed=11, num_threads=threads)
        for a in steps:
            sim.step([x[:num_envs] for x in a])
        return sim.state_hash(0)

    assert run(1, 1) == run(5, 3)


def test_invalid_scenarios_raise_value_error():
    with pytest.raises(ValueError, match="unknown field"):
        BatchSim(json.dumps({"groups": [{"colour": "red"}]}), 1)
    with pytest.raises(ValueError, match="policy"):
        BatchSim(json.dumps({"policy_hz": 70}), 1)
    with pytest.raises(ValueError):
        BatchSim(json.dumps({"groups": [{"vehicle": "no_such_drone"}]}), 1)
    with pytest.raises(ValueError, match="at least one world"):
        BatchSim(json.dumps({}), 0)


def test_scenario_helpers(tmp_path):
    d = default_scenario()
    # physics_hz 0: chosen by the vehicles when compiled (500 Hz for drones, 1 kHz with ground vehicles).
    assert d["physics_hz"] == 0 and d["groups"][0]["vehicle"] == "cf2x"
    full = json.loads(normalize_scenario('policy_hz = 100\n[[groups]]\nvehicle = "iris_like"\n', toml=True))
    assert full["policy_hz"] == 100 and full["groups"][0]["spawn"]["clearance"] == 1.0
    path = tmp_path / "s.toml"
    path.write_text('name = "x"\nmap = { type = "testworld", kind = "single_tree" }\n')
    assert load_scenario(path)["map"] == {"type": "testworld", "kind": "single_tree"}
    merged = deep_merge(TWO_GROUPS, {"groups": [{"count": 5}], "policy_hz": 25})
    assert merged["groups"][0]["count"] == 5 and merged["groups"][0]["vehicle"] == "cf2x"
    assert merged["groups"][1] == TWO_GROUPS["groups"][1] and merged["policy_hz"] == 25
    assert TWO_GROUPS["groups"][0]["count"] == 2


@pytest.mark.parametrize("path", sorted(ROOT.glob("assets/scenarios/*.toml")), ids=lambda p: p.name)
def test_example_scenarios_build(path, tmp_path, monkeypatch):
    monkeypatch.setenv("AUTONOMOUSIM_MAP_CACHE", str(tmp_path))
    sim = BatchSim(json.dumps(load_scenario(path)), 2, num_threads=1)
    sim.step([np.zeros((2, i["count"], i["act_dim"]), np.float32) for i in map(sim.group_info, range(sim.num_groups))])
    assert all(np.isfinite(sim.obs(g)).all() for g in range(sim.num_groups))


def test_events():
    assert Event.CRASH_TERRAIN == 1 and Event.DISABLED == 1 << 9 and Event.FINISHED == 1 << 11
    assert Event.ROLLOVER == 1 << 12 and Event.STUCK == 1 << 13 and Event.JACKKNIFE == 1 << 14
    assert TERMINAL == Event.CRASH_TERRAIN | Event.CRASH_OBSTACLE | Event.CRASH_AGENT | Event.WATER | (
        Event.OUT_OF_BOUNDS | Event.NAN | Event.ROLLOVER | Event.JACKKNIFE
    )
    assert autonomousim.events.names(int(Event.WATER | Event.LANDED)) == ["water", "landed"]
    # Falling from 2–3 m with the rotors off is a crash.
    sim = make(num_envs=2)
    a = zeros(sim)
    a[0][..., 3] = -1.0
    seen = np.zeros((2, 2), np.uint32)
    for _ in range(60):
        sim.step(a)
        seen |= sim.events(0)
    assert autonomousim.events.is_terminal(seen).all()


def test_recording(tmp_path):
    sim = make(num_envs=2)
    path = tmp_path / "run.mcap"
    sim.attach_recorder(1, str(path))
    for _ in range(10):
        sim.step(zeros(sim))
    sim.reset(np.array([False, True]))
    sim.step(zeros(sim))
    assert sim.detach_recorder(1) and not sim.detach_recorder(1)
    topics = {}
    with open(path, "rb") as f:
        for _schema, channel, message in make_reader(f).iter_messages():
            topics.setdefault(channel.topic, []).append(json.loads(message.data))
    assert len(topics["/meta"]) == 1 and len(topics["/episode"]) == 2
    # One state at the start of each episode and one per policy step (50 Hz).
    assert len(topics["/agent/0/state"]) == 1 + 10 + 1 + 1 and "/agent/2/pose" in topics
    assert topics["/meta"][0]["scenario"]["name"] == "two"
    # Closing (or dropping) finishes open recordings.
    sim.attach_recorder(0, tmp_path / "b.mcap")
    sim.step(zeros(sim))
    sim.close()
    with open(tmp_path / "b.mcap", "rb") as f:
        assert sum(1 for _ in make_reader(f).iter_messages()) > 0


def test_trailers():
    assert {"semitrailer_3axle", "farm_trailer"} <= set(autonomousim.trailer_presets())
    rig = {
        "name": "rig",
        "map": {"type": "testworld", "kind": "flat", "size": 300.0},
        "groups": [
            {
                "vehicle": "truck_6x4",
                "trailers": ["semitrailer_3axle"],
                "action_mode": "vk",
                "obs": [{"term": "articulation"}, {"term": "trailer_goal"}],
            }
        ],
    }
    sim = BatchSim(json.dumps(rig), 1, num_threads=1)
    assert sim.group_info(0)["obs_dim"] == 8
    for _ in range(100):
        sim.step([np.array([[[0.3, 0.8]]], np.float32)])
    state = sim.state(0)[0, 0]
    assert state[STATE["articulation"]][0] < -0.01
    tail = state[STATE["tail"]]
    assert np.linalg.norm(tail[:2] - state[STATE["position"]][:2]) > 8.0
    bad = deep_merge(rig, {"groups": [{"vehicle": "cf2x"}]})
    with pytest.raises(ValueError, match="tow"):
        BatchSim(json.dumps(bad), 1)
