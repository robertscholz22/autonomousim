"""Live streaming of training environments (``stream=``) to ``autonomousim-viewer attach``."""

import json
import socket

from mcap.exceptions import EndOfFile
from mcap.records import Channel, Message
from mcap.stream_reader import StreamReader

from autonomousim.vector_env import AutonomousimVectorEnv
from autonomousim.pettingzoo import AutonomousimParallelEnv


def messages(addr: str, sock_timeout: float = 20.0):
    """(topic, payload) of the stream at ``addr`` until it ends (a stream has no footer)."""
    host, port = addr.rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=sock_timeout) as s:
        topics = {}
        try:
            for record in StreamReader(s.makefile("rb"), skip_magic=False).records:
                if isinstance(record, Channel):
                    topics[record.id] = record.topic
                elif isinstance(record, Message):
                    yield topics[record.channel_id], record.data
        except EndOfFile:
            return


def test_a_vector_env_streams_world_0():
    env = AutonomousimVectorEnv(4, "hover", stream="127.0.0.1:0", seed=3)
    assert env.stream_addr.startswith("127.0.0.1:") and not env.stream_addr.endswith(":0")
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
    # A viewer attaching mid-run gets the recording's meta, the running episode, then states.
    stream = messages(env.stream_addr)
    topic, meta = next(stream)
    assert topic == "/meta" and json.loads(meta)["format"] == 1
    topic, episode = next(stream)
    assert topic == "/episode"
    for _ in range(40):
        env.step(env.action_space.sample())
    env.close()  # ends the stream
    states = [json.loads(d) for t, d in stream if t == "/agent/0/state"]
    # The policy rate: one state per step after the viewer came in.
    assert 39 <= len(states) <= 41, len(states)
    times = [s["time"] for s in states]
    # (Random actions may end an episode: the next one starts over.)
    assert all(abs(b - a - env.sim.policy_dt) < 1e-9 or b < a for a, b in zip(times, times[1:]))


def test_a_parallel_env_streams_its_world():
    env = AutonomousimParallelEnv("swarm_hover", count=3, stream="127.0.0.1:0")
    env.reset(seed=1)
    stream = messages(env.stream_addr)
    assert next(stream)[0] == "/meta"
    for _ in range(5):
        env.step({a: env.action_space(a).sample() for a in env.agents})
    env.close()
    agents = {t.split("/")[2] for t, _ in stream if t.endswith("/state")}
    assert agents == {"0", "1", "2"}
