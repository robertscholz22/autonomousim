"""Event bits reported per agent and policy step (``BatchSim.events``)."""

import enum

import numpy as np

from autonomousim._native import EVENTS, TERMINAL_EVENTS

Event = enum.IntFlag("Event", {name.upper(): bit for name, bit in EVENTS})
Event.__doc__ = (
    "Event bits: crashes, water, out of bounds, NaN (terminal), foliage, ground contact, landed, disabled,"
    " goal reached, finished (last goal reached)."
)

#: Bits that end an episode: crashes, water, out of bounds and NaN.
TERMINAL = Event(TERMINAL_EVENTS)


def names(bits: int) -> list[str]:
    """Names of the events set in ``bits``."""
    return [name for name, bit in EVENTS if bits & bit]


def is_terminal(events: np.ndarray) -> np.ndarray:
    """Element-wise: whether any terminal event is set."""
    return (events & TERMINAL_EVENTS) != 0
