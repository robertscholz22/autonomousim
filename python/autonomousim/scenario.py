"""Scenario helpers: loading files, merging overrides and slicing state rows.

Scenarios are plain dicts in the layout of the Rust ``Scenario`` (see ``default_scenario()``
and ``docs/PLAN.md``); ``BatchSim`` takes them as JSON.
"""

import copy
import json
import pathlib
from typing import Any

import numpy as np

from autonomousim._native import STATE_FIELDS, default_scenario as _default_json, normalize_scenario

#: Slices of the state row fields (``BatchSim.state``): position, orientation (x, y, z, w),
#: velocity (world), rates (body), goal, goal_yaw, agl, goal_index (the number of goals once
#: the last one is reached) and clearance (distance to the nearest terrain or solid obstacle,
#: up to 20 m).
STATE: dict[str, slice] = {}
_offset = 0
for _name, _len in STATE_FIELDS:
    STATE[_name] = slice(_offset, _offset + _len)
    _offset += _len
del _offset, _name, _len


def default_scenario() -> dict[str, Any]:
    """The default scenario with every field."""
    return json.loads(_default_json())


def load_scenario(path: str | pathlib.Path) -> dict[str, Any]:
    """Load a TOML or JSON scenario file as a dict (validated, defaults filled in)."""
    path = pathlib.Path(path)
    text = path.read_text()
    return json.loads(normalize_scenario(text, toml=path.suffix != ".json"))


def normalize(scenario: dict[str, Any]) -> dict[str, Any]:
    """Validate a scenario dict and fill in every default (maps are not built)."""
    return json.loads(normalize_scenario(json.dumps(scenario)))


def deep_merge(base: Any, override: Any) -> Any:
    """``base`` updated with ``override``: dicts merge recursively, lists merge element-wise
    (so ``{"groups": [{"count": 4}]}`` changes the first group), anything else is replaced."""
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = deep_merge(base[k], v) if k in base else copy.deepcopy(v)
        return out
    if isinstance(base, list) and isinstance(override, list):
        out = [deep_merge(b, o) for b, o in zip(base, override)]
        out.extend(copy.deepcopy(base[len(override) :]))
        out.extend(copy.deepcopy(override[len(base) :]))
        return out
    return copy.deepcopy(override)


def quat_up_z(q: np.ndarray) -> np.ndarray:
    """World z component of the body z axis for quaternions ``[..., (x, y, z, w)]``
    (1 upright, −1 inverted)."""
    return 1.0 - 2.0 * (q[..., 0] ** 2 + q[..., 1] ** 2)
