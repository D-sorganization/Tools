"""Canonical content identity for immutable simulation configurations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from shared.python.contracts import require

_IDENTITY_DOMAIN = b"rate-simulation-configurations/v1\0"


def simulation_configurations_sha256(configs: Sequence[object]) -> str:
    """Hash every configuration field in canonical trial order."""
    require(len(configs) > 0, "simulation configurations must be non-empty")
    return simulation_configuration_stream_sha256(configs, count=len(configs))


def simulation_configuration_stream_sha256(
    configs: Iterable[object], *, count: int
) -> str:
    """Hash an exact bounded configuration stream without retaining its roster."""
    require(type(count) is int and count > 0, "configuration count must be positive")
    digest = hashlib.sha256()
    digest.update(_IDENTITY_DOMAIN)
    digest.update(count.to_bytes(8, "big"))
    observed = 0
    for config in configs:
        payload = json.dumps(
            _canonical(config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        observed += 1
    require(observed == count, "configuration stream count does not match", observed)
    return digest.hexdigest()


def _canonical(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        require(math.isfinite(value), "configuration floats must be finite")
        return value
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "array_dtype": array.dtype.str,
            "array_shape": list(array.shape),
            "array_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        require(
            all(isinstance(key, str) for key in value),
            "configuration mapping keys must be strings",
        )
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        return sorted(items, key=_sort_key)
    require(False, "unsupported simulation configuration value", type(value).__name__)
    raise AssertionError("unreachable")


def _sort_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "simulation_configuration_stream_sha256",
    "simulation_configurations_sha256",
]
