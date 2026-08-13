"""Deterministic request identity for crash-safe ensemble continuation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum

import numpy as np

from rate_of_closure.simulation import SimulationConfig
from shared.python.contracts import require

from .simulation_types import SimulationEnsembleRequest

REQUEST_IDENTITY_SCHEMA = "rate-of-closure/ensemble-request-identity@1"
CONFIG_IDENTITY_SCHEMA = "rate-of-closure/simulation-config-identity@1"


def _qualified_type(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _canonical_value(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "$type": _qualified_type(value),
            "$fields": [
                [field.name, _canonical_value(getattr(value, field.name))]
                for field in dataclasses.fields(value)
            ],
        }
    if isinstance(value, Enum):
        return {
            "$enum": _qualified_type(value),
            "$value": _canonical_value(value.value),
        }
    if isinstance(value, Mapping):
        require(
            all(isinstance(key, str) for key in value),
            "request mapping keys must be strings",
        )
        return {
            "$mapping": [[key, _canonical_value(value[key])] for key in sorted(value)]
        }
    if isinstance(value, tuple):
        return {"$tuple": [_canonical_value(item) for item in value]}
    if isinstance(value, list):
        return {"$list": [_canonical_value(item) for item in value]}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        require(math.isfinite(number), "request identity numbers must be finite")
        return number
    require(False, "unsupported request identity value", _qualified_type(value))
    raise AssertionError("unreachable")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def config_identity_sha256(config: SimulationConfig) -> str:
    """Hash every field of one normalized simulation configuration."""
    require(isinstance(config, SimulationConfig), "invalid simulation config")
    digest = hashlib.sha256()
    digest.update(CONFIG_IDENTITY_SCHEMA.encode("ascii"))
    digest.update(_canonical_json(_canonical_value(config)))
    return digest.hexdigest()


def request_identity_sha256(request: SimulationEnsembleRequest) -> str:
    """Hash every plan, input, and per-trial configuration field in order."""
    require(isinstance(request, SimulationEnsembleRequest), "invalid request")
    digest = hashlib.sha256()
    digest.update(REQUEST_IDENTITY_SCHEMA.encode("ascii"))
    digest.update(_canonical_json(request.plan.to_json_dict()))
    inputs = np.asarray(request.sampled_inputs, dtype="<f8", order="C")
    digest.update(_canonical_json({"shape": list(inputs.shape), "dtype": "<f8"}))
    digest.update(inputs.tobytes(order="C"))
    for index, config in enumerate(request.configs):
        encoded = _canonical_json({"index": index, "config": _canonical_value(config)})
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "CONFIG_IDENTITY_SCHEMA",
    "REQUEST_IDENTITY_SCHEMA",
    "config_identity_sha256",
    "request_identity_sha256",
]
