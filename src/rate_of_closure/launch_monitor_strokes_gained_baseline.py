"""Validation authority for versioned expected-strokes baseline artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

CONTRACT_VERSION = "launch-monitor-strokes-gained-baseline/2.0.0"
MAX_BASELINE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class BaselineState:
    """One expected-strokes point for an explicit course state."""

    lie: str
    context: str
    target: str
    distance_yards: float
    expected_strokes: float
    standard_error: float | None


@dataclass(frozen=True)
class StrokesGainedBaseline:
    """Versioned provenance plus an immutable expected-strokes table."""

    baseline_id: str
    version: str
    source_url: str
    license: str
    table_sha256: str
    states: tuple[BaselineState, ...]


def baseline_table_hash(states: list[dict[str, object]]) -> str:
    """Return the canonical SHA-256 used by baseline artifacts."""

    canonical = [_canonical_state(state) for state in states]
    canonical.sort(
        key=lambda state: (
            str(state["lie"]),
            str(state["context"]),
            str(state["target"]),
            float(str(state["distance_yards"])),
        )
    )
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _canonical_number(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("baseline numbers must be numeric")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("baseline numbers must be finite")
    normalized = f"{numeric:.12f}".rstrip("0").rstrip(".")
    return "0" if normalized in {"", "-0"} else normalized


def _canonical_state(value: dict[str, object]) -> dict[str, object]:
    state = _state(value)
    return {
        "context": state.context,
        "distance_yards": _canonical_number(state.distance_yards),
        "expected_strokes": _canonical_number(state.expected_strokes),
        "lie": state.lie,
        "standard_error": (
            None
            if state.standard_error is None
            else _canonical_number(state.standard_error)
        ),
        "target": state.target,
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"baseline {key} must be non-empty text")
    return value.strip()


def _valid_source_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("baseline source_url must be HTTP(S)")
    return value


def _state(value: object) -> BaselineState:
    if not isinstance(value, dict) or set(value) != {
        "lie",
        "context",
        "target",
        "distance_yards",
        "expected_strokes",
        "standard_error",
    }:
        raise ValueError(
            "each baseline state requires lie, context, target, distance_yards, "
            "expected_strokes, and standard_error"
        )
    lie = _required_text(value, "lie").lower()
    context = _required_text(value, "context").lower()
    target = _required_text(value, "target").lower()
    distance = value["distance_yards"]
    expected = value["expected_strokes"]
    standard_error = value["standard_error"]
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        raise ValueError("baseline distance_yards must be numeric")
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        raise ValueError("baseline expected_strokes must be numeric")
    if not np.isfinite(distance) or float(distance) < 0:
        raise ValueError("baseline distance_yards must be finite and nonnegative")
    if not np.isfinite(expected) or float(expected) < 0:
        raise ValueError("baseline expected_strokes must be finite and nonnegative")
    if standard_error is not None and (
        isinstance(standard_error, bool)
        or not isinstance(standard_error, (int, float))
        or not np.isfinite(standard_error)
        or float(standard_error) < 0
    ):
        raise ValueError("baseline standard_error must be null or nonnegative")
    return BaselineState(
        lie,
        context,
        target,
        float(distance),
        float(expected),
        None if standard_error is None else float(standard_error),
    )


def load_strokes_gained_baseline(path: Path) -> StrokesGainedBaseline:
    """Load a bounded baseline artifact and verify schema and table digest."""

    if path.stat().st_size > MAX_BASELINE_BYTES:
        raise ValueError("strokes-gained baseline exceeds the 10 MiB limit")
    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )
    expected_keys = {
        "contract_version",
        "baseline_id",
        "version",
        "source_url",
        "license",
        "table_sha256",
        "states",
    }
    if isinstance(payload, dict) and set(payload) != expected_keys:
        raise ValueError("baseline artifact fields do not match the contract")
    if (
        not isinstance(payload, dict)
        or payload.get("contract_version") != CONTRACT_VERSION
    ):
        raise ValueError(f"baseline contract_version must be {CONTRACT_VERSION}")
    raw_states = payload.get("states")
    if not isinstance(raw_states, list) or len(raw_states) < 2:
        raise ValueError("baseline states must contain at least two rows")
    declared_hash = _required_text(payload, "table_sha256").lower()
    if len(declared_hash) != 64 or baseline_table_hash(raw_states) != declared_hash:
        raise ValueError("baseline table SHA-256 does not match states")
    states = tuple(_state(item) for item in raw_states)
    identities = {
        (state.lie, state.context, state.target, state.distance_yards)
        for state in states
    }
    if len(identities) != len(states):
        raise ValueError("baseline contains duplicate course states")
    return StrokesGainedBaseline(
        _required_text(payload, "baseline_id"),
        _required_text(payload, "version"),
        _valid_source_url(_required_text(payload, "source_url")),
        _required_text(payload, "license"),
        declared_hash,
        states,
    )


__all__ = [
    "CONTRACT_VERSION",
    "BaselineState",
    "StrokesGainedBaseline",
    "baseline_table_hash",
    "load_strokes_gained_baseline",
]
