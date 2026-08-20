"""Hash-verified expected-strokes baselines and source-backed SG bookkeeping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

CONTRACT_VERSION = "launch-monitor-strokes-gained-baseline/1.0.0"
MAX_BASELINE_BYTES = 10 * 1024 * 1024
YARDS_PER_METRE = 1.0936132983377078


@dataclass(frozen=True)
class BaselineState:
    """One expected-strokes point for a declared course lie and distance."""

    lie: str
    distance_yards: float
    expected_strokes: float


@dataclass(frozen=True)
class StrokesGainedBaseline:
    """Versioned provenance plus an immutable expected-strokes table."""

    baseline_id: str
    version: str
    source_url: str
    license: str
    table_sha256: str
    states: tuple[BaselineState, ...]


@dataclass(frozen=True)
class SourceBackedStrokesGainedRequest:
    """Map retained course-state columns to one verified baseline."""

    before_lie_column: str
    before_distance_column: str
    after_lie_column: str
    after_distance_column: str
    before_distance_unit: str
    after_distance_unit: str


@dataclass(frozen=True)
class StrokesGainedBackingRow:
    """Reproducible state lookup and SG result for one complete shot."""

    source_index: int
    before_lie: str
    before_distance_yards: float
    after_lie: str
    after_distance_yards: float
    expected_before: float
    expected_after: float
    strokes_gained: float


@dataclass(frozen=True)
class SourceBackedStrokesGainedResult:
    """Traceable source-backed SG values and baseline identity."""

    metric_name: str
    unit: str
    values: tuple[float, ...]
    mean: float
    baseline_id: str
    baseline_version: str
    source_url: str
    license: str
    table_sha256: str
    backing_rows: tuple[StrokesGainedBackingRow, ...]
    formula: str


def baseline_table_hash(states: list[dict[str, object]]) -> str:
    """Return the canonical SHA-256 used by baseline artifacts."""

    payload = json.dumps(
        states, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


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
        "distance_yards",
        "expected_strokes",
    }:
        raise ValueError(
            "each baseline state requires lie, distance_yards, expected_strokes"
        )
    lie = _required_text(value, "lie").lower()
    distance = value["distance_yards"]
    expected = value["expected_strokes"]
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        raise ValueError("baseline distance_yards must be numeric")
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        raise ValueError("baseline expected_strokes must be numeric")
    if not np.isfinite(distance) or float(distance) < 0:
        raise ValueError("baseline distance_yards must be finite and nonnegative")
    if not np.isfinite(expected) or float(expected) < 0:
        raise ValueError("baseline expected_strokes must be finite and nonnegative")
    return BaselineState(lie, float(distance), float(expected))


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
    identities = {(state.lie, state.distance_yards) for state in states}
    if len(identities) != len(states):
        raise ValueError("baseline contains duplicate lie/distance states")
    return StrokesGainedBaseline(
        _required_text(payload, "baseline_id"),
        _required_text(payload, "version"),
        _valid_source_url(_required_text(payload, "source_url")),
        _required_text(payload, "license"),
        declared_hash,
        states,
    )


def _yards(value: object, unit: str) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not np.isfinite(numeric):
        return None
    if unit == "yd":
        return float(numeric)
    if unit == "m":
        return float(numeric) * YARDS_PER_METRE
    raise ValueError("distance unit must be 'yd' or 'm'")


def _expected(baseline: StrokesGainedBaseline, lie: str, distance: float) -> float:
    candidates = sorted(
        (state for state in baseline.states if state.lie == lie),
        key=lambda state: state.distance_yards,
    )
    if (
        not candidates
        or distance < candidates[0].distance_yards
        or distance > candidates[-1].distance_yards
    ):
        raise ValueError(f"course state {lie}/{distance:g} yd is outside the baseline")
    return float(
        np.interp(
            distance,
            [state.distance_yards for state in candidates],
            [state.expected_strokes for state in candidates],
        )
    )


def _backing_rows(
    frame: pd.DataFrame,
    baseline: StrokesGainedBaseline,
    request: SourceBackedStrokesGainedRequest,
) -> tuple[StrokesGainedBackingRow, ...]:
    columns = (
        request.before_lie_column,
        request.before_distance_column,
        request.after_lie_column,
        request.after_distance_column,
    )
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"columns are unavailable: {missing}")
    rows: list[StrokesGainedBackingRow] = []
    for source_index, (_, row) in enumerate(frame.iterrows()):
        before_value = row[request.before_lie_column]
        after_value = row[request.after_lie_column]
        before_lie = "" if pd.isna(before_value) else str(before_value).strip().lower()
        after_lie = "" if pd.isna(after_value) else str(after_value).strip().lower()
        before_distance = _yards(
            row[request.before_distance_column], request.before_distance_unit
        )
        after_distance = _yards(
            row[request.after_distance_column], request.after_distance_unit
        )
        if (
            not before_lie
            or not after_lie
            or before_distance is None
            or after_distance is None
        ):
            continue
        expected_before = _expected(baseline, before_lie, before_distance)
        expected_after = _expected(baseline, after_lie, after_distance)
        gained = expected_before - 1.0 - expected_after
        rows.append(
            StrokesGainedBackingRow(
                source_index,
                before_lie,
                before_distance,
                after_lie,
                after_distance,
                expected_before,
                expected_after,
                gained,
            )
        )
    return tuple(rows)


def calculate_source_backed_strokes_gained(
    frame: pd.DataFrame,
    baseline: StrokesGainedBaseline,
    request: SourceBackedStrokesGainedRequest,
) -> SourceBackedStrokesGainedResult:
    """Calculate SG from verified expected-strokes course-state lookups."""

    rows = _backing_rows(frame, baseline, request)
    if not rows:
        raise ValueError(
            "source-backed strokes gained requires complete course-state rows"
        )
    values = tuple(row.strokes_gained for row in rows)
    return SourceBackedStrokesGainedResult(
        "source_backed_strokes_gained",
        "strokes",
        values,
        float(np.mean(values)),
        baseline.baseline_id,
        baseline.version,
        baseline.source_url,
        baseline.license,
        baseline.table_sha256,
        rows,
        "SG = verified E(before course state) - 1 - verified E(after course "
        "state); linear interpolation occurs only within the same lie.",
    )


__all__ = [
    "CONTRACT_VERSION",
    "SourceBackedStrokesGainedRequest",
    "SourceBackedStrokesGainedResult",
    "StrokesGainedBaseline",
    "baseline_table_hash",
    "calculate_source_backed_strokes_gained",
    "load_strokes_gained_baseline",
]
