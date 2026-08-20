"""Lossless exports for complete swing-variation ensembles."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from rate_of_closure.variation.simulation_types import SimulationEnsembleResult
from shared.python.contracts import ContractViolationError, require
from shared.python.swing_sim.variation.dataset_io import to_json_dict as dataset_json

from ._ensemble_json_contract import validate_decoded_tree
from ._ensemble_limits import MAX_ENSEMBLE_JSON_BYTES
from ._ensemble_parser import parse_ensemble_document

ENSEMBLE_EXPORT_SCHEMA_VERSION = 1


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Construct one JSON object while rejecting duplicate field names."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field", key)
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    """Reject non-standard JSON NaN and infinity spellings."""
    require(False, "JSON numbers must be finite", value)


def loads(text: str) -> SimulationEnsembleResult:
    """Parse a bounded, duplicate-safe complete ensemble JSON document."""
    require(isinstance(text, str), "ensemble JSON must be text")
    try:
        encoded = text.encode("utf-8")
        require(
            len(encoded) <= MAX_ENSEMBLE_JSON_BYTES,
            "ensemble JSON byte limit exceeded",
        )
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_constant,
        )
    except ContractViolationError:
        raise
    except (UnicodeError, RecursionError, ValueError, OverflowError) as exc:
        require(False, "ensemble document must be valid JSON", str(exc))
        raise AssertionError from exc
    return from_json_dict(document)


def from_json_dict(data: object) -> SimulationEnsembleResult:
    """Build an immutable typed ensemble from the exact v1 writer schema.

    Version 1 is the only accepted outer schema and has no implicit migration.
    Future schemas must add an explicit migration before this reader accepts
    them; unknown, partial, or extra fields fail closed.
    """
    return parse_ensemble_document(data, ENSEMBLE_EXPORT_SCHEMA_VERSION)


def read_json(path: str | Path) -> SimulationEnsembleResult:
    """Read a size-bounded ensemble written by :func:`write_json`."""
    source = Path(path)
    require(
        source.stat().st_size <= MAX_ENSEMBLE_JSON_BYTES,
        "ensemble JSON byte limit exceeded",
    )
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeError as exc:
        require(False, "ensemble document must be valid UTF-8", str(exc))
        raise AssertionError from exc
    return loads(text)


def to_json_dict(result: SimulationEnsembleResult) -> dict[str, Any]:
    """Return a documented JSON-safe document without inventing missing values."""
    traces = result.traces
    positions: list[list[list[list[float | None]]]] = []
    for trial_index in range(traces.n_trials):
        trial: list[list[list[float | None]]] = []
        for sample_index in range(traces.sample_times_s.size):
            if not traces.sample_valid[trial_index, sample_index]:
                trial.append([[None, None, None] for _ in traces.point_ids])
            else:
                trial.append(
                    [
                        [float(value) for value in point]
                        for point in traces.positions_m[trial_index, sample_index]
                    ]
                )
        positions.append(trial)
    document = {
        "schema_version": ENSEMBLE_EXPORT_SCHEMA_VERSION,
        "coordinate_frame": traces.coordinate_frame,
        "position_unit": "m",
        "time_unit": "s",
        "point_ids": list(traces.point_ids),
        "sample_times_s": traces.sample_times_s.tolist(),
        "sample_valid": traces.sample_valid.tolist(),
        "impact_sample_indices": traces.impact_sample_indices.tolist(),
        "positions_m": positions,
        "outcomes": [
            {
                "trial_index": outcome.trial_index,
                "status": outcome.status.value,
                "values": dict(outcome.values),
                "failure_type": outcome.failure_type,
                "failure_message": outcome.failure_message,
            }
            for outcome in result.outcomes
        ],
        "variation": dataset_json(result.variation),
    }
    validate_decoded_tree(document)
    return document


def _encoded_document(result: SimulationEnsembleResult, indent: int) -> str:
    """Encode one writer document under the reader's finite byte contract."""
    document = to_json_dict(result)
    try:
        text = json.dumps(document, indent=indent, allow_nan=False)
        encoded = text.encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        require(False, "ensemble document must contain strict finite JSON", str(exc))
        raise AssertionError from exc
    require(
        len(encoded) <= MAX_ENSEMBLE_JSON_BYTES,
        "ensemble JSON byte limit exceeded",
    )
    return text


def write_json(result: SimulationEnsembleResult, path: str | Path) -> None:
    """Write the complete typed outcomes and common-grid position traces."""
    text = _encoded_document(result, indent=2)
    Path(path).write_text(text, encoding="utf-8")


def write_trace_csv(result: SimulationEnsembleResult, path: str | Path) -> None:
    """Write one long-form row per trial, sample, and modeled point."""
    traces = result.traces
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trial",
                "status",
                "sample",
                "time_s",
                "point_id",
                "x_target_m",
                "y_up_m",
                "z_right_m",
                "sample_valid",
                "is_impact_sample",
                "coordinate_frame",
            ]
        )
        for trial_index, outcome in enumerate(result.outcomes):
            impact_index = int(traces.impact_sample_indices[trial_index])
            for sample_index, time_s in enumerate(traces.sample_times_s):
                valid = bool(traces.sample_valid[trial_index, sample_index])
                for point_index, point_id in enumerate(traces.point_ids):
                    xyz = traces.positions_m[trial_index, sample_index, point_index]
                    writer.writerow(
                        [
                            trial_index,
                            outcome.status.value,
                            sample_index,
                            repr(float(time_s)),
                            point_id,
                            *(
                                (
                                    repr(float(value))
                                    if valid and math.isfinite(value)
                                    else ""
                                )
                                for value in xyz
                            ),
                            int(valid),
                            int(sample_index == impact_index),
                            traces.coordinate_frame,
                        ]
                    )


__all__ = [
    "ENSEMBLE_EXPORT_SCHEMA_VERSION",
    "MAX_ENSEMBLE_JSON_BYTES",
    "from_json_dict",
    "loads",
    "read_json",
    "to_json_dict",
    "write_json",
    "write_trace_csv",
]
