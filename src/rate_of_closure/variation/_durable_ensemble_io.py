"""Bounded atomic wire helpers for durable Rate ensemble chunks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, cast
from zipfile import BadZipFile, ZipFile

import numpy as np

from shared.python.contracts import require
from shared.python.swing_sim.variation.execution_metadata import (
    make_execution_metadata,
)

from .ensemble_chunks import EnsembleStreamHeader, SimulationResultChunk
from .simulation_types import (
    ALL_OUTPUT_NAMES,
    SimulationTrialOutcome,
    TrialEvaluationStatus,
)

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
MAX_CHUNK_FILE_BYTES = 16_000_000
MAX_CHUNK_UNCOMPRESSED_BYTES = 32_000_000
_ARRAY_KEYS = frozenset(
    {
        "sampled_inputs",
        "positions_m",
        "sample_valid",
        "impact_sample_indices",
        "trial_indices",
        "statuses",
        "output_values",
        "failure_types",
        "failure_messages",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "header_sha256",
        "header",
        "next_index",
        "failed_count",
        "chunks",
        "elapsed_s",
    }
)
_CHUNK_RECORD_KEYS = frozenset(
    {"file", "start_index", "stop_index", "sha256", "failed_count"}
)


def header_document(header: EnsembleStreamHeader) -> dict[str, object]:
    """Describe immutable scientific identity without execution controls."""
    require(
        len(header.configuration_sha256) == 64,
        "durable headers require complete simulation configuration identity",
    )
    return {
        "plan": header.plan.to_json_dict(),
        "sampled_inputs": {
            "shape": [header.plan.n_runs, len(header.plan.noise)],
            "dtype": np.dtype(float).str,
            "sha256": header.sampled_input_sha256,
        },
        "sample_times_s": _array_identity(header.sample_times_s),
        "point_ids": list(header.point_ids),
        "coordinate_frame": header.coordinate_frame,
        "configuration_sha256": header.configuration_sha256,
        "execution_metadata": make_execution_metadata(header.plan).to_json_dict(),
    }


def json_sha256(value: object) -> str:
    """Hash canonical strict JSON bytes."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def new_manifest(header: dict[str, object], digest: str) -> dict[str, object]:
    """Return an empty in-progress manifest bound to one header."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "header_sha256": digest,
        "header": header,
        "next_index": 0,
        "failed_count": 0,
        "chunks": [],
        "elapsed_s": None,
    }


def write_json_atomic(path: Path, value: object) -> None:
    """Replace strict JSON only after a flushed temporary write."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(path: Path) -> dict[str, Any]:
    """Read the exact bounded durable-manifest schema."""
    require(path.stat().st_size <= 1_000_000, "durable manifest is too large")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        require(False, "durable manifest must be valid JSON", str(exc))
        raise AssertionError from exc
    require(type(value) is dict, "durable manifest must be an object")
    manifest = cast(dict[str, Any], value)
    require(set(manifest) == _MANIFEST_KEYS, "durable manifest fields are invalid")
    require(manifest["schema_version"] == SCHEMA_VERSION, "unsupported manifest schema")
    require(
        type(manifest["status"]) is str
        and manifest["status"] in {"in_progress", "complete"},
        "invalid manifest status",
    )
    require(type(manifest["chunks"]) is list, "manifest chunks must be a list")
    require(type(manifest["next_index"]) is int, "invalid manifest next_index")
    require(type(manifest["failed_count"]) is int, "invalid manifest failed_count")
    require(
        type(manifest["header_sha256"]) is str and len(manifest["header_sha256"]) == 64,
        "invalid manifest header checksum",
    )
    elapsed = manifest["elapsed_s"]
    if manifest["status"] == "in_progress":
        require(elapsed is None, "in-progress manifest cannot declare elapsed_s")
    else:
        require(
            type(elapsed) is float and math.isfinite(elapsed) and elapsed >= 0.0,
            "complete manifest requires finite elapsed_s",
        )
    return manifest


def verify_header(
    manifest: dict[str, Any], expected: dict[str, object], expected_sha256: str
) -> None:
    """Fail closed unless stored and requested header identities agree."""
    require(type(manifest["header"]) is dict, "manifest header must be an object")
    require(
        json_sha256(manifest["header"]) == manifest["header_sha256"],
        "manifest header checksum is invalid",
    )
    require(
        manifest["header_sha256"] == expected_sha256 and manifest["header"] == expected,
        "durable header identity does not match this request",
    )


def chunk_record(
    value: object, expected_start: int, trial_count: int
) -> dict[str, object]:
    """Validate one exact contiguous manifest chunk entry."""
    require(type(value) is dict, "chunk record must be an object")
    record = cast(dict[str, object], value)
    require(set(record) == _CHUNK_RECORD_KEYS, "chunk record fields are invalid")
    start_value = record["start_index"]
    stop_value = record["stop_index"]
    require(
        type(start_value) is int and start_value == expected_start,
        "chunk record is not contiguous",
    )
    start = cast(int, start_value)
    require(
        type(stop_value) is int and start < stop_value <= trial_count,
        "chunk record bounds are invalid",
    )
    stop = cast(int, stop_value)
    filename = record["file"]
    require(
        type(filename) is str
        and Path(filename).name == filename
        and filename == f"chunk-{start:08d}-{stop:08d}.npz",
        "chunk filename is invalid",
    )
    checksum = record["sha256"]
    require(type(checksum) is str and len(checksum) == 64, "chunk checksum is invalid")
    failures_value = record["failed_count"]
    require(
        type(failures_value) is int and 0 <= failures_value <= stop - start,
        "invalid chunk failures",
    )
    return record


def write_chunk_atomic(path: Path, chunk: SimulationResultChunk) -> None:
    """Replace one pickle-free NPZ only after a flushed bounded write."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    values = _outcome_values(chunk)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                sampled_inputs=chunk.sampled_inputs,
                positions_m=chunk.positions_m,
                sample_valid=chunk.sample_valid,
                impact_sample_indices=chunk.impact_sample_indices,
                trial_indices=np.array([item.trial_index for item in chunk.outcomes]),
                statuses=np.array([item.status.value for item in chunk.outcomes]),
                output_values=values,
                failure_types=np.array(
                    [item.failure_type or "" for item in chunk.outcomes]
                ),
                failure_messages=np.array(
                    [item.failure_message or "" for item in chunk.outcomes]
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())
        require(
            temporary.stat().st_size <= MAX_CHUNK_FILE_BYTES,
            "chunk file is too large",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_chunk(directory: Path, record: dict[str, object]) -> SimulationResultChunk:
    """Checksum, bound, decode, and reconstruct one immutable chunk."""
    path = directory / cast(str, record["file"])
    require(path.resolve().parent == directory, "durable chunk escapes archive")
    require(path.is_file(), "declared durable chunk is missing")
    require(path.stat().st_size <= MAX_CHUNK_FILE_BYTES, "chunk file is too large")
    require(file_sha256(path) == record["sha256"], "chunk checksum mismatch")
    _require_bounded_zip(path)
    try:
        with np.load(path, allow_pickle=False) as source:
            require(set(source.files) == _ARRAY_KEYS, "chunk array fields are invalid")
            arrays = {name: np.array(source[name], copy=True) for name in _ARRAY_KEYS}
    except (BadZipFile, OSError, ValueError) as exc:
        require(False, "durable chunk arrays are invalid", str(exc))
        raise AssertionError from exc
    return SimulationResultChunk(
        start_index=cast(int, record["start_index"]),
        sampled_inputs=arrays["sampled_inputs"],
        outcomes=_decode_outcomes(arrays),
        positions_m=arrays["positions_m"],
        sample_valid=arrays["sample_valid"],
        impact_sample_indices=arrays["impact_sample_indices"],
    )


def file_sha256(path: Path) -> str:
    """Hash one persisted file without materializing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_identity(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate durable manifest field", key)
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    require(False, "durable manifest numbers must be finite", value)


def _outcome_values(chunk: SimulationResultChunk) -> np.ndarray:
    result: np.ndarray = np.array(
        [
            [
                math.nan if outcome.value(name) is None else outcome.value(name)
                for name in ALL_OUTPUT_NAMES
            ]
            for outcome in chunk.outcomes
        ],
        dtype=float,
    )
    return result


def _require_bounded_zip(path: Path) -> None:
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            expected = {f"{name}.npy" for name in _ARRAY_KEYS}
            require(
                len(entries) == len(expected)
                and {item.filename for item in entries} == expected,
                "chunk ZIP members are invalid",
            )
            require(
                not any(item.flag_bits & 0x1 for item in entries),
                "encrypted chunk members are unsupported",
            )
            require(
                sum(item.file_size for item in entries) <= MAX_CHUNK_UNCOMPRESSED_BYTES,
                "chunk uncompressed byte limit exceeded",
            )
    except BadZipFile as exc:
        require(False, "durable chunk is not a valid NPZ archive", str(exc))


def _decode_outcomes(
    arrays: dict[str, np.ndarray],
) -> tuple[SimulationTrialOutcome, ...]:
    indices = arrays["trial_indices"]
    statuses = arrays["statuses"]
    values = arrays["output_values"]
    failure_types = arrays["failure_types"]
    failure_messages = arrays["failure_messages"]
    rows = indices.size
    require(indices.shape == (rows,), "chunk trial index shape is invalid")
    require(statuses.shape == (rows,), "chunk status shape is invalid")
    require(
        values.shape == (rows, len(ALL_OUTPUT_NAMES)),
        "chunk output shape is invalid",
    )
    require(failure_types.shape == (rows,), "chunk failure type shape is invalid")
    require(failure_messages.shape == (rows,), "chunk failure message shape is invalid")
    require(
        np.issubdtype(indices.dtype, np.integer)
        and not np.issubdtype(indices.dtype, np.bool_),
        "chunk trial indices must be genuine integers",
    )
    require(
        np.issubdtype(values.dtype, np.floating),
        "chunk outputs must be floating-point values",
    )
    require(
        bool(np.all(np.isfinite(values) | np.isnan(values))),
        "chunk outputs must be finite or unavailable NaN",
    )
    for name in ("statuses", "failure_types", "failure_messages"):
        require(
            np.issubdtype(arrays[name].dtype, np.str_),
            f"chunk {name} must contain strings",
        )
    outcomes: list[SimulationTrialOutcome] = []
    for row in range(rows):
        try:
            status = TrialEvaluationStatus(str(statuses[row]))
        except ValueError as exc:
            require(False, "chunk trial status is invalid", str(exc))
            raise AssertionError from exc
        outputs = {
            name: None if np.isnan(values[row, column]) else float(values[row, column])
            for column, name in enumerate(ALL_OUTPUT_NAMES)
        }
        outcomes.append(
            SimulationTrialOutcome(
                trial_index=int(indices[row]),
                status=status,
                values=outputs,
                failure_type=str(failure_types[row]) or None,
                failure_message=str(failure_messages[row]) or None,
            )
        )
    return tuple(outcomes)


__all__ = [
    "MANIFEST_NAME",
    "chunk_record",
    "file_sha256",
    "header_document",
    "json_sha256",
    "new_manifest",
    "read_chunk",
    "read_manifest",
    "verify_header",
    "write_chunk_atomic",
    "write_json_atomic",
]
