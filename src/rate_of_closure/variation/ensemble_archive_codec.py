"""Deterministic bounded binary codec for one complete ensemble chunk."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast

import numpy as np

from shared.python.contracts import require

from ._ensemble_limits import (
    MAX_ARCHIVE_DESCRIPTOR_BYTES,
    MAX_CHUNK_AUTHORITY_BYTES,
    MAX_INPUT_CELLS,
)
from .ensemble_archive_contracts import canonical_json_bytes, strict_json_bytes
from .ensemble_archive_documents import (
    event_document,
    event_from_document,
    exact_int,
    exact_mapping,
    outcome_document,
    outcome_from_document,
)
from .ensemble_chunks import (
    MAX_CHUNK_POSITION_CELLS,
    EnsembleStreamHeader,
    SimulationResultChunk,
    require_chunk_matches_header,
)
from .ensemble_trace_authority import ChunkTraceAuthority, EnsembleAuthorityLayout

CHUNK_MAGIC = b"ROCCHNK1\n"
_DIGEST_BYTES = 32
_ARRAY_ORDER = (
    "sampled_inputs",
    "positions_m",
    "sample_valid",
    "impact_sample_indices",
    "poses_app",
    "twists_app_si",
    "generalized_states",
    "applied_torques_nm",
    "preimpact_valid",
)
_DTYPES = {
    "sampled_inputs": "<f8",
    "positions_m": "<f8",
    "sample_valid": "|u1",
    "impact_sample_indices": "<i8",
    "poses_app": "<f8",
    "twists_app_si": "<f8",
    "generalized_states": "<f8",
    "applied_torques_nm": "<f8",
    "preimpact_valid": "|u1",
}


class _Digest(Protocol):
    """Minimal hash interface used by the streaming codec."""

    def update(self, data: bytes | bytearray | memoryview) -> None: ...

    def digest(self) -> bytes: ...


def _arrays(chunk: SimulationResultChunk) -> dict[str, np.ndarray]:
    authority = chunk.authority
    require(authority is not None, "durable chunks require complete trace authority")
    authority = cast(ChunkTraceAuthority, authority)
    return {
        "sampled_inputs": np.asarray(chunk.sampled_inputs, dtype="<f8", order="C"),
        "positions_m": np.asarray(chunk.positions_m, dtype="<f8", order="C"),
        "sample_valid": np.asarray(chunk.sample_valid, dtype="|u1", order="C"),
        "impact_sample_indices": np.asarray(
            chunk.impact_sample_indices, dtype="<i8", order="C"
        ),
        "poses_app": np.asarray(authority.poses_app, dtype="<f8", order="C"),
        "twists_app_si": np.asarray(authority.twists_app_si, dtype="<f8", order="C"),
        "generalized_states": np.asarray(
            authority.generalized_states, dtype="<f8", order="C"
        ),
        "applied_torques_nm": np.asarray(
            authority.applied_torques_nm, dtype="<f8", order="C"
        ),
        "preimpact_valid": np.asarray(
            authority.preimpact_valid, dtype="|u1", order="C"
        ),
    }


def _descriptor(
    chunk: SimulationResultChunk,
    arrays: dict[str, np.ndarray],
    previous_sha256: str,
) -> dict[str, object]:
    authority = chunk.authority
    assert authority is not None
    return {
        "schema_version": 1,
        "start_index": chunk.start_index,
        "row_count": len(chunk.outcomes),
        "previous_chunk_sha256": previous_sha256,
        "arrays": [
            {
                "name": name,
                "dtype": _DTYPES[name],
                "shape": list(arrays[name].shape),
                "nbytes": arrays[name].nbytes,
            }
            for name in _ARRAY_ORDER
        ],
        "outcomes": [outcome_document(item) for item in chunk.outcomes],
        "events": [event_document(item) for item in authority.events],
    }


def _seeded_digest(archive_sha256: str, previous_sha256: str) -> _Digest:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(archive_sha256))
    digest.update(bytes.fromhex(previous_sha256))
    return digest


def write_chunk_file(
    path: Path,
    chunk: SimulationResultChunk,
    archive_sha256: str,
    previous_sha256: str,
) -> str:
    """Stream one deterministic chunk to a new file and return its chain digest."""
    arrays = _arrays(chunk)
    descriptor = canonical_json_bytes(_descriptor(chunk, arrays, previous_sha256))
    require(
        len(descriptor) <= MAX_ARCHIVE_DESCRIPTOR_BYTES,
        "archive descriptor byte limit exceeded",
    )
    payload_bytes = sum(values.nbytes for values in arrays.values())
    require(
        payload_bytes
        <= MAX_CHUNK_AUTHORITY_BYTES
        + chunk.positions_m.nbytes
        + chunk.sampled_inputs.nbytes
        + chunk.sample_valid.nbytes
        + chunk.impact_sample_indices.nbytes,
        "chunk payload byte limit exceeded",
    )
    digest = _seeded_digest(archive_sha256, previous_sha256)
    with path.open("xb") as handle:
        _write_hashed(handle, CHUNK_MAGIC, digest)
        _write_hashed(handle, len(descriptor).to_bytes(4, "little"), digest)
        _write_hashed(handle, descriptor, digest)
        for name in _ARRAY_ORDER:
            if arrays[name].nbytes == 0:
                continue
            data = memoryview(cast(Any, arrays[name])).cast("B")
            _write_hashed(handle, data, digest)
        final = digest.digest()
        handle.write(final)
        handle.flush()
        os.fsync(handle.fileno())
    return final.hex()


def _write_hashed(handle: BinaryIO, data: object, digest: _Digest) -> None:
    block = cast(bytes | bytearray | memoryview, data)
    handle.write(block)
    digest.update(block)


def _read_descriptor(path: Path) -> tuple[dict[str, object], int]:
    with path.open("rb") as handle:
        require(handle.read(len(CHUNK_MAGIC)) == CHUNK_MAGIC, "invalid chunk magic")
        raw_length = handle.read(4)
        require(len(raw_length) == 4, "truncated chunk descriptor length")
        length = int.from_bytes(raw_length, "little")
        require(0 < length <= MAX_ARCHIVE_DESCRIPTOR_BYTES, "invalid descriptor length")
        raw = handle.read(length)
        require(len(raw) == length, "truncated chunk descriptor")
    value = strict_json_bytes(raw, maximum_bytes=MAX_ARCHIVE_DESCRIPTOR_BYTES)
    data = exact_mapping(
        value,
        {
            "schema_version",
            "start_index",
            "row_count",
            "previous_chunk_sha256",
            "arrays",
            "outcomes",
            "events",
        },
        "chunk descriptor",
    )
    require(data["schema_version"] == 1, "unsupported chunk schema version")
    return data, len(CHUNK_MAGIC) + 4 + length


def _verify_digest(path: Path, archive_sha256: str, previous_sha256: str) -> str:
    size = path.stat().st_size
    require(size > _DIGEST_BYTES, "truncated chunk file")
    digest = _seeded_digest(archive_sha256, previous_sha256)
    remaining = size - _DIGEST_BYTES
    with path.open("rb") as handle:
        while remaining:
            block = handle.read(min(1_048_576, remaining))
            require(bool(block), "truncated chunk payload")
            digest.update(block)
            remaining -= len(block)
        stored = handle.read(_DIGEST_BYTES)
        require(len(stored) == _DIGEST_BYTES, "truncated chunk checksum")
    calculated = digest.digest()
    require(stored == calculated, "chunk checksum mismatch")
    return calculated.hex()


def _array_specs(
    value: object, expected_shapes: dict[str, tuple[int, ...]]
) -> tuple[list[dict[str, object]], int]:
    require(
        isinstance(value, list) and len(value) == len(_ARRAY_ORDER), "invalid arrays"
    )
    items = cast(list[object], value)
    specs: list[dict[str, object]] = []
    total = 0
    for raw, name in zip(items, _ARRAY_ORDER, strict=True):
        spec = exact_mapping(raw, {"name", "dtype", "shape", "nbytes"}, "array")
        require(
            spec["name"] == name and spec["dtype"] == _DTYPES[name],
            "array schema changed",
        )
        require(isinstance(spec["shape"], list), "array shape must be an array")
        shape = tuple(exact_int(item, "array dimension") for item in spec["shape"])
        require(shape == expected_shapes[name], "array shape does not match header")
        expected_bytes = (
            int(np.prod(shape, dtype=object)) * np.dtype(_DTYPES[name]).itemsize
        )
        require(spec["nbytes"] == expected_bytes, "array byte count is invalid")
        total += expected_bytes
        specs.append(spec)
    return specs, total


def read_chunk_file(
    path: Path,
    header: EnsembleStreamHeader,
    archive_sha256: str,
    previous_sha256: str,
    next_index: int,
) -> tuple[SimulationResultChunk, str]:
    """Verify then load one bounded chunk without retaining adjacent chunks."""
    data, payload_offset = _read_descriptor(path)
    require(data["previous_chunk_sha256"] == previous_sha256, "broken chunk chain")
    start = exact_int(data["start_index"], "start_index")
    rows = exact_int(data["row_count"], "row_count", minimum=1)
    require(start == next_index, "chunk stream contains a gap or overlap")
    samples = header.sample_times_s.size
    points = len(header.point_ids)
    layout = header.authority_layout
    require(layout is not None, "archive header requires authority layout")
    layout = cast(EnsembleAuthorityLayout, layout)
    shapes = {
        "sampled_inputs": (rows, len(header.plan.noise)),
        "positions_m": (rows, samples, points, 3),
        "sample_valid": (rows, samples),
        "impact_sample_indices": (rows,),
        "poses_app": (rows, samples, 4, 4),
        "twists_app_si": (rows, samples, 6),
        "generalized_states": (rows, samples, len(layout.state_ids)),
        "applied_torques_nm": (rows, samples, len(layout.torque_joint_ids)),
        "preimpact_valid": (rows, samples),
    }
    specs, payload_bytes = _array_specs(data["arrays"], shapes)
    by_name = {cast(str, spec["name"]): spec for spec in specs}
    authority_bytes = sum(
        exact_int(by_name[name]["nbytes"], "authority byte count")
        for name in (
            "poses_app",
            "twists_app_si",
            "generalized_states",
            "applied_torques_nm",
            "preimpact_valid",
        )
    )
    require(
        authority_bytes <= MAX_CHUNK_AUTHORITY_BYTES, "authority byte limit exceeded"
    )
    require(
        int(np.prod(shapes["positions_m"], dtype=object)) <= MAX_CHUNK_POSITION_CELLS,
        "chunk position cell limit exceeded",
    )
    require(
        int(np.prod(shapes["sampled_inputs"], dtype=object)) <= MAX_INPUT_CELLS,
        "chunk input cell limit exceeded",
    )
    expected_name = f"{start:012d}-{start + rows:012d}.roc"
    require(path.name == expected_name, "chunk filename does not match descriptor")
    require(
        path.stat().st_size == payload_offset + payload_bytes + _DIGEST_BYTES,
        "chunk file size does not match its descriptor",
    )
    chunk_sha256 = _verify_digest(path, archive_sha256, previous_sha256)
    arrays: dict[str, np.ndarray] = {}
    with path.open("rb") as handle:
        handle.seek(payload_offset)
        for spec, name in zip(specs, _ARRAY_ORDER, strict=True):
            count = (
                exact_int(spec["nbytes"], "array byte count")
                // np.dtype(_DTYPES[name]).itemsize
            )
            values = np.fromfile(handle, dtype=_DTYPES[name], count=count)
            require(values.size == count, "truncated chunk array")
            if name in ("sample_valid", "preimpact_valid"):
                require(
                    bool(np.all((values == 0) | (values == 1))),
                    f"{name} bytes must be canonical booleans",
                )
            arrays[name] = values.reshape(shapes[name])
    require(isinstance(data["outcomes"], list), "outcomes must be an array")
    require(isinstance(data["events"], list), "events must be an array")
    raw_outcomes = cast(list[object], data["outcomes"])
    raw_events = cast(list[object], data["events"])
    outcomes = tuple(outcome_from_document(item) for item in raw_outcomes)
    events = tuple(event_from_document(item) for item in raw_events)
    require(len(outcomes) == rows and len(events) == rows, "row metadata mismatch")
    authority = ChunkTraceAuthority(
        arrays["poses_app"],
        arrays["twists_app_si"],
        arrays["generalized_states"],
        arrays["applied_torques_nm"],
        arrays["preimpact_valid"].astype(bool, copy=False),
        events,
    )
    chunk = SimulationResultChunk(
        start,
        arrays["sampled_inputs"],
        outcomes,
        arrays["positions_m"],
        arrays["sample_valid"].astype(bool, copy=False),
        arrays["impact_sample_indices"],
        authority,
    )
    require_chunk_matches_header(header, chunk, next_index)
    return chunk, chunk_sha256


__all__ = ["CHUNK_MAGIC", "read_chunk_file", "write_chunk_file"]
