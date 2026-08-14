"""Header, side-array, and commit storage for ensemble chunk archives."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, cast

import numpy as np

from shared.python.contracts import require
from shared.python.swing_sim.variation.spec import VariationPlan

from ._ensemble_limits import (
    MAX_ARCHIVE_CHUNKS,
    MAX_ARCHIVE_HEADER_BYTES,
    MAX_INPUT_CELLS,
    MAX_SAMPLES,
)
from .ensemble_archive_contracts import (
    ARCHIVE_SCHEMA_ID,
    ARCHIVE_SCHEMA_VERSION,
    CommittedEnsembleArchive,
    canonical_json_bytes,
    require_sha256,
    strict_json_bytes,
)
from .ensemble_archive_documents import (
    exact_int,
    exact_mapping,
    finite_number,
    layout_document,
    layout_from_document,
    string_tuple,
)
from .ensemble_chunks import EnsembleStreamHeader
from .ensemble_trace_authority import EnsembleAuthorityLayout

HEADER_NAME = "archive.json"
INPUTS_NAME = "sampled-inputs.f64"
TIMES_NAME = "sample-times.f64"
COMMIT_NAME = "commit.json"
_CHUNK_PATTERN = re.compile(r"^(\d{12})-(\d{12})\.roc$")


def sha256_file(path: Path) -> str:
    """Return a bounded-memory SHA-256 digest for one archive file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    """Atomically replace ``path`` with flushed bytes."""
    temporary = path.with_name(f"{path.name}.partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_buffer(path: Path, data: memoryview) -> None:
    """Atomically write a contiguous buffer without a same-size bytes copy."""
    temporary = path.with_name(f"{path.name}.partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _array_file(path: Path, values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8", order="C")
    _atomic_buffer(path, memoryview(cast(Any, array)).cast("B"))
    return sha256_file(path)


def _header_document(
    header: EnsembleStreamHeader, input_sha256: str, time_sha256: str
) -> dict[str, object]:
    layout = header.authority_layout
    require(layout is not None, "durable archive requires trace authority")
    layout = cast(EnsembleAuthorityLayout, layout)
    request_sha = require_sha256(header.request_identity_sha256, "request identity")
    return {
        "schema_id": ARCHIVE_SCHEMA_ID,
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "plan": header.plan.to_json_dict(),
        "coordinate_frame": header.coordinate_frame,
        "point_ids": list(header.point_ids),
        "authority_layout": layout_document(layout),
        "request_identity_sha256": request_sha,
        "sampled_inputs": {
            "file": INPUTS_NAME,
            "dtype": "<f8",
            "shape": list(header.sampled_inputs.shape),
            "sha256": input_sha256,
        },
        "sample_times_s": {
            "file": TIMES_NAME,
            "dtype": "<f8",
            "shape": list(header.sample_times_s.shape),
            "sha256": time_sha256,
        },
    }


def initialize_archive(path: Path, header: EnsembleStreamHeader) -> str:
    """Create or clean one headerless provisional archive."""
    if path.exists():
        require(path.is_dir(), "archive path must be a directory")
        entries = list(path.iterdir())
        allowed = {INPUTS_NAME, TIMES_NAME, "chunks", f"{HEADER_NAME}.partial"}
        require(
            all(
                item.name in allowed or item.name.endswith(".partial")
                for item in entries
            ),
            "headerless archive contains unrecognized files",
        )
        chunks = path / "chunks"
        require(
            not chunks.exists() or not any(chunks.glob("*.roc")),
            "headerless archive cannot contain committed chunks",
        )
        for item in entries:
            if item.is_file():
                item.unlink()
        if chunks.exists():
            for item in chunks.iterdir():
                require(
                    item.is_file() and item.name.endswith(".partial"),
                    "invalid provisional chunk",
                )
                item.unlink()
    else:
        path.mkdir(parents=True)
    (path / "chunks").mkdir(exist_ok=True)
    input_sha = _array_file(path / INPUTS_NAME, header.sampled_inputs)
    time_sha = _array_file(path / TIMES_NAME, header.sample_times_s)
    encoded = canonical_json_bytes(_header_document(header, input_sha, time_sha))
    require(
        len(encoded) <= MAX_ARCHIVE_HEADER_BYTES, "archive header byte limit exceeded"
    )
    atomic_bytes(path / HEADER_NAME, encoded)
    return hashlib.sha256(encoded).hexdigest()


def _load_array(
    path: Path, value: object, name: str, expected_shape: tuple[int, ...]
) -> np.ndarray:
    data = exact_mapping(value, {"file", "dtype", "shape", "sha256"}, name)
    require(data["file"] == path.name and data["dtype"] == "<f8", f"invalid {name}")
    require(isinstance(data["shape"], list), f"{name} shape must be an array")
    shape = tuple(exact_int(item, f"{name} dimension") for item in data["shape"])
    require(shape == expected_shape, f"{name} shape does not match plan")
    cells = int(np.prod(shape, dtype=object))
    require(cells <= MAX_INPUT_CELLS or name == "sample times", f"{name} is too large")
    require(path.stat().st_size == cells * 8, f"{name} file size is invalid")
    require(
        sha256_file(path) == require_sha256(data["sha256"], name),
        f"{name} checksum mismatch",
    )
    values = np.fromfile(path, dtype="<f8", count=cells).reshape(shape)
    require(bool(np.all(np.isfinite(values))), f"{name} must be finite")
    return cast(np.ndarray, values)


def load_header(path: Path) -> tuple[EnsembleStreamHeader, str]:
    """Load and verify archive metadata plus its side arrays."""
    header_path = path / HEADER_NAME
    require(
        header_path.is_file()
        and header_path.stat().st_size <= MAX_ARCHIVE_HEADER_BYTES,
        "archive header is absent or too large",
    )
    raw = header_path.read_bytes()
    value = strict_json_bytes(raw, maximum_bytes=MAX_ARCHIVE_HEADER_BYTES)
    data = exact_mapping(
        value,
        {
            "schema_id",
            "schema_version",
            "plan",
            "coordinate_frame",
            "point_ids",
            "authority_layout",
            "request_identity_sha256",
            "sampled_inputs",
            "sample_times_s",
        },
        "archive header",
    )
    require(data["schema_id"] == ARCHIVE_SCHEMA_ID, "unsupported archive schema")
    require(
        data["schema_version"] == ARCHIVE_SCHEMA_VERSION,
        "unsupported archive version",
    )
    require(isinstance(data["plan"], dict), "plan must be an object")
    plan = VariationPlan.from_json_dict(data["plan"])
    inputs = _load_array(
        path / INPUTS_NAME,
        data["sampled_inputs"],
        "sampled inputs",
        (plan.n_runs, len(plan.noise)),
    )
    raw_times = exact_mapping(
        data["sample_times_s"], {"file", "dtype", "shape", "sha256"}, "times"
    )
    require(
        isinstance(raw_times["shape"], list) and len(raw_times["shape"]) == 1,
        "invalid time shape",
    )
    sample_count = exact_int(raw_times["shape"][0], "sample count", minimum=1)
    require(sample_count <= MAX_SAMPLES, "sample limit exceeded")
    times = _load_array(
        path / TIMES_NAME, data["sample_times_s"], "sample times", (sample_count,)
    )
    require(isinstance(data["coordinate_frame"], str), "coordinate frame must be text")
    header = EnsembleStreamHeader(
        plan,
        inputs,
        times,
        string_tuple(data["point_ids"], "point_ids"),
        data["coordinate_frame"],
        layout_from_document(data["authority_layout"]),
        require_sha256(data["request_identity_sha256"], "request identity"),
    )
    return header, hashlib.sha256(raw).hexdigest()


def chunk_paths(path: Path) -> list[Path]:
    """Return the bounded canonical chunk path sequence."""
    chunks = sorted((path / "chunks").glob("*.roc"))
    require(len(chunks) <= MAX_ARCHIVE_CHUNKS, "archive chunk-count limit exceeded")
    require(
        all(_CHUNK_PATTERN.fullmatch(item.name) for item in chunks),
        "invalid chunk filename",
    )
    return chunks


def require_same_header(
    stored: EnsembleStreamHeader, supplied: EnsembleStreamHeader
) -> None:
    """Require exact immutable request/header identity for resume."""
    require(stored.plan == supplied.plan, "resume plan does not match archive")
    require(
        np.array_equal(stored.sampled_inputs, supplied.sampled_inputs),
        "resume inputs changed",
    )
    require(
        np.array_equal(stored.sample_times_s, supplied.sample_times_s),
        "resume grid changed",
    )
    require(stored.point_ids == supplied.point_ids, "resume point IDs changed")
    require(
        stored.coordinate_frame == supplied.coordinate_frame, "resume frame changed"
    )
    require(
        stored.authority_layout == supplied.authority_layout, "resume layout changed"
    )
    require(
        stored.request_identity_sha256 == supplied.request_identity_sha256,
        "resume request changed",
    )


def load_commit(path: Path, archive_sha256: str) -> CommittedEnsembleArchive:
    """Load one completed commit marker bound to its header digest."""
    commit_path = path / COMMIT_NAME
    require(
        commit_path.is_file(), "archive is provisional and not readable as completed"
    )
    require(
        commit_path.stat().st_size <= MAX_ARCHIVE_HEADER_BYTES, "commit is too large"
    )
    value = strict_json_bytes(
        commit_path.read_bytes(), maximum_bytes=MAX_ARCHIVE_HEADER_BYTES
    )
    data = exact_mapping(
        value,
        {
            "schema_id",
            "schema_version",
            "archive_sha256",
            "scientific_root_sha256",
            "trial_count",
            "chunk_count",
            "elapsed_s",
        },
        "archive commit",
    )
    require(data["schema_id"] == ARCHIVE_SCHEMA_ID, "invalid commit schema")
    require(data["schema_version"] == ARCHIVE_SCHEMA_VERSION, "invalid commit version")
    require(data["archive_sha256"] == archive_sha256, "commit header digest mismatch")
    return CommittedEnsembleArchive(
        path,
        require_sha256(data["scientific_root_sha256"], "scientific root"),
        exact_int(data["trial_count"], "trial count"),
        exact_int(data["chunk_count"], "chunk count"),
        finite_number(data["elapsed_s"], "elapsed_s"),
    )


__all__ = [
    "COMMIT_NAME",
    "HEADER_NAME",
    "atomic_bytes",
    "chunk_paths",
    "initialize_archive",
    "load_commit",
    "load_header",
    "require_same_header",
]
