"""Content-integrity-checked HDF5 persistence for variation datasets."""

from __future__ import annotations

import importlib
import json
import math
import os
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np

from shared.python.contracts import PreconditionError, require

from ._execution_digest import canonical_sha256
from .dataset_io import DATASET_JSON_SCHEMA_VERSION, from_json_dict, to_json_dict
from .engine import VariationDataset

DATASET_HDF5_SCHEMA_ID = "rate-of-closure/variation-dataset-hdf5"
DATASET_HDF5_SCHEMA_VERSION = 1

_ATTRIBUTES = frozenset({"schema_id", "schema_version", "content_sha256"})
_MEMBERS = frozenset(
    {
        "plan_document_json",
        "input_names",
        "output_names",
        "inputs",
        "outputs",
        "success",
        "elapsed_s",
    }
)


class Hdf5UnavailableError(RuntimeError):
    """Raised when the optional HDF5 persistence dependency is unavailable."""


def _require_h5py() -> ModuleType:
    try:
        return importlib.import_module("h5py")
    except ModuleNotFoundError as exc:
        if exc.name != "h5py":
            raise
        raise Hdf5UnavailableError(
            "HDF5 variation persistence requires the optional dependency; "
            'install ud-tools with the "variation-hdf5" extra'
        ) from exc


def _archive_require(condition: bool, message: str, value: Any = None) -> None:
    """Enforce persisted-evidence checks independently of ``DBC_LEVEL``."""
    if not condition:
        raise PreconditionError(message, value)


def _write_archive(dataset: VariationDataset, path: Path, h5py: ModuleType) -> None:
    payload = to_json_dict(dataset)
    text_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as archive:
        archive.attrs["schema_id"] = DATASET_HDF5_SCHEMA_ID
        archive.attrs["schema_version"] = DATASET_HDF5_SCHEMA_VERSION
        archive.attrs["content_sha256"] = canonical_sha256(payload)
        archive.create_dataset(
            "plan_document_json",
            data=json.dumps(
                payload["plan_document"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            dtype=text_dtype,
        )
        archive.create_dataset(
            "input_names", data=list(dataset.input_names), dtype=text_dtype
        )
        archive.create_dataset(
            "output_names", data=list(dataset.output_names), dtype=text_dtype
        )
        archive.create_dataset("inputs", data=dataset.inputs, dtype="f8")
        archive.create_dataset("outputs", data=dataset.outputs, dtype="f8")
        archive.create_dataset(
            "success", data=dataset.success.astype(np.uint8), dtype="u1"
        )
        archive.create_dataset("elapsed_s", data=dataset.elapsed_s, dtype="f8")
        archive.flush()


def write_hdf5(dataset: VariationDataset, path: str | Path) -> None:
    """Atomically publish a new self-contained HDF5 dataset.

    Existing evidence is never replaced. Publication uses a sibling hard link,
    which is atomic and fails if another writer publishes the target first.
    """
    require(isinstance(dataset, VariationDataset), "dataset must be a VariationDataset")
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"HDF5 dataset already exists: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(
            f"HDF5 parent directory does not exist: {target.parent}"
        )
    h5py = _require_h5py()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_archive(dataset, temporary, h5py)
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FileExistsError(f"HDF5 dataset already exists: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _text(value: Any, field: str) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{field} must be valid UTF-8") from exc
    _archive_require(
        isinstance(value, str), f"{field} must be UTF-8 text", type(value).__name__
    )
    return cast(str, value)


def _names(values: Any, field: str) -> list[str]:
    array = np.asarray(values)
    _archive_require(array.ndim == 1, f"{field} must be one-dimensional", array.shape)
    names = [_text(value, field) for value in array.tolist()]
    _archive_require(all(names), f"{field} entries must be non-empty")
    _archive_require(len(set(names)) == len(names), f"{field} entries must be unique")
    return names


def _logical_document(archive: Any) -> dict[str, Any]:
    plan_document = json.loads(
        _text(archive["plan_document_json"][()], "plan_document_json")
    )
    inputs = np.asarray(archive["inputs"][()], dtype=float)
    outputs = np.asarray(archive["outputs"][()], dtype=float)
    _archive_require(inputs.ndim == 2, "inputs must be two-dimensional", inputs.shape)
    _archive_require(
        outputs.ndim == 2, "outputs must be two-dimensional", outputs.shape
    )
    output_rows = [
        [None if math.isnan(float(value)) else float(value) for value in row]
        for row in outputs.tolist()
    ]
    raw_success = np.asarray(archive["success"][()])
    _archive_require(
        raw_success.ndim == 1, "success must be one-dimensional", raw_success.shape
    )
    _archive_require(
        bool(np.all(np.isin(raw_success, (0, 1)))),
        "success values must be zero or one",
    )
    elapsed_raw = np.asarray(archive["elapsed_s"][()])
    _archive_require(
        elapsed_raw.shape == (), "elapsed_s must be scalar", elapsed_raw.shape
    )
    elapsed_s = float(elapsed_raw)
    _archive_require(
        math.isfinite(elapsed_s) and elapsed_s >= 0.0,
        "elapsed_s must be finite and nonnegative",
    )
    return {
        "schema_version": DATASET_JSON_SCHEMA_VERSION,
        "plan_document": plan_document,
        "input_names": _names(archive["input_names"][()], "input_names"),
        "output_names": _names(archive["output_names"][()], "output_names"),
        "inputs": inputs.tolist(),
        "outputs": output_rows,
        "success": [bool(value) for value in raw_success.tolist()],
        "elapsed_s": elapsed_s,
    }


def read_hdf5(path: str | Path) -> VariationDataset:
    """Read and verify a canonical HDF5 variation dataset."""
    h5py = _require_h5py()
    with h5py.File(Path(path), "r") as archive:
        _archive_require(set(archive.attrs) == _ATTRIBUTES, "HDF5 attributes mismatch")
        _archive_require(
            set(archive) == _MEMBERS, "HDF5 members mismatch", sorted(archive)
        )
        schema_id = _text(archive.attrs["schema_id"], "schema_id")
        _archive_require(schema_id == DATASET_HDF5_SCHEMA_ID, "HDF5 schema_id mismatch")
        raw_version = archive.attrs["schema_version"]
        _archive_require(
            isinstance(raw_version, (int, np.integer))
            and not isinstance(raw_version, (bool, np.bool_)),
            "schema_version must be an integer",
            raw_version,
        )
        version = int(raw_version)
        _archive_require(
            version == DATASET_HDF5_SCHEMA_VERSION,
            "unsupported HDF5 schema_version",
            version,
        )
        expected_digest = _text(archive.attrs["content_sha256"], "content_sha256")
        _archive_require(
            len(expected_digest) == 64
            and all(character in "0123456789abcdef" for character in expected_digest),
            "content_sha256 must be lowercase SHA-256",
        )
        document = _logical_document(archive)
        _archive_require(
            canonical_sha256(document) == expected_digest,
            "HDF5 content digest mismatch",
        )
    return from_json_dict(document)


__all__ = [
    "DATASET_HDF5_SCHEMA_ID",
    "DATASET_HDF5_SCHEMA_VERSION",
    "Hdf5UnavailableError",
    "read_hdf5",
    "write_hdf5",
]
