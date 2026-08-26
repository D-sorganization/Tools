"""Variation-dataset export/import (CSV + JSON) — epic #4120, V3.

Schemas (mirrored by ``rate_of_closure/web/src/model/variation.ts``):

JSON (full round-trip, one document)::

    {
      "schema_version": 2,
      "plan_document": { ... canonical execution document v3 ... },
      "input_names": ["swing_sim.impact.delivery.face_angle_deg", ...],
      "output_names": ["club_path_deg", ...],
      "inputs":  [[...], ...],   # n_runs x n_inputs
      "outputs": [[...], ...],   # n_runs x n_outputs, null for failures
      "success": [true, ...],
      "elapsed_s": 1.23
    }

CSV (spreadsheet-friendly; the plan itself is JSON-only)::

    run,success,<input key>...,<output name>...
    0,1,-0.31,...,231.2,...

Failed runs keep their sampled inputs and write empty output cells.
CSV import therefore needs the plan (pass the JSON document, or the
plan object, alongside) — :func:`read_csv` takes it explicitly.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from shared.python.contracts import require
from shared.python.swing_sim._numeric_contracts import integer

from .engine import VariationDataset
from .execution_metadata import (
    execution_document_from_json_dict,
    execution_document_to_json_dict,
)
from .spec import VariationPlan

_SCHEMA_VERSION = 2
DATASET_JSON_SCHEMA_VERSION = _SCHEMA_VERSION
DATASET_HDF5_SCHEMA_ID = "rate-of-closure/variation-dataset-hdf5"
DATASET_HDF5_SCHEMA_VERSION = 1
_JSON_FIELDS = frozenset(
    {
        "schema_version",
        "plan_document",
        "input_names",
        "output_names",
        "inputs",
        "outputs",
        "success",
        "elapsed_s",
    }
)


def to_json_dict(dataset: VariationDataset) -> dict[str, Any]:
    """Dataset -> plain-JSON dict (documented module schema)."""
    outputs: list[list[float | None]] = [
        [None if math.isnan(v) else float(v) for v in row]
        for row in dataset.outputs.tolist()
    ]
    return {
        "schema_version": _SCHEMA_VERSION,
        "plan_document": execution_document_to_json_dict(dataset.plan),
        "input_names": list(dataset.input_names),
        "output_names": list(dataset.output_names),
        "inputs": dataset.inputs.tolist(),
        "outputs": outputs,
        "success": [bool(flag) for flag in dataset.success.tolist()],
        "elapsed_s": dataset.elapsed_s,
    }


def from_json_dict(data: dict[str, Any]) -> VariationDataset:
    """Inverse of :func:`to_json_dict` (DbC-validated on construction)."""
    require(isinstance(data, dict), "dataset must be a JSON object", data)
    version = integer(
        data.get("schema_version"),
        "schema_version",
        minimum=1,
    )
    require(
        version != 1,
        "legacy dataset plan is not self-contained; re-run or explicitly migrate it",
        version,
    )
    require(version == _SCHEMA_VERSION, "unsupported schema_version", version)
    require(set(data) == _JSON_FIELDS, "dataset fields mismatch", sorted(data))
    plan_document = execution_document_from_json_dict(data["plan_document"])
    outputs = np.array(
        [[math.nan if v is None else float(v) for v in row] for row in data["outputs"]],
        dtype=float,
    )
    return VariationDataset(
        plan=plan_document.plan,
        input_names=tuple(str(name) for name in data["input_names"]),
        inputs=np.asarray(data["inputs"], dtype=float),
        output_names=tuple(str(name) for name in data["output_names"]),
        outputs=outputs,
        success=np.asarray(data["success"], dtype=bool),
        elapsed_s=float(data.get("elapsed_s", 0.0)),
    )


def write_json(dataset: VariationDataset, path: str | Path) -> None:
    """Write the dataset (plan included) to a JSON file."""
    Path(path).write_text(json.dumps(to_json_dict(dataset), indent=2), encoding="utf-8")


def read_json(path: str | Path) -> VariationDataset:
    """Read a dataset written by :func:`write_json`."""
    return from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_csv(dataset: VariationDataset, path: str | Path) -> None:
    """Write the runs table to CSV (documented module schema)."""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "success", *dataset.input_names, *dataset.output_names])
        for i in range(dataset.plan.n_runs):
            outputs = [
                "" if math.isnan(v) else repr(float(v))
                for v in dataset.outputs[i].tolist()
            ]
            writer.writerow(
                [
                    i,
                    int(dataset.success[i]),
                    *[repr(float(v)) for v in dataset.inputs[i].tolist()],
                    *outputs,
                ]
            )


def read_csv(path: str | Path, plan: VariationPlan) -> VariationDataset:
    """Read a dataset written by :func:`write_csv`.

    The CSV carries no plan (see module docstring), so the caller supplies
    it; header names must match the plan's noise keys and mode outputs.
    """
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    require(len(rows) >= 1, "CSV must contain a header row", None)
    header = rows[0]
    n_inputs = len(plan.noise)
    require(
        header[:2] == ["run", "success"],
        "CSV header must start with run,success",
        header[:2],
    )
    input_names = tuple(header[2 : 2 + n_inputs])
    output_names = tuple(header[2 + n_inputs :])
    expected = tuple(spec.variable_key for spec in plan.noise)
    require(input_names == expected, "CSV input columns must match plan", input_names)
    body = rows[1:]
    require(len(body) == plan.n_runs, "CSV row count must match plan", len(body))
    inputs: np.ndarray = np.empty((plan.n_runs, n_inputs), dtype=float)
    outputs = np.full((plan.n_runs, len(output_names)), np.nan)
    success: np.ndarray = np.zeros(plan.n_runs, dtype=bool)
    for row in body:
        i = int(row[0])
        require(0 <= i < plan.n_runs, "run index out of range", i)
        success[i] = bool(int(row[1]))
        inputs[i] = [float(v) for v in row[2 : 2 + n_inputs]]
        outputs[i] = [math.nan if v == "" else float(v) for v in row[2 + n_inputs :]]
    return VariationDataset(
        plan=plan,
        input_names=input_names,
        inputs=inputs,
        output_names=output_names,
        outputs=outputs,
        success=success,
    )


def write_hdf5(dataset: VariationDataset, path: str | Path) -> None:
    """Write a self-contained variation dataset to the versioned HDF5 schema."""
    import h5py

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        handle.attrs["schema_id"] = DATASET_HDF5_SCHEMA_ID
        handle.attrs["schema_version"] = DATASET_HDF5_SCHEMA_VERSION
        handle.attrs["elapsed_s"] = float(dataset.elapsed_s)
        handle.attrs["plan_document"] = json.dumps(
            execution_document_to_json_dict(dataset.plan)
        )
        handle.attrs["input_names"] = json.dumps(list(dataset.input_names))
        handle.attrs["output_names"] = json.dumps(list(dataset.output_names))
        handle.create_dataset("inputs", data=np.asarray(dataset.inputs, dtype=float))
        handle.create_dataset("outputs", data=np.asarray(dataset.outputs, dtype=float))
        handle.create_dataset("success", data=np.asarray(dataset.success, dtype=bool))


def read_hdf5(path: str | Path) -> VariationDataset:
    """Read the exact versioned HDF5 dataset emitted by :func:`write_hdf5`."""
    import h5py

    with h5py.File(Path(path), "r") as handle:
        schema_id = handle.attrs.get("schema_id")
        if isinstance(schema_id, bytes):
            schema_id = schema_id.decode("utf-8")
        require(
            schema_id == DATASET_HDF5_SCHEMA_ID,
            "unsupported HDF5 schema_id",
            schema_id,
        )
        schema_version = int(handle.attrs.get("schema_version", 0))
        require(
            schema_version == DATASET_HDF5_SCHEMA_VERSION,
            "unsupported HDF5 schema_version",
            schema_version,
        )

        def json_attribute(name: str) -> Any:
            raw = handle.attrs.get(name)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            require(isinstance(raw, str), f"HDF5 {name} must be JSON text", raw)
            return json.loads(raw)

        plan_document = execution_document_from_json_dict(
            json_attribute("plan_document")
        )
        input_names = tuple(json_attribute("input_names"))
        output_names = tuple(json_attribute("output_names"))
        inputs = np.asarray(handle["inputs"], dtype=float)
        outputs = np.asarray(handle["outputs"], dtype=float)
        success = np.asarray(handle["success"], dtype=bool)
        elapsed_s = float(handle.attrs.get("elapsed_s", 0.0))
    return VariationDataset(
        plan=plan_document.plan,
        input_names=input_names,
        inputs=inputs,
        output_names=output_names,
        outputs=outputs,
        success=success,
        elapsed_s=elapsed_s,
    )


__all__ = [
    "DATASET_HDF5_SCHEMA_ID",
    "DATASET_HDF5_SCHEMA_VERSION",
    "DATASET_JSON_SCHEMA_VERSION",
    "from_json_dict",
    "read_csv",
    "read_hdf5",
    "read_json",
    "to_json_dict",
    "write_csv",
    "write_hdf5",
    "write_json",
]
