"""Variation-dataset export/import (CSV + JSON) — epic #4120, V3.

Schemas (mirrored by ``rate_of_closure/web/src/model/variation.ts``):

JSON (full round-trip, one document)::

    {
      "schema_version": 1,
      "plan": { ... VariationPlan.to_json_dict() ... },
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
from .spec import VariationPlan

_SCHEMA_VERSION = 1


def to_json_dict(dataset: VariationDataset) -> dict[str, Any]:
    """Dataset -> plain-JSON dict (documented module schema)."""
    outputs: list[list[float | None]] = [
        [None if math.isnan(v) else float(v) for v in row]
        for row in dataset.outputs.tolist()
    ]
    return {
        "schema_version": _SCHEMA_VERSION,
        "plan": dataset.plan.to_json_dict(),
        "input_names": list(dataset.input_names),
        "output_names": list(dataset.output_names),
        "inputs": dataset.inputs.tolist(),
        "outputs": outputs,
        "success": [bool(flag) for flag in dataset.success.tolist()],
        "elapsed_s": dataset.elapsed_s,
    }


def from_json_dict(data: dict[str, Any]) -> VariationDataset:
    """Inverse of :func:`to_json_dict` (DbC-validated on construction)."""
    version = integer(
        data.get("schema_version", _SCHEMA_VERSION),
        "schema_version",
        minimum=1,
    )
    require(version == _SCHEMA_VERSION, "unsupported schema_version", version)
    outputs = np.array(
        [[math.nan if v is None else float(v) for v in row] for row in data["outputs"]],
        dtype=float,
    )
    return VariationDataset(
        plan=VariationPlan.from_json_dict(data["plan"]),
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


__all__ = [
    "from_json_dict",
    "read_csv",
    "read_json",
    "to_json_dict",
    "write_csv",
    "write_json",
]
