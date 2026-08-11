"""Lossless exports for complete swing-variation ensembles."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from rate_of_closure.variation.simulation_types import SimulationEnsembleResult
from shared.python.swing_sim.variation.dataset_io import to_json_dict as dataset_json

ENSEMBLE_EXPORT_SCHEMA_VERSION = 1


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
    return {
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


def write_json(result: SimulationEnsembleResult, path: str | Path) -> None:
    """Write the complete typed outcomes and common-grid position traces."""
    Path(path).write_text(json.dumps(to_json_dict(result), indent=2), encoding="utf-8")


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
    "to_json_dict",
    "write_json",
    "write_trace_csv",
]
