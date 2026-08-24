"""Strict JSON and all-trial CSV exports for chip forgiveness studies."""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from shared.python.swing_sim.variation.execution_metadata import (
    execution_document_to_json_dict,
)

from .chip_forgiveness import ChipStudySummary
from .forgiveness_runner import ChipForgivenessStudy


def _wire_value(value: Any) -> Any:
    """Convert immutable domain inputs to strict deterministic JSON values."""
    if isinstance(value, Enum):
        return _wire_value(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("study input contains a nonfinite number")
        return converted
    if isinstance(value, np.ndarray):
        return _wire_value(value.tolist())
    if is_dataclass(value):
        return {
            item.name: _wire_value(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _wire_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_wire_value(item) for item in value]
    raise TypeError(f"unsupported study input type: {type(value).__name__}")


def _summary_dict(summary: ChipStudySummary) -> dict[str, Any]:
    return {
        "sample_count": summary.sample_count,
        "cohorts": {
            cohort.value: {
                "count": estimate.count,
                "probability": estimate.probability,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
            }
            for cohort, estimate in summary.cohorts.items()
        },
        "expected_loss": summary.expected_loss,
        "expected_loss_ci": [
            summary.expected_loss_ci_low,
            summary.expected_loss_ci_high,
        ],
        "cvar_loss": summary.cvar_loss,
        "cvar_tail_fraction": summary.cvar_tail_fraction,
        "constraint_violation_rate": summary.constraint_violation_rate,
        "clean_contact_probability": summary.clean_contact_probability,
        "supports_turf_rankings": summary.supports_turf_rankings,
        "ranking_scope": summary.ranking_scope,
        "metric_distributions": [
            {
                "name": item.name,
                "support_count": item.support_count,
                "unavailable_count": item.unavailable_count,
                "p05": item.p05,
                "p50": item.p50,
                "p95": item.p95,
            }
            for item in summary.metric_distributions
        ],
        "convergence": [
            {
                "sample_count": item.sample_count,
                "mean_loss": item.mean_loss,
                "standard_error": item.standard_error,
            }
            for item in summary.convergence
        ],
    }


def chip_forgiveness_study_to_dict(study: ChipForgivenessStudy) -> dict[str, Any]:
    """Return the complete strict-JSON wire payload for one study."""
    if not isinstance(study, ChipForgivenessStudy):
        raise TypeError("study must be ChipForgivenessStudy")
    metadata = study.summary.metadata
    return {
        "schema_version": 2,
        "metadata": {
            "candidate_id": metadata.candidate_id,
            "plan_schema": metadata.plan_schema,
            "coordinate_frame": metadata.coordinate_frame,
            "seed": metadata.seed,
            "noise_model_id": metadata.noise_model_id,
            "objective_id": metadata.objective_id,
            "turf_profile_id": metadata.turf_profile_id,
            "turf_calibration_status": metadata.turf_calibration_status,
            "solver_id": metadata.solver_id,
            "sampling_design": metadata.sampling_design,
            "inference_method_id": metadata.inference_method_id,
            "limitations": metadata.limitations,
        },
        "input_names": list(study.input_names),
        "plan_document": execution_document_to_json_dict(study.plan),
        "sampled_inputs": study.sampled_inputs.tolist(),
        "physics_inputs": {
            "simulation_configs": _wire_value(study.request.ensemble.configs),
            "wedge_parameters": _wire_value(study.request.wedge_parameters),
            "ground": _wire_value(study.request.ground),
            "turf_profile": _wire_value(study.request.turf_profile),
            "loss_model": _wire_value(study.request.loss_model),
            "cvar_tail_fraction": study.request.cvar_tail_fraction,
            "bootstrap_samples": study.request.bootstrap_samples,
        },
        "records": [
            {
                "trial_index": record.trial_index,
                "cohort": record.cohort.value,
                "loss": record.loss,
                "constraint_violated": record.constraint_violated,
                "diagnostic": record.diagnostic,
                "turf_contact_status": record.turf_contact_status,
                "metrics": dict(record.metrics),
            }
            for record in study.records
        ],
        "summary": _summary_dict(study.summary),
    }


def chip_forgiveness_study_to_json(study: ChipForgivenessStudy) -> str:
    """Serialize complete evidence while rejecting non-standard NaN tokens."""
    return json.dumps(
        chip_forgiveness_study_to_dict(study),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def chip_forgiveness_study_to_csv(study: ChipForgivenessStudy) -> str:
    """Serialize one row per configured trial with unavailable metrics blank."""
    if not isinstance(study, ChipForgivenessStudy):
        raise TypeError("study must be ChipForgivenessStudy")
    metric_names = sorted({name for record in study.records for name in record.metrics})
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "trial",
            "cohort",
            "loss",
            "constraint_violated",
            "diagnostic",
            "turf_contact_status",
            *study.input_names,
            *metric_names,
        ]
    )
    for record in study.records:
        writer.writerow(
            [
                record.trial_index,
                record.cohort.value,
                record.loss,
                int(record.constraint_violated),
                record.diagnostic or "",
                record.turf_contact_status or "",
                *study.sampled_inputs[record.trial_index],
                *(record.metrics.get(name) for name in metric_names),
            ]
        )
    return stream.getvalue()


__all__ = [
    "chip_forgiveness_study_to_csv",
    "chip_forgiveness_study_to_dict",
    "chip_forgiveness_study_to_json",
]
