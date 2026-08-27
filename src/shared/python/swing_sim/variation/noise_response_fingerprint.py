"""Deterministic identities for response inputs, state, and final fields."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from shared.python.contracts import require

from .noise_response_record import PositionNoiseResponseField
from .noise_response_types import ResponseFieldInput


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    identity = f"{array.dtype.str}:{array.shape}:".encode()
    return hashlib.sha256(identity + array.tobytes(order="C")).hexdigest()


def input_contract_fingerprint(inputs: tuple[ResponseFieldInput, ...]) -> str:
    """Bind input metadata, trace arrays, masks, and paired perturbations."""
    records: list[dict[str, object]] = []
    for field_input in inputs:
        baseline = field_input.baseline.traces
        perturbed = field_input.perturbed.traces
        metadata = field_input.execution_metadata
        records.append(
            {
                "spec_id": field_input.spec_id,
                "adapter_id": field_input.adapter_id,
                "source_layout_id": field_input.source_layout_id,
                "trial_ids": list(field_input.trial_ids),
                "source_sha256": field_input.source_sha256,
                "plan_sha256": metadata.plan_sha256,
                "registry_sha256": metadata.registry_sha256,
                "provenance_sha256": metadata.provenance_sha256,
                "baseline_positions": _array_sha256(baseline.positions_m),
                "baseline_valid": _array_sha256(baseline.sample_valid),
                "perturbed_positions": _array_sha256(perturbed.positions_m),
                "perturbed_valid": _array_sha256(perturbed.sample_valid),
                "input_delta": _array_sha256(field_input.input_delta),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _field_metadata(field: PositionNoiseResponseField) -> dict[str, object]:
    return {
        "schema_id": field.schema_id,
        "schema_version": field.schema_version,
        "method_id": field.method_id,
        "normalization_id": field.normalization_id,
        "resampling_policy_id": field.resampling_policy_id,
        "coordinate_frame": field.coordinate_frame,
        "point_ids": list(field.point_ids),
        "trial_ids": list(field.trial_ids),
        "input_ids": list(field.input_ids),
        "input_units": list(field.input_units),
        "source_layout_ids": list(field.source_layout_ids),
        "adapter_ids": list(field.adapter_ids),
        "source_sha256": list(field.source_sha256),
        "plan_sha256": list(field.plan_sha256),
        "registry_sha256": list(field.registry_sha256),
        "execution_provenance_sha256": list(field.execution_provenance_sha256),
        "metric_ids": list(field.metric_ids),
        "metric_units": list(field.metric_units),
        "scientific_boundary": field.scientific_boundary,
    }


def response_field_fingerprint(field: PositionNoiseResponseField) -> str:
    """Return a deterministic SHA-256 binding metadata and every field array."""
    require(isinstance(field, PositionNoiseResponseField), "invalid response field")
    array_names = (
        "sample_times_s",
        "input_declared_scales",
        "input_normalization_scales",
        "availability_count",
        "all_eligible_count",
        "adequacy",
        "signed_response_m_per_declared_scale",
        "response_magnitude_m_per_declared_scale",
        "matched_absolute_rms_scatter_m",
        "all_eligible_absolute_rms_scatter_m",
    )
    arrays = {
        name: _array_sha256(np.asarray(getattr(field, name))) for name in array_names
    }
    payload = json.dumps(
        {"metadata": _field_metadata(field), "arrays": arrays},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = ["input_contract_fingerprint", "response_field_fingerprint"]
