"""Paired localized attribution production, resume, and reviewer exports."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict

import numpy as np

from shared.python.contracts import require

from .paired_attribution_record import PairedAttributionRecord
from .paired_attribution_snapshot import (
    PairedAttributionAccumulator,
    PairedAttributionSnapshot,
    snapshot_from_json,
    snapshot_to_json,
)
from .paired_attribution_types import (
    AVAILABILITY_AVAILABLE,
    AVAILABILITY_MISSING,
    AVAILABILITY_NO_IMPACT,
    AVAILABILITY_NONFINITE,
    AVAILABILITY_NUMERICAL_FAILURE,
    AVAILABILITY_UNSUPPORTED,
    INTERPRETATION_BOUNDARY,
    MAX_PAIRS,
    PAIRED_INTERVENTION_METHOD_ID,
    TRIAL_EVALUATED_NO_IMPACT,
    TRIAL_NUMERICAL_FAILURE,
    AttributionPair,
    AttributionRunContext,
    AttributionSource,
    AttributionTarget,
    PairedAttributionContract,
    PairedAttributionInput,
)

_COUNT_STATES = (
    AVAILABILITY_AVAILABLE,
    AVAILABILITY_NO_IMPACT,
    AVAILABILITY_NUMERICAL_FAILURE,
    AVAILABILITY_MISSING,
    AVAILABILITY_NONFINITE,
    AVAILABILITY_UNSUPPORTED,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _contract_payload(contract: PairedAttributionContract) -> dict[str, object]:
    return {
        "source": asdict(contract.source),
        "targets": [asdict(target) for target in contract.targets],
        "context": asdict(contract.context),
        "source_sha256": contract.source_sha256,
    }


def attribution_contract_fingerprint(contract: PairedAttributionContract) -> str:
    """Return a canonical digest over pair-independent attribution semantics."""
    require(isinstance(contract, PairedAttributionContract), "invalid contract")
    return _sha256(_contract_payload(contract))


def _pair_availability(
    pair: AttributionPair, target: AttributionTarget, index: int
) -> str:
    if TRIAL_NUMERICAL_FAILURE in (pair.baseline_status, pair.perturbed_status):
        return AVAILABILITY_NUMERICAL_FAILURE
    if target.kind != "state" and TRIAL_EVALUATED_NO_IMPACT in (
        pair.baseline_status,
        pair.perturbed_status,
    ):
        return AVAILABILITY_NO_IMPACT
    states = (pair.baseline_value_states[index], pair.perturbed_value_states[index])
    for state in (
        AVAILABILITY_UNSUPPORTED,
        AVAILABILITY_MISSING,
        AVAILABILITY_NONFINITE,
    ):
        if state in states:
            return state
    return AVAILABILITY_AVAILABLE


def _availability_matrix(field_input: PairedAttributionInput) -> np.ndarray:
    return np.asarray(
        [
            [
                _pair_availability(pair, target, target_index)
                for target_index, target in enumerate(field_input.targets)
            ]
            for pair in field_input.pairs
        ],
        dtype=str,
    )


def _response_arrays(
    baseline: np.ndarray,
    perturbed: np.ndarray,
    availability: np.ndarray,
    source_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    available = availability == AVAILABILITY_AVAILABLE
    signed = np.full(baseline.shape, np.nan)
    signed[available] = perturbed[available] - baseline[available]
    magnitude = np.abs(signed)
    local = signed / source_delta[:, np.newaxis]
    return signed, magnitude, local


def _counts(availability: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.sum(availability == state, axis=0) for state in _COUNT_STATES)


def compute_paired_attribution(
    field_input: PairedAttributionInput,
) -> PairedAttributionRecord:
    """Compute exact paired response rows without causal over-interpretation."""
    require(
        isinstance(field_input, PairedAttributionInput), "invalid attribution input"
    )
    require(bool(field_input.pairs), "attribution requires at least one pair")
    pairs = field_input.pairs
    baseline = np.stack([pair.baseline_values for pair in pairs])
    perturbed = np.stack([pair.perturbed_values for pair in pairs])
    baseline_source = np.asarray([pair.baseline_source_value for pair in pairs])
    perturbed_source = np.asarray([pair.perturbed_source_value for pair in pairs])
    availability = _availability_matrix(field_input)
    responses = _response_arrays(
        baseline,
        perturbed,
        availability,
        perturbed_source - baseline_source,
    )
    counts = _counts(availability)
    return PairedAttributionRecord(
        source=field_input.source,
        targets=field_input.targets,
        context=field_input.baseline_context,
        source_sha256=field_input.source_sha256,
        pair_ids=tuple(pair.pair_id for pair in pairs),
        baseline_trial_ids=tuple(pair.baseline_trial_id for pair in pairs),
        perturbed_trial_ids=tuple(pair.perturbed_trial_id for pair in pairs),
        baseline_statuses=tuple(pair.baseline_status for pair in pairs),
        perturbed_statuses=tuple(pair.perturbed_status for pair in pairs),
        baseline_source_values=baseline_source,
        perturbed_source_values=perturbed_source,
        baseline_values=baseline,
        perturbed_values=perturbed,
        availability=availability,
        signed_response=responses[0],
        response_magnitude=responses[1],
        local_response_per_source_unit=responses[2],
        available_count=counts[0],
        no_impact_count=counts[1],
        numerical_failure_count=counts[2],
        missing_count=counts[3],
        nonfinite_count=counts[4],
        unsupported_count=counts[5],
    )


def attribution_rows(
    record: PairedAttributionRecord,
    *,
    source_id: str | None = None,
    target_id: str | None = None,
    kind: str | None = None,
    point_id: str | None = None,
    coordinate_value: float | None = None,
) -> tuple[dict[str, object], ...]:
    """Select precomputed reviewer rows without rerunning numerical analysis."""
    require(isinstance(record, PairedAttributionRecord), "invalid record")
    if source_id is not None and source_id != record.source.source_id:
        return ()
    target_indices = [
        index
        for index, target in enumerate(record.targets)
        if (target_id is None or target.target_id == target_id)
        and (kind is None or target.kind == kind)
        and (point_id is None or target.point_id == point_id)
        and (coordinate_value is None or target.coordinate_value == coordinate_value)
    ]
    return tuple(
        _row(record, pair_index, target_index)
        for pair_index in range(len(record.pair_ids))
        for target_index in target_indices
    )


def _optional_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _row(
    record: PairedAttributionRecord, pair_index: int, target_index: int
) -> dict[str, object]:
    target = record.targets[target_index]
    return {
        "source_id": record.source.source_id,
        "source_point_id": record.source.point_id,
        "source_time_window_s": record.source.time_window_s,
        "pair_id": record.pair_ids[pair_index],
        "baseline_trial_id": record.baseline_trial_ids[pair_index],
        "perturbed_trial_id": record.perturbed_trial_ids[pair_index],
        "baseline_status": record.baseline_statuses[pair_index],
        "perturbed_status": record.perturbed_statuses[pair_index],
        "baseline_source_value": float(record.baseline_source_values[pair_index]),
        "perturbed_source_value": float(record.perturbed_source_values[pair_index]),
        "source_delta": float(
            record.perturbed_source_values[pair_index]
            - record.baseline_source_values[pair_index]
        ),
        "target_id": target.target_id,
        "target_metric_id": target.metric_id,
        "target_kind": target.kind,
        "target_point_id": target.point_id,
        "target_coordinate_value": target.coordinate_value,
        "target_unit": target.unit,
        "availability": str(record.availability[pair_index, target_index]),
        "baseline_value": _optional_number(
            record.baseline_values[pair_index, target_index]
        ),
        "perturbed_value": _optional_number(
            record.perturbed_values[pair_index, target_index]
        ),
        "signed_response": _optional_number(
            record.signed_response[pair_index, target_index]
        ),
        "response_magnitude": _optional_number(
            record.response_magnitude[pair_index, target_index]
        ),
        "local_response_per_source_unit": _optional_number(
            record.local_response_per_source_unit[pair_index, target_index]
        ),
        "method_id": record.method_id,
    }


def attribution_record_fingerprint(record: PairedAttributionRecord) -> str:
    """Return a deterministic digest over semantics, counts, and exact rows."""
    payload = {
        "schema_id": record.schema_id,
        "schema_version": record.schema_version,
        "method_id": record.method_id,
        "interpretation_boundary": record.interpretation_boundary,
        "contract": _contract_payload(
            PairedAttributionContract(
                record.source, record.targets, record.context, record.source_sha256
            )
        ),
        "counts": [np.asarray(values).tolist() for values in _record_counts(record)],
        "rows": attribution_rows(record),
    }
    return _sha256(payload)


def _record_counts(record: PairedAttributionRecord) -> tuple[np.ndarray, ...]:
    return (
        record.available_count,
        record.no_impact_count,
        record.numerical_failure_count,
        record.missing_count,
        record.nonfinite_count,
        record.unsupported_count,
    )


def _csv_value(value: object) -> object:
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, tuple):
        return _canonical_json(list(value))
    return "" if value is None else value


def attribution_csv(record: PairedAttributionRecord, **selectors: object) -> str:
    """Export selected exact reviewer rows as spreadsheet-safe CSV text."""
    rows = attribution_rows(record, **selectors)  # type: ignore[arg-type]
    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {key: _csv_value(value) for key, value in row.items()} for row in rows
    )
    return output.getvalue()


__all__ = [
    "AttributionPair",
    "AttributionRunContext",
    "AttributionSource",
    "AttributionTarget",
    "INTERPRETATION_BOUNDARY",
    "MAX_PAIRS",
    "PAIRED_INTERVENTION_METHOD_ID",
    "PairedAttributionAccumulator",
    "PairedAttributionInput",
    "PairedAttributionRecord",
    "PairedAttributionSnapshot",
    "attribution_contract_fingerprint",
    "attribution_csv",
    "attribution_record_fingerprint",
    "attribution_rows",
    "compute_paired_attribution",
    "snapshot_from_json",
    "snapshot_to_json",
] + [name for name in globals() if name.startswith("AVAILABILITY_")]
