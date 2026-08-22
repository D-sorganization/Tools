"""Strict evidence contract for durable ensemble scaling measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast


class ScalingEvidenceError(ValueError):
    """Raised when a scaling report cannot support its declared conclusion."""


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ScalingEvidenceError(f"{name} must be an object with string keys")
    return cast(dict[str, object], value)


def _exact_keys(value: dict[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ScalingEvidenceError(f"{name} fields do not match the schema")


def _integer(value: object, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ScalingEvidenceError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str, minimum: float = 0.0) -> float:
    if type(value) not in {int, float}:
        raise ScalingEvidenceError(f"{name} must be a finite number")
    result = float(cast(float, value))
    if not math.isfinite(result) or result <= minimum:
        raise ScalingEvidenceError(f"{name} must be finite and > {minimum}")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScalingEvidenceError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ScalingBudgets:
    """Declared portable ceilings for the synthetic transport diagnostic."""

    max_peak_resident_bytes: int
    max_trial_axis_peak_growth_bytes: int
    max_trace_axis_peak_growth_bytes: int
    min_trials_per_second: float


@dataclass(frozen=True, slots=True)
class ScalingObservation:
    """One fresh-process measurement at an independent scale point."""

    case_id: str
    trial_count: int
    trace_sample_count: int
    point_count: int
    chunk_size: int
    elapsed_s: float
    trials_per_second: float
    peak_resident_bytes: int
    archive_bytes: int
    logical_trace_bytes: int
    passed: bool


@dataclass(frozen=True, slots=True)
class EnsembleScalingEvidence:
    """Validated report whose conclusion is recomputed from observations."""

    schema_id: str
    schema_version: int
    measurement_policy: str
    source_commit: str
    generated_utc: str
    python_version: str
    platform: str
    workload: str
    budgets: ScalingBudgets
    observations: tuple[ScalingObservation, ...]
    passed: bool


_BUDGET_KEYS = {
    "max_peak_resident_bytes",
    "max_trial_axis_peak_growth_bytes",
    "max_trace_axis_peak_growth_bytes",
    "min_trials_per_second",
}
_OBSERVATION_KEYS = {
    "case_id",
    "trial_count",
    "trace_sample_count",
    "point_count",
    "chunk_size",
    "elapsed_s",
    "trials_per_second",
    "peak_resident_bytes",
    "archive_bytes",
    "logical_trace_bytes",
    "passed",
}
_REPORT_KEYS = {
    "schema_id",
    "schema_version",
    "measurement_policy",
    "source_commit",
    "generated_utc",
    "python_version",
    "platform",
    "workload",
    "budgets",
    "observations",
    "passed",
}


def _parse_budgets(value: object) -> ScalingBudgets:
    data = _mapping(value, "budgets")
    _exact_keys(data, _BUDGET_KEYS, "budgets")
    return ScalingBudgets(
        _integer(data["max_peak_resident_bytes"], "resident budget"),
        _integer(data["max_trial_axis_peak_growth_bytes"], "trial growth budget"),
        _integer(data["max_trace_axis_peak_growth_bytes"], "trace growth budget"),
        _finite(data["min_trials_per_second"], "throughput budget"),
    )


def _parse_observation(value: object) -> ScalingObservation:
    data = _mapping(value, "observation")
    _exact_keys(data, _OBSERVATION_KEYS, "observation")
    observation = ScalingObservation(
        _text(data["case_id"], "case_id"),
        _integer(data["trial_count"], "trial_count"),
        _integer(data["trace_sample_count"], "trace_sample_count"),
        _integer(data["point_count"], "point_count"),
        _integer(data["chunk_size"], "chunk_size"),
        _finite(data["elapsed_s"], "elapsed"),
        _finite(data["trials_per_second"], "throughput"),
        _integer(data["peak_resident_bytes"], "peak resident bytes"),
        _integer(data["archive_bytes"], "archive bytes"),
        _integer(data["logical_trace_bytes"], "logical trace bytes"),
        data["passed"] is True,
    )
    expected = observation.trial_count / observation.elapsed_s
    if not math.isclose(observation.trials_per_second, expected, rel_tol=1e-5):
        raise ScalingEvidenceError("observation throughput does not reproduce")
    if data["passed"] is not True:
        raise ScalingEvidenceError("observation budget is not satisfied")
    return observation


def _axis_cases(
    observations: tuple[ScalingObservation, ...],
) -> tuple[ScalingObservation, ScalingObservation, ScalingObservation]:
    baseline = min(
        observations, key=lambda item: (item.trial_count, item.trace_sample_count)
    )
    trial = [
        item
        for item in observations
        if item.trace_sample_count == baseline.trace_sample_count
        and item.trial_count > baseline.trial_count
    ]
    trace = [
        item
        for item in observations
        if item.trial_count == baseline.trial_count
        and item.trace_sample_count > baseline.trace_sample_count
    ]
    if not trial or not trace:
        raise ScalingEvidenceError("observations lack independent scaling axes")
    return (
        baseline,
        max(trial, key=lambda item: item.trial_count),
        max(trace, key=lambda item: item.trace_sample_count),
    )


def _require_budgets(
    budgets: ScalingBudgets, observations: tuple[ScalingObservation, ...]
) -> None:
    baseline, trial, trace = _axis_cases(observations)
    if any(
        item.peak_resident_bytes > budgets.max_peak_resident_bytes
        for item in observations
    ):
        raise ScalingEvidenceError("peak resident budget is not satisfied")
    trial_growth = max(0, trial.peak_resident_bytes - baseline.peak_resident_bytes)
    trace_growth = max(0, trace.peak_resident_bytes - baseline.peak_resident_bytes)
    if trial_growth > budgets.max_trial_axis_peak_growth_bytes:
        raise ScalingEvidenceError("trial-axis resident growth budget is not satisfied")
    if trace_growth > budgets.max_trace_axis_peak_growth_bytes:
        raise ScalingEvidenceError("trace-axis resident growth budget is not satisfied")
    if any(
        item.trials_per_second < budgets.min_trials_per_second for item in observations
    ):
        raise ScalingEvidenceError("throughput budget is not satisfied")
    if trial.archive_bytes <= baseline.archive_bytes:
        raise ScalingEvidenceError("archive bytes must grow along the trial axis")


def parse_scaling_evidence(value: object) -> EnsembleScalingEvidence:
    """Parse strict JSON-shaped evidence and recompute every pass condition."""
    data = _mapping(value, "scaling evidence")
    _exact_keys(data, _REPORT_KEYS, "scaling evidence")
    raw_observations = data["observations"]
    if not isinstance(raw_observations, list) or len(raw_observations) < 3:
        raise ScalingEvidenceError("observations must contain at least three cases")
    observations = tuple(_parse_observation(item) for item in raw_observations)
    budgets = _parse_budgets(data["budgets"])
    _require_budgets(budgets, observations)
    evidence = EnsembleScalingEvidence(
        _text(data["schema_id"], "schema_id"),
        _integer(data["schema_version"], "schema_version"),
        _text(data["measurement_policy"], "measurement_policy"),
        _text(data["source_commit"], "source_commit"),
        _text(data["generated_utc"], "generated_utc"),
        _text(data["python_version"], "python_version"),
        _text(data["platform"], "platform"),
        _text(data["workload"], "workload"),
        budgets,
        observations,
        data["passed"] is True,
    )
    if evidence.schema_id != "rate-of-closure/ensemble-stream-scaling-evidence":
        raise ScalingEvidenceError("schema_id is unsupported")
    if evidence.schema_version != 1:
        raise ScalingEvidenceError("schema_version is unsupported")
    if evidence.measurement_policy != "synthetic-failure-transport-diagnostic":
        raise ScalingEvidenceError("measurement_policy is unsupported")
    if len(evidence.source_commit) not in {40, 64}:
        raise ScalingEvidenceError("source_commit is not a Git object ID")
    if data["passed"] is not True:
        raise ScalingEvidenceError("report budget is not satisfied")
    return evidence


__all__ = [
    "EnsembleScalingEvidence",
    "ScalingBudgets",
    "ScalingEvidenceError",
    "ScalingObservation",
    "parse_scaling_evidence",
]
