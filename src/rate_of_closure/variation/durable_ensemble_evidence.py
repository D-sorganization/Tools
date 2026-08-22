"""Strict path-free evidence wire for durable ensemble client surfaces."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

from shared.python.contracts import require

from .simulation_types import ALL_OUTPUT_NAMES, TrialEvaluationStatus
from .streaming_ensemble_analysis import (
    DurableEnsembleLayout,
    DurableEnsembleSummary,
    StreamingOutputMoments,
)

DURABLE_ENSEMBLE_EVIDENCE_SCHEMA = "rate/durable-ensemble-evidence/v1"
DURABLE_ENSEMBLE_ANALYSIS_METHOD = "incremental-welford-sample-moments/v1"
DURABLE_ENSEMBLE_LIMITATIONS = (
    "Model-scenario output is not human evidence or a coaching recommendation.",
    "Incremental moments do not retain quantiles, correlations, or trial rows.",
    "An in-progress archive describes only its verified contiguous prefix.",
)
_STATUS_NAMES = tuple(item.value for item in TrialEvaluationStatus)
_ROOT_FIELDS = {
    "schema_version",
    "archive",
    "analysis",
    "status_counts",
    "failure_type_counts",
    "output_moments",
    "limitations",
}


@dataclass(frozen=True, slots=True)
class DurableArchiveEvidence:
    """Filesystem-neutral identity and lifecycle of one verified archive."""

    header_sha256: str
    status: Literal["in_progress", "complete"]
    trial_count: int
    analyzed_trial_count: int
    failed_count: int
    chunk_count: int
    elapsed_s: float | None

    def __post_init__(self) -> None:
        require(_is_sha256(self.header_sha256), "header_sha256 is invalid")
        require(self.status in {"in_progress", "complete"}, "status is invalid")
        for name, value in (
            ("trial_count", self.trial_count),
            ("analyzed_trial_count", self.analyzed_trial_count),
            ("failed_count", self.failed_count),
            ("chunk_count", self.chunk_count),
        ):
            require(type(value) is int and value >= 0, f"{name} is invalid")
        require(self.trial_count >= 1, "trial_count must be positive")
        require(self.analyzed_trial_count <= self.trial_count, "prefix is invalid")
        require(self.failed_count <= self.analyzed_trial_count, "failures are invalid")
        require(
            self.chunk_count <= self.analyzed_trial_count,
            "chunk_count exceeds analyzed trials",
        )
        require(
            self.elapsed_s is None
            or (math.isfinite(self.elapsed_s) and self.elapsed_s >= 0.0),
            "elapsed_s is invalid",
        )
        require(
            (self.status == "complete") == (self.elapsed_s is not None),
            "elapsed_s availability does not match status",
        )
        require(
            self.status != "complete" or self.analyzed_trial_count == self.trial_count,
            "complete archive is partial",
        )


@dataclass(frozen=True, slots=True)
class DurableAnalysisEvidence:
    """Declared incremental method and source trace layout."""

    method_id: str
    sample_count: int
    point_ids: tuple[str, ...]
    coordinate_frame: str

    def __post_init__(self) -> None:
        require(
            self.method_id == DURABLE_ENSEMBLE_ANALYSIS_METHOD,
            "analysis method is unsupported",
        )
        DurableEnsembleLayout(self.sample_count, self.point_ids, self.coordinate_frame)


@dataclass(frozen=True, slots=True)
class DurableEnsembleEvidence:
    """Client-safe scalar result tied to one verified durable prefix."""

    schema_version: str
    archive: DurableArchiveEvidence
    analysis: DurableAnalysisEvidence
    status_counts: Mapping[str, int]
    failure_type_counts: Mapping[str, int]
    output_moments: tuple[StreamingOutputMoments, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        require(
            self.schema_version == DURABLE_ENSEMBLE_EVIDENCE_SCHEMA,
            "evidence schema is unsupported",
        )
        statuses = dict(self.status_counts)
        failures = dict(self.failure_type_counts)
        require(tuple(statuses) == _STATUS_NAMES, "status count keys are invalid")
        require(
            all(type(value) is int and value >= 0 for value in statuses.values()),
            "status counts are invalid",
        )
        require(
            sum(statuses.values()) == self.archive.analyzed_trial_count,
            "status counts do not match analyzed prefix",
        )
        require(
            statuses["numerical_failure"] == self.archive.failed_count,
            "failure count does not match status counts",
        )
        _require_failure_counts(failures, self.archive.failed_count)
        require(
            tuple(item.name for item in self.output_moments) == ALL_OUTPUT_NAMES,
            "output moments are not canonical",
        )
        require(
            all(
                item.available_count <= self.archive.analyzed_trial_count
                for item in self.output_moments
            ),
            "output availability exceeds analyzed prefix",
        )
        require(
            self.limitations == DURABLE_ENSEMBLE_LIMITATIONS,
            "limitations do not match the registered boundary",
        )
        object.__setattr__(self, "status_counts", MappingProxyType(statuses))
        object.__setattr__(self, "failure_type_counts", MappingProxyType(failures))


def durable_ensemble_evidence(
    summary: DurableEnsembleSummary,
) -> DurableEnsembleEvidence:
    """Project a verified in-memory summary onto the path-free client wire."""
    require(isinstance(summary, DurableEnsembleSummary), "summary type is invalid")
    archive = summary.archive
    return DurableEnsembleEvidence(
        DURABLE_ENSEMBLE_EVIDENCE_SCHEMA,
        DurableArchiveEvidence(
            archive.header_sha256,
            archive.status,
            archive.trial_count,
            summary.analyzed_trial_count,
            archive.failed_count,
            archive.chunk_count,
            archive.elapsed_s,
        ),
        DurableAnalysisEvidence(
            DURABLE_ENSEMBLE_ANALYSIS_METHOD,
            summary.layout.sample_count,
            summary.layout.point_ids,
            summary.layout.coordinate_frame,
        ),
        summary.status_counts,
        summary.failure_type_counts,
        summary.output_moments,
        DURABLE_ENSEMBLE_LIMITATIONS,
    )


def durable_ensemble_evidence_to_json(evidence: DurableEnsembleEvidence) -> str:
    """Serialize one validated evidence value with stable field ordering."""
    require(isinstance(evidence, DurableEnsembleEvidence), "evidence type is invalid")
    return json.dumps(_evidence_document(evidence), indent=2) + "\n"


def durable_ensemble_evidence_from_json(text: str) -> DurableEnsembleEvidence:
    """Parse one bounded exact evidence document from an untrusted client source."""
    require(type(text) is str, "evidence JSON must be text")
    require(len(text.encode("utf-8")) <= 4_000_000, "evidence JSON is too large")
    try:
        value = json.loads(text)
    except (TypeError, ValueError) as exc:
        require(False, "evidence JSON is invalid", str(exc))
    root = _exact_mapping(value, _ROOT_FIELDS, "evidence")
    return DurableEnsembleEvidence(
        _text(root["schema_version"], "schema_version"),
        _parse_archive(root["archive"]),
        _parse_analysis(root["analysis"]),
        _parse_status_counts(root["status_counts"]),
        _parse_failure_counts(root["failure_type_counts"]),
        _parse_moments(root["output_moments"]),
        tuple(_text(item, "limitation") for item in _list(root["limitations"])),
    )


def _evidence_document(evidence: DurableEnsembleEvidence) -> dict[str, object]:
    archive, analysis = evidence.archive, evidence.analysis
    return {
        "schema_version": evidence.schema_version,
        "archive": {
            "header_sha256": archive.header_sha256,
            "status": archive.status,
            "trial_count": archive.trial_count,
            "analyzed_trial_count": archive.analyzed_trial_count,
            "failed_count": archive.failed_count,
            "chunk_count": archive.chunk_count,
            "elapsed_s": archive.elapsed_s,
        },
        "analysis": {
            "method_id": analysis.method_id,
            "sample_count": analysis.sample_count,
            "point_ids": list(analysis.point_ids),
            "coordinate_frame": analysis.coordinate_frame,
        },
        "status_counts": dict(evidence.status_counts),
        "failure_type_counts": dict(evidence.failure_type_counts),
        "output_moments": [
            {
                "name": item.name,
                "unit": item.unit,
                "available_count": item.available_count,
                "mean": item.mean,
                "sample_std": item.sample_std,
            }
            for item in evidence.output_moments
        ],
        "limitations": list(evidence.limitations),
    }


def _parse_archive(value: object) -> DurableArchiveEvidence:
    fields = {
        "header_sha256",
        "status",
        "trial_count",
        "analyzed_trial_count",
        "failed_count",
        "chunk_count",
        "elapsed_s",
    }
    item = _exact_mapping(value, fields, "archive")
    status = _text(item["status"], "archive status")
    require(status in {"in_progress", "complete"}, "archive status is invalid")
    elapsed = item["elapsed_s"]
    return DurableArchiveEvidence(
        _text(item["header_sha256"], "header_sha256"),
        cast(Literal["in_progress", "complete"], status),
        _integer(item["trial_count"], "trial_count"),
        _integer(item["analyzed_trial_count"], "analyzed_trial_count"),
        _integer(item["failed_count"], "failed_count"),
        _integer(item["chunk_count"], "chunk_count"),
        None if elapsed is None else _finite(elapsed, "elapsed_s"),
    )


def _parse_analysis(value: object) -> DurableAnalysisEvidence:
    item = _exact_mapping(
        value,
        {"method_id", "sample_count", "point_ids", "coordinate_frame"},
        "analysis",
    )
    points = tuple(_text(point, "point_id") for point in _list(item["point_ids"]))
    return DurableAnalysisEvidence(
        _text(item["method_id"], "method_id"),
        _integer(item["sample_count"], "sample_count"),
        points,
        _text(item["coordinate_frame"], "coordinate frame"),
    )


def _parse_status_counts(value: object) -> dict[str, int]:
    item = _exact_mapping(value, set(_STATUS_NAMES), "status counts")
    return {
        name: _integer(item[name], f"status count {name}") for name in _STATUS_NAMES
    }


def _parse_failure_counts(value: object) -> dict[str, int]:
    item = _mapping(value, "failure type counts")
    require(len(item) <= 256, "failure type counts exceed the registered bound")
    return {
        _text(name, "failure type"): _integer(count, "failure type count")
        for name, count in item.items()
    }


def _parse_moments(value: object) -> tuple[StreamingOutputMoments, ...]:
    values = _list(value)
    require(len(values) == len(ALL_OUTPUT_NAMES), "output moments are not canonical")
    result = []
    for index, raw in enumerate(values):
        item = _exact_mapping(
            raw,
            {"name", "unit", "available_count", "mean", "sample_std"},
            f"output moment {index}",
        )
        mean, sample_std = item["mean"], item["sample_std"]
        result.append(
            StreamingOutputMoments(
                _text(item["name"], "output name"),
                _text(item["unit"], "output unit"),
                _integer(item["available_count"], "available_count"),
                None if mean is None else _finite(mean, "mean"),
                None if sample_std is None else _finite(sample_std, "sample_std"),
            )
        )
    return tuple(result)


def _require_failure_counts(counts: Mapping[str, int], total: int) -> None:
    require(len(counts) <= 256, "failure type counts exceed the registered bound")
    require(
        all(
            _safe_text(name) and type(count) is int and count > 0
            for name, count in counts.items()
        ),
        "failure type counts are invalid",
    )
    require(sum(counts.values()) == total, "failure types do not cover failures")


def _exact_mapping(value: object, fields: set[str], name: str) -> dict[str, Any]:
    item = _mapping(value, name)
    require(set(item) == fields, f"{name} fields are invalid")
    return item


def _mapping(value: object, name: str) -> dict[str, Any]:
    require(type(value) is dict, f"{name} must be an object")
    return cast(dict[str, Any], value)


def _list(value: object) -> list[Any]:
    require(type(value) is list, "evidence field must be an array")
    return cast(list[Any], value)


def _safe_text(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and value == value.strip()
        and len(value) <= 256
        and not any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    )


def _text(value: object, name: str) -> str:
    require(_safe_text(value), f"{name} must be bounded nonblank text")
    return cast(str, value)


def _integer(value: object, name: str) -> int:
    require(type(value) is int and value >= 0, f"{name} must be non-negative")
    require(cast(int, value) <= 9_007_199_254_740_991, f"{name} exceeds safe range")
    return cast(int, value)


def _finite(value: object, name: str) -> float:
    require(type(value) in {int, float}, f"{name} must be numeric")
    result = float(cast(float, value))
    require(math.isfinite(result), f"{name} must be finite")
    return result


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


__all__ = [
    "DURABLE_ENSEMBLE_ANALYSIS_METHOD",
    "DURABLE_ENSEMBLE_EVIDENCE_SCHEMA",
    "DURABLE_ENSEMBLE_LIMITATIONS",
    "DurableAnalysisEvidence",
    "DurableArchiveEvidence",
    "DurableEnsembleEvidence",
    "durable_ensemble_evidence",
    "durable_ensemble_evidence_from_json",
    "durable_ensemble_evidence_to_json",
]
