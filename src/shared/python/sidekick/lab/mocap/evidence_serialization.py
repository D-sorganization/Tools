"""Strict canonical JSON serialization for comparison evidence receipts."""

from __future__ import annotations

import json
from typing import Any

from .experiment_contracts import CapabilityAvailability, CapabilitySupport, ReplayMode
from .experiment_evidence import (
    COMPARISON_EVIDENCE_SCHEMA_VERSION,
    ComparisonEvidenceReceipt,
    ComparisonEvidenceRow,
    ComparisonLevel,
    ComparisonRowRequirement,
    DriveMode,
    EvidenceArtifactKind,
    EvidenceArtifactReference,
    ImplementationEvidence,
    ImplementationEvidenceKind,
)
from .experiment_replay import _canonical_json
from .replay_serialization import load_experiment_replay_bundle


def dumps_comparison_evidence_receipt(receipt: ComparisonEvidenceReceipt) -> str:
    """Serialize a validated evidence receipt to stable canonical JSON."""
    if not isinstance(receipt, ComparisonEvidenceReceipt):
        raise TypeError("receipt must be a ComparisonEvidenceReceipt")
    return _canonical_json(receipt) + "\n"


def load_comparison_evidence_receipt(text: str) -> ComparisonEvidenceReceipt:
    """Load strict evidence JSON, validating nested replay bundle integrity."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("comparison evidence must contain valid JSON") from exc
    data = _require_fields(
        payload,
        {"schema_version", "receipt_id", "comparison_level", "requirements", "rows"},
        "receipt",
    )
    if data["schema_version"] != COMPARISON_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {COMPARISON_EVIDENCE_SCHEMA_VERSION!r}"
        )
    requirements = tuple(_load_requirement(value) for value in data["requirements"])
    rows = tuple(_load_row(value) for value in data["rows"])
    return ComparisonEvidenceReceipt(
        data["receipt_id"],
        ComparisonLevel(data["comparison_level"]),
        requirements,
        rows,
        data["schema_version"],
    )


def _load_requirement(value: Any) -> ComparisonRowRequirement:
    data = _require_fields(
        value,
        {
            "row_id",
            "required",
            "replay_mode",
            "required_implementation_kinds",
            "required_artifact_kinds",
        },
        "requirement",
    )
    return ComparisonRowRequirement(
        data["row_id"],
        data["required"],
        None if data["replay_mode"] is None else ReplayMode(data["replay_mode"]),
        tuple(
            ImplementationEvidenceKind(item)
            for item in data["required_implementation_kinds"]
        ),
        tuple(EvidenceArtifactKind(item) for item in data["required_artifact_kinds"]),
    )


def _load_row(value: Any) -> ComparisonEvidenceRow:
    data = _require_fields(
        value,
        {
            "row_id",
            "package_id",
            "variant_id",
            "drive_mode",
            "replay_bundle",
            "support",
            "availability",
            "implementation_evidence",
            "artifacts",
            "reason",
        },
        "row",
    )
    bundle = data["replay_bundle"]
    replay_bundle = (
        None
        if bundle is None
        else load_experiment_replay_bundle(json.dumps(bundle, separators=(",", ":")))
    )
    implementations = tuple(
        _load_implementation(item) for item in data["implementation_evidence"]
    )
    artifacts = tuple(_load_artifact(item) for item in data["artifacts"])
    return ComparisonEvidenceRow(
        data["row_id"],
        data["package_id"],
        data["variant_id"],
        DriveMode(data["drive_mode"]),
        replay_bundle,
        CapabilitySupport(data["support"]),
        CapabilityAvailability(data["availability"]),
        implementations,
        artifacts,
        data["reason"],
    )


def _load_implementation(value: Any) -> ImplementationEvidence:
    data = _require_fields(
        value,
        {
            "kind",
            "implementation_id",
            "version",
            "sha256",
            "required",
            "support",
            "availability",
            "reason",
            "evidence_reference_id",
        },
        "implementation_evidence",
    )
    data["kind"] = ImplementationEvidenceKind(data["kind"])
    data["support"] = CapabilitySupport(data["support"])
    data["availability"] = CapabilityAvailability(data["availability"])
    return ImplementationEvidence(**data)


def _load_artifact(value: Any) -> EvidenceArtifactReference:
    data = _require_fields(value, {"kind", "reference_id", "sha256"}, "artifact")
    return EvidenceArtifactReference(
        EvidenceArtifactKind(data["kind"]),
        **{key: data[key] for key in ("reference_id", "sha256")},
    )


def _require_fields(value: Any, expected: set[str], field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"{field_name} fields differ; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return value


__all__ = ["dumps_comparison_evidence_receipt", "load_comparison_evidence_receipt"]
