"""Deterministic JSON serialization for validated Morris workspaces."""

from __future__ import annotations

import json
from typing import Any

from rate_of_closure.application._workspace_validation import thaw_json

from ._response_types import MorrisResponseJob, MorrisResponseReport
from .workspace_types import MorrisWorkspace
from .workspace_validation import workspace_factor_dict

MAX_WORKSPACE_BYTES = 2_000_000
MAX_WORKSPACE_DEPTH = 32
MAX_WORKSPACE_NODES = 25_000


def _source_dict(source: object) -> dict[str, object]:
    return {
        "spec_id": source.spec_id,  # type: ignore[attr-defined]
        "variable_key": source.variable_key,  # type: ignore[attr-defined]
        "unit": source.unit,  # type: ignore[attr-defined]
        "bounds": list(source.bounds),  # type: ignore[attr-defined]
        "time_window_s": (
            None if source.time_window_s is None else list(source.time_window_s)  # type: ignore[attr-defined]
        ),
        "point_ids": list(source.point_ids),  # type: ignore[attr-defined]
    }


def _target_dict(target: object) -> dict[str, object]:
    return {
        "name": target.name,  # type: ignore[attr-defined]
        "unit": target.unit,  # type: ignore[attr-defined]
        "kind": target.kind,  # type: ignore[attr-defined]
        "time_s": target.time_s,  # type: ignore[attr-defined]
        "point_id": target.point_id,  # type: ignore[attr-defined]
        "coordinate_frame": target.coordinate_frame,  # type: ignore[attr-defined]
    }


def _report_dict(report: MorrisResponseReport) -> dict[str, object]:
    estimates = []
    for estimate in report.estimates:
        estimates.append(
            {
                "source": _source_dict(estimate.source),
                "target": _target_dict(estimate.target),
                "effects": {
                    "mu": estimate.effects.mu,
                    "mu_star": estimate.effects.mu_star,
                    "mu_star_standard_error": estimate.effects.mu_star_standard_error,
                    "sigma": estimate.effects.sigma,
                },
                "availability": estimate.availability,
                "sample_adequacy": estimate.sample_adequacy,
                "denominator": {
                    "total_pairs": estimate.denominator.total_pairs,
                    "valid_pairs": estimate.denominator.valid_pairs,
                    "typed_no_impact_pairs": estimate.denominator.typed_no_impact_pairs,
                    "no_impact_unavailable_pairs": (
                        estimate.denominator.no_impact_unavailable_pairs
                    ),
                    "failed_pairs": estimate.denominator.failed_pairs,
                    "nonfinite_pairs": estimate.denominator.nonfinite_pairs,
                },
            }
        )
    return {
        "schema_id": "swing-sim/morris-global-sensitivity-report",
        "schema_version": 1,
        "method": "morris-elementary-effects",
        "design": {
            "trajectories": report.trajectories,
            "levels": report.levels,
            "seed": report.seed,
            "total_samples": report.total_samples,
            "normalized_step": report.normalized_step,
        },
        "assumptions": list(report.assumptions),
        "interaction_caveat": report.interaction_caveat,
        "estimates": estimates,
    }


def _job_dict(job: MorrisResponseJob) -> dict[str, object]:
    error = None
    if job.error_code is not None:
        error = {"code": job.error_code, "message": job.error_message}
    return {
        "schema_id": "rate-of-closure/morris-job",
        "schema_version": 1,
        "job_id": job.job_id,
        "request_id": job.request_id,
        "status": job.status,
        "completed_samples": job.completed_samples,
        "total_samples": job.total_samples,
        "cancel_requested": job.cancel_requested,
        "report": None if job.report is None else _report_dict(job.report),
        "error": error,
    }


def morris_workspace_dict(workspace: MorrisWorkspace) -> dict[str, object]:
    """Return the exact detached v1 document for a validated workspace."""
    setup = workspace.setup
    evidence: dict[str, object] | None = None
    if workspace.completed_evidence is not None:
        evidence = {
            "request": workspace.completed_evidence.request.to_json_dict(),
            "job": _job_dict(workspace.completed_evidence.job),
        }
    return {
        "schema_id": workspace.schema_id,
        "schema_version": workspace.schema_version,
        "setup": {
            "export_scope": setup.export_scope,
            "base": thaw_json(setup.base),
            "factor_drafts": [
                workspace_factor_dict(draft) for draft in setup.factor_drafts
            ],
            "trajectories": setup.trajectories,
            "levels": setup.levels,
            "seed": setup.seed,
            "minimum_effects": setup.minimum_effects,
            "worker_count": setup.worker_count,
        },
        "completed_evidence": evidence,
    }


def dumps_morris_workspace(workspace: MorrisWorkspace) -> str:
    """Serialize deterministically; output contains no ambient identity or paths."""
    from .workspace_validation import parse_morris_workspace

    workspace = parse_morris_workspace(morris_workspace_dict(workspace))
    return (
        json.dumps(
            morris_workspace_dict(workspace),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _validate_shape(
    value: object, level: int = 0, nodes: list[int] | None = None
) -> None:
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > MAX_WORKSPACE_NODES:
        raise ValueError("Morris workspace exceeds the decoded node limit")
    if level > MAX_WORKSPACE_DEPTH:
        raise ValueError("Morris workspace nesting exceeds the depth limit")
    if isinstance(value, dict):
        for child in value.values():
            _validate_shape(child, level + 1, nodes)
    elif isinstance(value, list):
        for child in value:
            _validate_shape(child, level + 1, nodes)


def loads_json_document(text: str) -> object:
    """Decode bounded strict JSON while rejecting duplicate keys and constants."""
    if not isinstance(text, str):
        raise TypeError("Morris workspace text must be a string")
    if len(text.encode("utf-8")) > MAX_WORKSPACE_BYTES:
        raise ValueError("Morris workspace exceeds the payload limit")
    value = json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    _validate_shape(value)
    return value


__all__ = [
    "dumps_morris_workspace",
    "loads_json_document",
    "morris_workspace_dict",
]
