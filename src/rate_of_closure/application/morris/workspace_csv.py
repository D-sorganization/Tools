"""Deterministic aggregate-only CSV export for archived Morris evidence."""

from __future__ import annotations

import csv
import io
import json

from .workspace_types import MorrisWorkspace

CSV_FIELDS = (
    "evidence_state",
    "export_scope",
    "request_id",
    "job_id",
    "source_spec_id",
    "source_variable_key",
    "source_unit",
    "source_lower",
    "source_upper",
    "source_time_window_start_s",
    "source_time_window_end_s",
    "source_point_ids",
    "target_name",
    "target_unit",
    "target_kind",
    "target_time_s",
    "target_point_id",
    "target_coordinate_frame",
    "mu",
    "mu_star",
    "mu_star_standard_error",
    "sigma",
    "availability",
    "sample_adequacy",
    "denominator_total_pairs",
    "denominator_valid_pairs",
    "denominator_typed_no_impact_pairs",
    "denominator_no_impact_unavailable_pairs",
    "denominator_failed_pairs",
    "denominator_nonfinite_pairs",
    "design_trajectories",
    "design_levels",
    "design_seed",
    "design_total_samples",
    "design_normalized_step",
    "request_minimum_effects",
    "request_worker_count",
    "report_assumptions_json",
    "report_interaction_caveat",
)


def _value(value: object | None) -> object:
    if value is None:
        return ""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{value}"
    return value


def _row(workspace: MorrisWorkspace, estimate: object) -> dict[str, object]:
    evidence = workspace.completed_evidence
    assert evidence is not None and evidence.job.report is not None
    report = evidence.job.report
    source = estimate.source  # type: ignore[attr-defined]
    target = estimate.target  # type: ignore[attr-defined]
    effects = estimate.effects  # type: ignore[attr-defined]
    denominator = estimate.denominator  # type: ignore[attr-defined]
    window = source.time_window_s
    return {
        "evidence_state": "archived-completed-unverified-live",
        "export_scope": workspace.setup.export_scope,
        "request_id": evidence.request.request_id,
        "job_id": evidence.job.job_id,
        "source_spec_id": source.spec_id,
        "source_variable_key": source.variable_key,
        "source_unit": source.unit,
        "source_lower": source.bounds[0],
        "source_upper": source.bounds[1],
        "source_time_window_start_s": _value(None if window is None else window[0]),
        "source_time_window_end_s": _value(None if window is None else window[1]),
        "source_point_ids": json.dumps(list(source.point_ids), separators=(",", ":")),
        "target_name": target.name,
        "target_unit": target.unit,
        "target_kind": target.kind,
        "target_time_s": _value(target.time_s),
        "target_point_id": _value(target.point_id),
        "target_coordinate_frame": _value(target.coordinate_frame),
        "mu": _value(effects.mu),
        "mu_star": _value(effects.mu_star),
        "mu_star_standard_error": _value(effects.mu_star_standard_error),
        "sigma": _value(effects.sigma),
        "availability": estimate.availability,  # type: ignore[attr-defined]
        "sample_adequacy": estimate.sample_adequacy,  # type: ignore[attr-defined]
        "denominator_total_pairs": denominator.total_pairs,
        "denominator_valid_pairs": denominator.valid_pairs,
        "denominator_typed_no_impact_pairs": denominator.typed_no_impact_pairs,
        "denominator_no_impact_unavailable_pairs": (
            denominator.no_impact_unavailable_pairs
        ),
        "denominator_failed_pairs": denominator.failed_pairs,
        "denominator_nonfinite_pairs": denominator.nonfinite_pairs,
        "design_trajectories": report.trajectories,
        "design_levels": report.levels,
        "design_seed": report.seed,
        "design_total_samples": report.total_samples,
        "design_normalized_step": report.normalized_step,
        "request_minimum_effects": evidence.request.minimum_effects,
        "request_worker_count": evidence.request.worker_count,
        "report_assumptions_json": json.dumps(
            list(report.assumptions), separators=(",", ":")
        ),
        "report_interaction_caveat": report.interaction_caveat,
    }


def morris_report_csv(workspace: MorrisWorkspace) -> str:
    """Export report aggregates and provenance; raw samples are not retained."""
    evidence = workspace.completed_evidence
    if evidence is None or evidence.job.report is None:
        raise ValueError("Morris CSV export requires completed evidence")
    source_order = {
        factor.spec_id: index for index, factor in enumerate(evidence.request.factors)
    }
    estimates = sorted(
        evidence.job.report.estimates,
        key=lambda item: (
            source_order[item.source.spec_id],
            item.target.name,
            item.target.kind,
        ),
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {key: _value(value) for key, value in _row(workspace, estimate).items()}
        for estimate in estimates
    )
    return stream.getvalue()


__all__ = ["CSV_FIELDS", "morris_report_csv"]
