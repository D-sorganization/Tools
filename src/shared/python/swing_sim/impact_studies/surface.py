"""Truthful application surface for impact studies (IA-T6, #5075).

Renders a study for user-facing surfaces: the evidence tier is always
labelled, ball launch is shown separately from sound/feel, and
unavailable acoustics raise :class:`AcousticsUnavailableError` instead of
fabricating zeros.
"""

from __future__ import annotations

from shared.python.swing_sim.impact_studies.wire import (
    EvidenceTier,
    ImpactStudyV1,
    MetricRecord,
)

_TIER_LABELS: dict[EvidenceTier, str] = {
    EvidenceTier.ILLUSTRATIVE: "illustrative",
    EvidenceTier.VERIFIED: "verified",
    EvidenceTier.MEASURED_VALIDATED: "measured/validated",
}


class AcousticsUnavailableError(LookupError):
    """Raised when acoustics are requested but no measured record exists."""

    def __init__(self, study_id: str) -> None:
        super().__init__(
            f"study {study_id!r} carries no acoustic metrics; "
            "refusing to fabricate zeros"
        )


def acoustics_available(study: ImpactStudyV1) -> bool:
    """Return ``True`` only when the study carries a non-empty acoustic section."""
    if not isinstance(study, ImpactStudyV1):
        raise TypeError("study must be ImpactStudyV1")
    return bool(study.acoustic_metrics)


def acoustic_metrics_or_raise(study: ImpactStudyV1) -> tuple[MetricRecord, ...]:
    """Return the study's acoustic records or refuse explicitly.

    Raises:
        AcousticsUnavailableError: If the study has no acoustic section.
            Callers must surface the refusal, never substitute zeros.
    """
    metrics: tuple[MetricRecord, ...] | None = study.acoustic_metrics
    if not metrics:
        raise AcousticsUnavailableError(study.study_id)
    return metrics


def _format_metrics(title: str, records: tuple[MetricRecord, ...] | None) -> str:
    if not records:
        return f"{title}: unavailable"
    items: list[str] = []
    for record in records:
        item = f"{record.name}={record.value:g} {record.unit}"
        if record.uncertainty is not None:
            item += f" ±{record.uncertainty:g}"
            if record.confidence_level is not None:
                item += (
                    f" ({record.confidence_level * 100:g}% "
                    f"{record.uncertainty_convention})"
                )
            elif (
                record.uncertainty_convention
                and record.uncertainty_convention != "one_sigma"
            ):
                item += f" ({record.uncertainty_convention})"
        items.append(item)

    return f"{title}: {', '.join(items)}"


def study_statement(study: ImpactStudyV1) -> str:
    """Render a truthful user-facing statement for ``study``.

    Ball launch and sound are rendered as separate labelled sections so a
    launch metric can never be read as a sound prediction.
    """
    if not isinstance(study, ImpactStudyV1):
        raise TypeError("study must be ImpactStudyV1")
    tier_label = _TIER_LABELS[study.model_tier]
    lines = [
        f"Impact study {study.study_id!r} — evidence level: {tier_label}",
        _format_metrics("Ball launch", study.launch_metrics),
        _format_metrics("Contact", study.contact_metrics),
        _format_metrics("Vibration", study.vibration_metrics),
    ]
    if acoustics_available(study):
        lines.append(_format_metrics("Sound (acoustic)", study.acoustic_metrics))
    else:
        lines.append(
            "Sound (acoustic): unavailable — no qualified acoustic model "
            "or measurement backs this study; no values are fabricated"
        )
    if study.invalid_cases:
        rendered = "; ".join(
            f"{case.case_id}: {case.reason}" for case in study.invalid_cases
        )
        lines.append(f"Invalid cases: {rendered}")
    return "\n".join(lines)
