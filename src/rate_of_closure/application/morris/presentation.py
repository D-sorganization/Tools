"""Pure presentation models for Morris jobs and target-scoped reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shared.python.swing_sim.ball_setup import BallSupportMode

if TYPE_CHECKING:
    from rate_of_closure.simulation.records import SimulationConfig

from .request_document import (
    CANONICAL_MORRIS_FACTOR_KEYS,
    MorrisFactorDraft,
    spec_id_for_key,
)
from .response_contract import (
    MorrisResponseEstimate,
    MorrisResponseJob,
    MorrisResponseReport,
    MorrisTarget,
)

_TERMINAL = frozenset({"completed", "cancelled", "failed"})
_TEE_KEY = CANONICAL_MORRIS_FACTOR_KEYS[-1]


@dataclass(frozen=True)
class MorrisFactorRow:
    """Registry-enriched editable factor state for either UI."""

    spec_id: str
    variable_key: str
    label: str
    unit: str
    guidance: str
    applicability: str | None
    enabled: bool
    applicable: bool
    lower: float
    upper: float
    validation_error: str | None


@dataclass(frozen=True)
class MorrisJobPresentation:
    """Display-ready job lifecycle state without transport behavior."""

    status: str
    terminal: bool
    completed_samples: int
    total_samples: int
    progress_fraction: float | None
    cancel_requested: bool
    can_cancel: bool
    can_present_results: bool
    message: str
    error_code: str | None
    error_message: str | None


@dataclass(frozen=True)
class MorrisResultRow:
    """One target-scoped factor row retaining exact diagnostics."""

    rank: int | None
    spec_id: str
    variable_key: str
    label: str
    source_unit: str
    source_lower: float
    source_upper: float
    mu: float | None
    mu_star: float | None
    mu_star_standard_error: float | None
    sigma: float | None
    availability: str
    sample_adequacy: str
    total_pairs: int
    valid_pairs: int
    typed_no_impact_pairs: int
    no_impact_unavailable_pairs: int
    failed_pairs: int
    nonfinite_pairs: int


@dataclass(frozen=True)
class MorrisReportPresentation:
    """One selected target and its stable ranked factor rows."""

    target: MorrisTargetPresentation
    rows: tuple[MorrisResultRow, ...]


@dataclass(frozen=True)
class MorrisTargetPresentation:
    """Target provenance plus its established display label."""

    name: str
    label: str
    unit: str
    kind: str
    time_s: float | None
    point_id: str | None
    coordinate_frame: str | None


def present_morris_factor_rows(
    config: SimulationConfig, drafts: tuple[MorrisFactorDraft, ...]
) -> tuple[MorrisFactorRow, ...]:
    """Enrich drafts from the registry without mutating or duplicating validation."""
    from rate_of_closure.simulation.records import SimulationConfig
    from shared.python.swing_sim.variation.spec import variable_registry

    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if not isinstance(drafts, tuple):
        raise TypeError("drafts must be a tuple")
    registry = variable_registry()
    rows: list[MorrisFactorRow] = []
    for draft in drafts:
        if not isinstance(draft, MorrisFactorDraft):
            raise TypeError("drafts must contain MorrisFactorDraft values")
        definition = registry.get(draft.variable_key)
        error: str | None = None
        applicability: str | None
        applicable = not (
            draft.variable_key == _TEE_KEY
            and config.ball_setup.support_mode is BallSupportMode.GROUND
        )
        if definition is None:
            error = "Unsupported Morris factor" if draft.enabled else None
            label = draft.variable_key
            unit = ""
            guidance = ""
            applicability = "always"
        else:
            label = definition.label
            unit = definition.unit
            guidance = definition.guidance
            applicability = (
                None
                if definition.applicability == "always"
                else definition.applicability
            )
            if draft.enabled and draft.lower >= draft.upper:
                error = "Lower bound must be less than upper bound"
            elif draft.enabled and not applicable:
                error = "Tee height requires tee support"
        rows.append(
            MorrisFactorRow(
                spec_id_for_key(draft.variable_key),
                draft.variable_key,
                label,
                unit,
                guidance,
                applicability,
                draft.enabled,
                applicable,
                draft.lower,
                draft.upper,
                error,
            )
        )
    return tuple(rows)


def present_morris_job(job: MorrisResponseJob) -> MorrisJobPresentation:
    """Project a strict response into actions and status text."""
    if not isinstance(job, MorrisResponseJob):
        raise TypeError("job must be a MorrisResponseJob")
    terminal = job.status in _TERMINAL
    fraction = job.completed_samples / job.total_samples if job.total_samples else None
    message = {
        "queued": "Morris study queued",
        "running": f"Morris study running: {job.completed_samples}/{job.total_samples}",
        "completed": "Morris study completed",
        "cancelled": "Morris study cancelled",
        "failed": "Morris study failed",
    }[job.status]
    if job.cancel_requested and not terminal:
        message = f"{message}; cancellation requested"
    return MorrisJobPresentation(
        job.status,
        terminal,
        job.completed_samples,
        job.total_samples,
        fraction,
        job.cancel_requested,
        not terminal and not job.cancel_requested,
        job.status == "completed",
        message,
        job.error_code,
        job.error_message,
    )


def _source_order(variable_key: str) -> int:
    try:
        return int(CANONICAL_MORRIS_FACTOR_KEYS.index(variable_key))
    except ValueError:
        return len(CANONICAL_MORRIS_FACTOR_KEYS)


def _result_row(estimate: MorrisResponseEstimate, rank: int | None) -> MorrisResultRow:
    from shared.python.swing_sim.variation.spec import variable_registry

    source = estimate.source
    effects = estimate.effects
    denominator = estimate.denominator
    registry = variable_registry()
    label = (
        registry[source.variable_key].label
        if source.variable_key in registry
        else source.variable_key
    )
    return MorrisResultRow(
        rank,
        source.spec_id,
        source.variable_key,
        label,
        source.unit,
        source.bounds[0],
        source.bounds[1],
        effects.mu,
        effects.mu_star,
        effects.mu_star_standard_error,
        effects.sigma,
        estimate.availability,
        estimate.sample_adequacy,
        denominator.total_pairs,
        denominator.valid_pairs,
        denominator.typed_no_impact_pairs,
        denominator.no_impact_unavailable_pairs,
        denominator.failed_pairs,
        denominator.nonfinite_pairs,
    )


def _finite_sort_key(item: MorrisResponseEstimate) -> tuple[float, int, str]:
    assert item.effects.mu_star is not None
    return (
        -item.effects.mu_star,
        _source_order(item.source.variable_key),
        item.source.spec_id,
    )


def _unavailable_sort_key(item: MorrisResponseEstimate) -> tuple[int, str]:
    return (_source_order(item.source.variable_key), item.source.spec_id)


def present_morris_report(
    report: MorrisResponseReport | None, target_name: str
) -> MorrisReportPresentation:
    """Rank one unambiguous name-selected target for legacy callers."""
    if not isinstance(report, MorrisResponseReport):
        raise TypeError("report must be a MorrisResponseReport")
    if not isinstance(target_name, str) or not target_name:
        raise ValueError("target_name must be nonempty")
    targets = {
        estimate.target
        for estimate in report.estimates
        if estimate.target.name == target_name
    }
    if not targets:
        raise ValueError("unknown Morris report target")
    if len(targets) != 1:
        raise ValueError("Morris report target name is ambiguous")
    return present_morris_target(report, next(iter(targets)))


def present_morris_target(
    report: MorrisResponseReport, target: MorrisTarget
) -> MorrisReportPresentation:
    """Rank finite effects only within one provenance-complete target."""
    from rate_of_closure.variation.plot_labels import OUTPUT_LABELS

    if not isinstance(report, MorrisResponseReport):
        raise TypeError("report must be a MorrisResponseReport")
    if not isinstance(target, MorrisTarget):
        raise TypeError("target must be a MorrisTarget")
    selected = tuple(
        estimate for estimate in report.estimates if estimate.target == target
    )
    if not selected:
        raise ValueError("unknown Morris report target")
    finite = sorted(
        (item for item in selected if item.effects.mu_star is not None),
        key=_finite_sort_key,
    )
    unavailable = sorted(
        (item for item in selected if item.effects.mu_star is None),
        key=_unavailable_sort_key,
    )
    ranked = tuple(
        _result_row(item, index) for index, item in enumerate(finite, start=1)
    )
    unranked = tuple(_result_row(item, None) for item in unavailable)
    target_view = MorrisTargetPresentation(
        target.name,
        OUTPUT_LABELS.get(target.name, target.name.replace("_", " ").title()),
        target.unit,
        target.kind,
        target.time_s,
        target.point_id,
        target.coordinate_frame,
    )
    return MorrisReportPresentation(target_view, ranked + unranked)


__all__ = [
    "MorrisFactorRow",
    "MorrisJobPresentation",
    "MorrisReportPresentation",
    "MorrisResultRow",
    "MorrisTargetPresentation",
    "present_morris_job",
    "present_morris_factor_rows",
    "present_morris_report",
    "present_morris_target",
]
