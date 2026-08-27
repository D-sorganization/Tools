"""Provenance-complete selection over immutable Morris reports."""

from __future__ import annotations

from dataclasses import dataclass

from ._response_types import MorrisResponseReport, MorrisTarget
from .presentation import MorrisReportPresentation, present_morris_target

TARGET_SELECTION_SCHEMA_ID = "rate-of-closure/morris-target-selection"
TARGET_SELECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MorrisTargetIdentity:
    """Versioned identity for one exact observation target."""

    schema_id: str
    schema_version: int
    name: str
    unit: str
    kind: str
    time_s: float | None
    point_id: str | None
    coordinate_frame: str | None

    def target(self) -> MorrisTarget:
        """Return the immutable authority target encoded by this identity."""
        if (
            self.schema_id != TARGET_SELECTION_SCHEMA_ID
            or self.schema_version != TARGET_SELECTION_SCHEMA_VERSION
        ):
            raise ValueError("unsupported Morris target selection identity")
        return MorrisTarget(
            self.name,
            self.unit,
            self.kind,
            self.time_s,
            self.point_id,
            self.coordinate_frame,
        )


@dataclass(frozen=True)
class MorrisTargetOption:
    """Display label bound to one exact target identity."""

    identity: MorrisTargetIdentity
    label: str


@dataclass(frozen=True)
class MorrisSourceOption:
    """One source available for a selected target."""

    spec_id: str
    variable_key: str
    label: str


@dataclass(frozen=True)
class MorrisReportSelection:
    """Target and optional single-source projection request."""

    target: MorrisTargetIdentity
    source_spec_id: str | None


def _identity(target: MorrisTarget) -> MorrisTargetIdentity:
    return MorrisTargetIdentity(
        TARGET_SELECTION_SCHEMA_ID,
        TARGET_SELECTION_SCHEMA_VERSION,
        target.name,
        target.unit,
        target.kind,
        target.time_s,
        target.point_id,
        target.coordinate_frame,
    )


def _target_sort_key(target: MorrisTarget) -> tuple[object, ...]:
    return (
        target.kind,
        target.name,
        target.unit,
        target.point_id or "",
        target.time_s is None,
        target.time_s or 0.0,
        target.coordinate_frame or "",
    )


def _label(target: MorrisTarget) -> str:
    from rate_of_closure.variation.plot_labels import OUTPUT_LABELS

    base = OUTPUT_LABELS.get(target.name, target.name.replace("_", " ").title())
    context = [target.kind]
    if target.point_id is not None:
        context.append(target.point_id)
    if target.time_s is not None:
        context.append(f"t={target.time_s:g} s")
    return f"{base} — {' · '.join(context)}"


def list_morris_target_options(
    report: MorrisResponseReport,
) -> tuple[MorrisTargetOption, ...]:
    """Enumerate exact report targets in deterministic provenance order."""
    if not isinstance(report, MorrisResponseReport):
        raise TypeError("report must be a MorrisResponseReport")
    targets = sorted(
        {estimate.target for estimate in report.estimates}, key=_target_sort_key
    )
    return tuple(
        MorrisTargetOption(_identity(target), _label(target)) for target in targets
    )


def list_morris_source_options(
    report: MorrisResponseReport, target: MorrisTargetIdentity
) -> tuple[MorrisSourceOption, ...]:
    """Enumerate the sources present for one exact target."""
    from shared.python.swing_sim.variation.spec import variable_registry

    if not isinstance(report, MorrisResponseReport):
        raise TypeError("report must be a MorrisResponseReport")
    exact_target = target.target()
    registry = variable_registry()
    sources = {
        estimate.source
        for estimate in report.estimates
        if estimate.target == exact_target
    }
    if not sources:
        raise ValueError("unknown Morris report target")
    return tuple(
        MorrisSourceOption(
            source.spec_id,
            source.variable_key,
            registry[source.variable_key].label
            if source.variable_key in registry
            else source.variable_key,
        )
        for source in sorted(
            sources, key=lambda value: (value.variable_key, value.spec_id)
        )
    )


def select_morris_report(
    report: MorrisResponseReport, selection: MorrisReportSelection
) -> MorrisReportPresentation:
    """Project immutable results without invoking simulation or analysis."""
    if not isinstance(report, MorrisResponseReport):
        raise TypeError("report must be a MorrisResponseReport")
    if not isinstance(selection, MorrisReportSelection):
        raise TypeError("selection must be a MorrisReportSelection")
    presentation = present_morris_target(report, selection.target.target())
    if selection.source_spec_id is None:
        return presentation
    rows = tuple(
        row for row in presentation.rows if row.spec_id == selection.source_spec_id
    )
    if not rows:
        raise ValueError("unknown Morris report source for selected target")
    return MorrisReportPresentation(presentation.target, rows)


__all__ = [
    "MorrisReportSelection",
    "MorrisSourceOption",
    "MorrisTargetIdentity",
    "MorrisTargetOption",
    "TARGET_SELECTION_SCHEMA_ID",
    "TARGET_SELECTION_SCHEMA_VERSION",
    "list_morris_source_options",
    "list_morris_target_options",
    "select_morris_report",
]
