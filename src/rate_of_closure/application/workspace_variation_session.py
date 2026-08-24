"""Strict workspace selection around the canonical variation plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from rate_of_closure.variation.simulation_types import ALL_OUTPUT_NAMES
from shared.python.swing_sim.variation import VariationPlan, outputs_for_mode
from shared.python.swing_sim.variation.persisted_plan_io import (
    PersistedPlanResolution,
    persisted_plan_dumps,
    persisted_plan_loads,
)

VARIATION_WORKSPACE_SCHEMA = "rate_of_closure.variation_workspace_selection"
VARIATION_WORKSPACE_SCHEMA_VERSION = 1

_ENVELOPE_FIELDS = frozenset({"schema", "schema_version", "data"})
_DATA_FIELDS = frozenset({"analysis_execution", "selected_output_metrics"})


class LegacyVariationMigrationRequired(ValueError):
    """A legacy explorer session needs an explicit variation-state fallback."""


class VariationAnalysisExecution(str, Enum):  # noqa: UP042 - Python 3.10
    """Supported simultaneous and one-at-a-time analysis policies."""

    ALL_TOGETHER = "all_together"
    INDIVIDUAL = "individual"
    BOTH = "both"


def available_output_metrics(mode: str) -> tuple[str, ...]:
    """Return the canonical selectable metrics for one persisted plan mode."""
    return ALL_OUTPUT_NAMES if mode == "swing" else tuple(outputs_for_mode(mode))


@dataclass(frozen=True)
class VariationWorkspaceState:
    """One authored plan plus UI execution and output-selection policy."""

    plan: VariationPlan
    analysis_execution: VariationAnalysisExecution
    selected_output_metrics: tuple[str, ...]
    plan_evidence: PersistedPlanResolution | None = None

    def __post_init__(self) -> None:
        """Validate membership and normalize metric order deterministically."""
        if not isinstance(self.plan, VariationPlan):
            raise TypeError("plan must be a VariationPlan")
        if not isinstance(self.analysis_execution, VariationAnalysisExecution):
            raise TypeError("analysis_execution must be a supported execution policy")
        metrics = tuple(self.selected_output_metrics)
        if not metrics or any(not isinstance(metric, str) for metric in metrics):
            raise TypeError("selected output metrics must be non-empty strings")
        if len(set(metrics)) != len(metrics):
            raise ValueError("selected output metrics must be unique")
        available = available_output_metrics(self.plan.mode)
        unknown = set(metrics) - set(available)
        if unknown:
            raise ValueError(
                f"selected output metric is not available: {sorted(unknown)}"
            )
        evidence = self.plan_evidence
        if evidence is None:
            evidence = persisted_plan_loads(persisted_plan_dumps(self.plan))
            object.__setattr__(self, "plan_evidence", evidence)
        if (
            not isinstance(evidence, PersistedPlanResolution)
            or evidence.plan != self.plan
        ):
            raise ValueError("plan evidence must match the authored plan")
        object.__setattr__(
            self,
            "selected_output_metrics",
            tuple(metric for metric in available if metric in metrics),
        )


def _exact_mapping(
    value: object,
    expected: frozenset[str],
    context: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TypeError(f"{context} has invalid fields")
    return value


def variation_workspace_to_payload(
    state: VariationWorkspaceState,
) -> dict[str, object]:
    """Serialize selection only; the canonical plan remains at the root."""
    if not isinstance(state, VariationWorkspaceState):
        raise TypeError("state must be a VariationWorkspaceState")
    return {
        "schema": VARIATION_WORKSPACE_SCHEMA,
        "schema_version": VARIATION_WORKSPACE_SCHEMA_VERSION,
        "data": {
            "analysis_execution": state.analysis_execution.value,
            "selected_output_metrics": list(state.selected_output_metrics),
        },
    }


def variation_workspace_from_payload(
    value: object,
    plan: VariationPlan,
    plan_evidence: PersistedPlanResolution | None = None,
) -> VariationWorkspaceState:
    """Parse a strict selection against its canonical root plan."""
    envelope = _exact_mapping(value, _ENVELOPE_FIELDS, "variation workspace")
    if (
        envelope["schema"] != VARIATION_WORKSPACE_SCHEMA
        or envelope["schema_version"] != VARIATION_WORKSPACE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported variation workspace selection payload")
    data = _exact_mapping(envelope["data"], _DATA_FIELDS, "variation workspace.data")
    raw_metrics = data["selected_output_metrics"]
    if not isinstance(raw_metrics, (list, tuple)):
        raise TypeError("selected_output_metrics must be a JSON array")
    try:
        execution = VariationAnalysisExecution(data["analysis_execution"])
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported variation analysis execution") from exc
    return VariationWorkspaceState(plan, execution, tuple(raw_metrics), plan_evidence)


def migrate_legacy_variation_fallback(
    fallback: VariationWorkspaceState,
    document_plan: VariationPlan | None,
    document_evidence: PersistedPlanResolution | None = None,
) -> VariationWorkspaceState:
    """Preserve explicit live policy unless a legacy root plan conflicts."""
    if not isinstance(fallback, VariationWorkspaceState):
        raise TypeError("legacy variation fallback must be complete")
    if (
        document_plan is not None
        and document_plan.to_json_dict() != fallback.plan.to_json_dict()
    ):
        raise LegacyVariationMigrationRequired(
            "legacy workspace variation plan conflicts with the explicit fallback"
        )
    if document_plan is None:
        return fallback
    return VariationWorkspaceState(
        fallback.plan,
        fallback.analysis_execution,
        fallback.selected_output_metrics,
        document_evidence,
    )


__all__ = [
    "LegacyVariationMigrationRequired",
    "VARIATION_WORKSPACE_SCHEMA",
    "VARIATION_WORKSPACE_SCHEMA_VERSION",
    "VariationAnalysisExecution",
    "VariationWorkspaceState",
    "available_output_metrics",
    "migrate_legacy_variation_fallback",
    "variation_workspace_from_payload",
    "variation_workspace_to_payload",
]
