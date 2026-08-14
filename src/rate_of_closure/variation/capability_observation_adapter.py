"""Adapt shared capability observations to the app scalar-ensemble contract."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal, cast

from rate_of_closure.variation.scalar_ensemble_contract import (
    SCALAR_ENSEMBLE_SCHEMA_VERSION,
    ScalarCohortDefinition,
    ScalarEnsembleDataset,
    ScalarEnsembleProvenance,
    ScalarEnsembleRow,
    ScalarEnsembleStage,
    ScalarVariableCategory,
    ScalarVariableDefinition,
    scalar_ensemble_row_id,
)
from shared.python.contracts import require
from shared.python.swing_sim.flight.capability_contract import TargetDefinition
from shared.python.swing_sim.flight.capability_observation import (
    CapabilitySampleObservation,
    CapabilitySampleParameter,
    CapabilitySampleStatus,
)
from shared.python.swing_sim.flight.result_contract import (
    FlightMetricId,
    SignRule,
    flight_metric_catalog,
)
from shared.python.swing_sim.solver.targets import TargetRegion

from .canonical_numeric_json import canonical_numeric_json

MAX_CAPABILITY_OBSERVATION_ROWS = 100_000
_ADAPTER_ID = "capability-sample-observation/scalar-ensemble/v1"
_SOURCE_SCHEMA = "capability-sample-observation/v1"

_STAGES = (
    ScalarEnsembleStage("nominal", "Nominal Parameters"),
    ScalarEnsembleStage("perturbed", "Perturbed Parameters"),
    ScalarEnsembleStage("evaluation", "Evaluator Metrics"),
    ScalarEnsembleStage("target", "Target Diagnostics"),
)
_CATEGORIES = (
    ScalarVariableCategory("parameter", "Capability Parameters"),
    ScalarVariableCategory("metric", "Evaluator Metrics"),
    ScalarVariableCategory("target", "Target Diagnostics"),
)
_COHORTS = (
    ScalarCohortDefinition("complete", "Complete"),
    ScalarCohortDefinition("no_impact", "No Impact"),
    ScalarCohortDefinition("failed", "Failed"),
)
_TARGET_VARIABLE_DECLARATIONS = (
    ("target_downrange_residual", "Target Downrange Residual", "m"),
    ("target_lateral_residual", "Target Lateral Residual", "m"),
    ("target_residual", "Target Center Miss Distance", "m"),
    ("target_signed_distance", "Target Signed Distance", "m"),
    ("target_solver_residual", "Target Solver Residual", "m"),
    ("target_contains", "Inside Target", "1"),
)
_TARGET_VARIABLES = tuple(
    ScalarVariableDefinition(key, label, unit, "target", "target")
    for key, label, unit in _TARGET_VARIABLE_DECLARATIONS
)


def _ordered_observations(
    observations: tuple[CapabilitySampleObservation, ...],
) -> tuple[CapabilitySampleObservation, ...]:
    ordered = tuple(sorted(observations, key=lambda item: item.attempt_ordinal))
    require(bool(ordered), "capability observations must be nonempty")
    attempts = tuple(item.attempt_ordinal for item in ordered)
    require(
        len(set(attempts)) == len(attempts), "attempt_ordinal values must be unique"
    )
    require(
        attempts == tuple(range(len(ordered))),
        "attempt_ordinal values must form a contiguous prefix from zero",
    )
    require(
        len({item.problem_id for item in ordered}) == 1,
        "observations must share one problem_id",
    )
    require(
        len({item.total_count for item in ordered}) == 1,
        "observations must share one total_count",
    )
    scalar_ids = set(_metric_ids())
    require(
        all(
            metric.metric_id in scalar_ids
            for item in ordered
            for metric in item.metrics
        ),
        "capability observations may contain only scalar flight metrics",
    )
    return ordered


def _parameter_declarations(
    observations: tuple[CapabilitySampleObservation, ...],
) -> tuple[CapabilitySampleParameter, ...]:
    by_club: dict[str, tuple[tuple[str, str], ...]] = {}
    declared: dict[str, CapabilitySampleParameter] = {}
    for observation in observations:
        signature = tuple(
            (item.parameter_id, item.unit) for item in observation.parameters
        )
        previous = by_club.setdefault(observation.club_id, signature)
        require(previous == signature, "parameter declarations changed within a club")
        for parameter in observation.parameters:
            prior = declared.get(parameter.parameter_id)
            require(
                prior is None or prior.unit == parameter.unit,
                "parameter declarations use conflicting units",
                parameter.parameter_id,
            )
            declared.setdefault(parameter.parameter_id, parameter)
    return tuple(declared.values())


def _metric_ids() -> tuple[FlightMetricId, ...]:
    return tuple(
        definition.metric_id
        for definition in flight_metric_catalog().definitions
        if definition.sign_rule is not SignRule.VECTOR_COMPONENTS
    )


def _variables(
    parameters: tuple[CapabilitySampleParameter, ...],
    metric_ids: tuple[FlightMetricId, ...],
) -> tuple[ScalarVariableDefinition, ...]:
    variables: list[ScalarVariableDefinition] = []
    for parameter in parameters:
        label = _ascii_parameter_label(parameter.parameter_id)
        variables.extend(
            (
                ScalarVariableDefinition(
                    f"nominal.{parameter.parameter_id}",
                    f"Nominal {label}",
                    parameter.unit,
                    "nominal",
                    "parameter",
                ),
                ScalarVariableDefinition(
                    f"perturbed.{parameter.parameter_id}",
                    f"Perturbed {label}",
                    parameter.unit,
                    "perturbed",
                    "parameter",
                ),
            )
        )
    catalog = flight_metric_catalog()
    variables.extend(
        ScalarVariableDefinition(
            f"metric.{metric_id.value}",
            catalog.definition(metric_id).label,
            catalog.definition(metric_id).unit,
            "evaluation",
            "metric",
        )
        for metric_id in metric_ids
    )
    return tuple(variables) + _TARGET_VARIABLES


def _ascii_parameter_label(parameter_id: str) -> str:
    """Build a label without Unicode or locale-dependent case conversion."""
    words = parameter_id.split("_")
    return " ".join(
        chr(ord(word[0]) - 32) + word[1:] if word and "a" <= word[0] <= "z" else word
        for word in words
    )


def _target_region(target: TargetDefinition) -> TargetRegion:
    # TargetDefinition.__post_init__ already restricts kind to these two
    # values; TargetRegion re-validates it, so the cast asserts no more
    # than the constructor it feeds.
    kind = cast(Literal["green", "fairway"], target.kind)
    return TargetRegion(
        kind,
        target.distance_m,
        target.radius_m,
        target.lateral_m,
        target.band_half_length_m,
        target.half_width_m,
    )


def _target_values(
    observation: CapabilitySampleObservation, target: TargetRegion
) -> dict[str, float]:
    metrics = {item.metric_id: item.value for item in observation.metrics}
    require(
        FlightMetricId.CARRY_DISTANCE in metrics
        and FlightMetricId.CARRY_OFFLINE in metrics,
        "complete observation requires carry and offline metrics",
    )
    carry = metrics[FlightMetricId.CARRY_DISTANCE]
    offline = metrics[FlightMetricId.CARRY_OFFLINE]
    center_carry, center_offline = target.center
    return {
        "target_downrange_residual": carry - center_carry,
        "target_lateral_residual": offline - center_offline,
        "target_residual": math.hypot(carry - center_carry, offline - center_offline),
        "target_signed_distance": target.signed_distance(carry, offline),
        "target_solver_residual": target.residual_m(carry, offline),
        "target_contains": float(target.contains(carry, offline)),
    }


def _row_values(
    observation: CapabilitySampleObservation,
    variable_keys: tuple[str, ...],
    target: TargetRegion,
) -> dict[str, float | None]:
    values: dict[str, float | None] = dict.fromkeys(variable_keys)
    for parameter in observation.parameters:
        values[f"nominal.{parameter.parameter_id}"] = parameter.nominal_value
        values[f"perturbed.{parameter.parameter_id}"] = parameter.perturbed_value
    if observation.effective_status is CapabilitySampleStatus.COMPLETE:
        for metric in observation.metrics:
            values[f"metric.{metric.metric_id.value}"] = metric.value
        values.update(_target_values(observation, target))
    return values


def _row_attributes(
    observation: CapabilitySampleObservation,
    metric_ids: tuple[FlightMetricId, ...],
) -> dict[str, str | None]:
    provenance = {item.metric_id: item.provenance for item in observation.metrics}
    attributes: dict[str, str | None] = {
        "club_id": observation.club_id,
        "attempt_ordinal": str(observation.attempt_ordinal),
        "attempted_count": str(observation.attempted_count),
        "total_count": str(observation.total_count),
        "candidate_ordinal": str(observation.candidate_ordinal),
        "club_candidate_ordinal": str(observation.club_candidate_ordinal),
        "sample_ordinal": str(observation.sample_ordinal),
        "source_status": (
            None
            if observation.source_status is None
            else observation.source_status.value
        ),
        "effective_status": observation.effective_status.value,
        "reason_code": observation.reason_code,
        "source_reason": observation.source_reason,
    }
    attributes.update(
        {
            f"metric.{metric_id.value}.provenance": provenance.get(metric_id)
            for metric_id in metric_ids
        }
    )
    return attributes


def _rows(
    observations: tuple[CapabilitySampleObservation, ...],
    variables: tuple[ScalarVariableDefinition, ...],
    metric_ids: tuple[FlightMetricId, ...],
    target: TargetRegion,
) -> tuple[ScalarEnsembleRow, ...]:
    variable_keys = tuple(item.key for item in variables)
    return tuple(
        ScalarEnsembleRow(
            scalar_ensemble_row_id(
                observation.sample_ordinal,
                f"candidate:{observation.candidate_ordinal}/club:{observation.club_id}",
            ),
            observation.sample_ordinal,
            observation.effective_status.value,
            _row_values(observation, variable_keys, target),
            f"candidate:{observation.candidate_ordinal}/club:{observation.club_id}",
            _row_attributes(observation, metric_ids),
        )
        for observation in observations
    )


def _build_dataset(
    observations: tuple[CapabilitySampleObservation, ...],
    target: TargetDefinition,
    source_provenance: str,
) -> ScalarEnsembleDataset:
    ordered = _ordered_observations(observations)
    parameters = _parameter_declarations(ordered)
    metric_ids = _metric_ids()
    variables = _variables(parameters, metric_ids)
    rows = _rows(ordered, variables, metric_ids, _target_region(target))
    return ScalarEnsembleDataset(
        SCALAR_ENSEMBLE_SCHEMA_VERSION,
        ordered[0].problem_id,
        ScalarEnsembleProvenance(_ADAPTER_ID, _SOURCE_SCHEMA, source_provenance),
        _STAGES,
        _CATEGORIES,
        variables,
        _COHORTS,
        rows,
    )


class CapabilityObservationEnsembleBuilder:
    """Bounded in-memory observation sink that rejects before overflowing."""

    def __init__(
        self,
        target: TargetDefinition,
        max_rows: int,
        source_provenance: str,
    ) -> None:
        require(
            type(max_rows) is int and 1 <= max_rows <= MAX_CAPABILITY_OBSERVATION_ROWS,
            "max_rows must be an integer within "
            f"[1, {MAX_CAPABILITY_OBSERVATION_ROWS}]",
            max_rows,
        )
        require(bool(source_provenance.strip()), "source_provenance must be nonempty")
        self._target = target
        self._max_rows = max_rows
        self._source_provenance = source_provenance
        self._observations: list[CapabilitySampleObservation] = []

    @property
    def retained_count(self) -> int:
        """Return the number of accepted rows, never greater than max_rows."""
        return len(self._observations)

    def accept(self, observation: CapabilitySampleObservation) -> None:
        """Accept one row or reject it before bounded storage is exceeded."""
        require(
            self.retained_count < self._max_rows,
            f"observation row exceeds max_rows {self._max_rows}",
        )
        require(
            isinstance(observation, CapabilitySampleObservation),
            "observation must be a CapabilitySampleObservation",
        )
        self._observations.append(observation)

    def __call__(self, observation: CapabilitySampleObservation) -> None:
        """Accept one observation when used directly as an optimizer sink."""
        self.accept(observation)

    def build(self) -> ScalarEnsembleDataset:
        """Build an immutable dataset from the accepted trace prefix."""
        return _build_dataset(
            tuple(self._observations), self._target, self._source_provenance
        )


def build_capability_observation_ensemble(
    observations: Iterable[CapabilitySampleObservation],
    target: TargetDefinition,
    max_rows: int,
    source_provenance: str,
) -> ScalarEnsembleDataset:
    """Stream observations through the bounded builder and return the dataset."""
    builder = CapabilityObservationEnsembleBuilder(target, max_rows, source_provenance)
    for observation in observations:
        builder.accept(observation)
    return builder.build()


def capability_observation_ensemble_json(dataset: ScalarEnsembleDataset) -> str:
    """Serialize one ensemble with deterministic cross-runtime numeric rounding."""
    return canonical_numeric_json(dataset.to_wire())


__all__ = [
    "CapabilityObservationEnsembleBuilder",
    "MAX_CAPABILITY_OBSERVATION_ROWS",
    "build_capability_observation_ensemble",
    "capability_observation_ensemble_json",
]
