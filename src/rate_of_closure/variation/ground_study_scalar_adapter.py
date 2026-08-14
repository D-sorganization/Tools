"""Adapt explicit ground-study samples to the shared scalar-ensemble contract."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

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
from shared.python.swing_sim.ground.study_record import GroundStudyProjection
from shared.python.swing_sim.ground.study_types import (
    GROUND_STUDY_SCHEMA_VERSION,
    GroundTargetEvaluation,
)
from shared.python.swing_sim.solver.spatial_targets import (
    SurfaceCircleTolerance,
    SurfaceCorridorTolerance,
)

GROUND_STUDY_SCALAR_ADAPTER_ID = "ground-study/scalar-ensemble/v1"
MAX_GROUND_STUDY_SCALAR_ROWS = 10_000

GROUND_STUDY_STAGES = (
    ScalarEnsembleStage("model", "Model Evidence"),
    ScalarEnsembleStage("motion", "Ground Motion"),
    ScalarEnsembleStage("target", "Target Evaluation"),
)
GROUND_STUDY_CATEGORIES = (
    ScalarVariableCategory("evidence", "Qualification Evidence"),
    ScalarVariableCategory("distance", "Distance and Position"),
    ScalarVariableCategory("event", "Event Count"),
    ScalarVariableCategory("target", "Target Result"),
)
GROUND_STUDY_COHORTS = (
    ScalarCohortDefinition("complete", "Complete"),
    ScalarCohortDefinition("censored", "Censored"),
    ScalarCohortDefinition("failed", "Failed"),
    ScalarCohortDefinition("unavailable", "Unavailable"),
)

_VARIABLE_ROWS = (
    "calibration_confidence|Model Calibration Confidence|1|model|evidence",
    "carry_distance_m|Carry Distance|m|motion|distance",
    "bounce_air_distance_m|Post-Contact Air Distance|m|motion|distance",
    "skid_distance_m|Skid Distance|m|motion|distance",
    "roll_distance_m|Roll Distance|m|motion|distance",
    "surface_path_distance_m|Surface Path Distance|m|motion|distance",
    "total_distance_m|Total Distance|m|motion|distance",
    "final_downrange_m|Final Downrange|m|motion|distance",
    "final_offline_m|Final Offline|m|motion|distance",
    "bounce_count|Bounce Count|count|motion|event",
    "ground_elapsed_s|Ground Elapsed Time|s|motion|event",
    "first_position_x_m|First Contact Downrange|m|motion|distance",
    "first_position_y_m|First Contact Up|m|motion|distance",
    "first_position_z_m|First Contact Right|m|motion|distance",
    "final_position_x_m|Final Observed Downrange|m|motion|distance",
    "final_position_y_m|Final Observed Up|m|motion|distance",
    "final_position_z_m|Final Observed Right|m|motion|distance",
    "first_target_miss_distance_m|First Contact Target Miss|m|target|target",
    "first_target_accepted|First Contact Target Accepted|1|target|target",
    "final_target_miss_distance_m|Final Target Miss|m|target|target",
    "final_target_accepted|Final Target Accepted|1|target|target",
)


def _variables() -> tuple[ScalarVariableDefinition, ...]:
    return tuple(ScalarVariableDefinition(*row.split("|")) for row in _VARIABLE_ROWS)


GROUND_STUDY_VARIABLES = _variables()
_VARIABLE_KEYS = tuple(item.key for item in GROUND_STUDY_VARIABLES)


@dataclass(frozen=True)
class GroundStudySample:
    """One study with caller-supplied series and trial identity."""

    series_id: str
    trial_index: int
    study: GroundStudyProjection

    def __post_init__(self) -> None:
        scalar_ensemble_row_id(self.trial_index, self.series_id)
        require(
            type(self.study) is GroundStudyProjection,
            "study must use the exact GroundStudyProjection type",
        )


def _target_values(
    values: dict[str, float | None],
    prefix: str,
    evaluation: GroundTargetEvaluation | None,
) -> None:
    if evaluation is None:
        return
    values[f"{prefix}_target_miss_distance_m"] = evaluation.miss.distance_m
    values[f"{prefix}_target_accepted"] = float(evaluation.accepted)


def _values(study: GroundStudyProjection) -> dict[str, float | None]:
    values: dict[str, float | None] = dict.fromkeys(_VARIABLE_KEYS)
    metrics = study.metrics
    if metrics is None:
        return values
    summary = metrics.summary
    values.update(
        {
            "calibration_confidence": study.calibration.confidence,
            "carry_distance_m": summary.carry_distance_m,
            "bounce_air_distance_m": summary.bounce_air_distance_m,
            "skid_distance_m": summary.skid_distance_m,
            "roll_distance_m": summary.roll_distance_m,
            "surface_path_distance_m": summary.surface_path_distance_m,
            "total_distance_m": summary.total_distance_m,
            "final_downrange_m": summary.final_downrange_m,
            "final_offline_m": summary.final_offline_m,
            "bounce_count": float(summary.bounce_count),
            "ground_elapsed_s": metrics.ground_elapsed_s,
        }
    )
    for prefix, position in (
        ("first", metrics.first_contact_position_m),
        ("final", metrics.final_observed_position_m),
    ):
        for axis, value in zip("xyz", position, strict=True):
            values[f"{prefix}_position_{axis}_m"] = value
    _target_values(values, "first", study.first_contact_target)
    _target_values(values, "final", study.final_target)
    return values


def _profile_attributes(study: GroundStudyProjection) -> dict[str, str | None]:
    profile = study.profile
    condition = None if profile is None else profile.operating_condition
    return {
        "profile_id": None if profile is None else profile.profile_id,
        "profile_revision": None if profile is None else profile.profile_revision,
        "profile_sha256": None if profile is None else profile.profile_sha256,
        "profile_qualification_status": (
            None if profile is None else profile.qualification_status.value
        ),
        "profile_model_use_status": (
            None if profile is None else profile.model_use_status.value
        ),
        "operating_surface_class": (
            None if condition is None else condition.surface_class
        ),
        "operating_temperature_k": (
            None if condition is None else str(condition.temperature_k)
        ),
        "operating_moisture_fraction": (
            None if condition is None else str(condition.moisture_fraction)
        ),
    }


def _profile_warning_attributes(study: GroundStudyProjection) -> dict[str, str]:
    if study.profile is None:
        return {}
    return {
        f"profile_warning.{index}": warning
        for index, warning in enumerate(study.profile.warnings)
    }


def _evidence_attributes(study: GroundStudyProjection) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for index, warning in enumerate(study.warnings):
        prefix = f"warning.{index}"
        attributes[f"{prefix}.code"] = warning.code
        attributes[f"{prefix}.severity"] = warning.severity.value
        attributes[f"{prefix}.message"] = warning.message
    for index, unavailable in enumerate(study.unavailable_fields):
        prefix = f"unavailable.{index}"
        attributes[f"{prefix}.field_id"] = unavailable.field_id.value
        attributes[f"{prefix}.reason"] = unavailable.reason.value
        attributes[f"{prefix}.provenance"] = unavailable.provenance
    return attributes


def _target_attributes(study: GroundStudyProjection) -> dict[str, str | None]:
    target = study.target
    if target is None:
        return {
            "target_label": None,
            "target_kind": None,
            "target_geometry": None,
        }
    tolerance = target.tolerance
    if isinstance(tolerance, SurfaceCircleTolerance):
        geometry = "surface_circle"
        dimensions = {"target_radius_m": str(tolerance.radius_m)}
    else:
        require(
            isinstance(tolerance, SurfaceCorridorTolerance),
            "ground target must use a surface tolerance",
        )
        corridor = cast(SurfaceCorridorTolerance, tolerance)
        geometry = "surface_corridor"
        dimensions = {
            "target_half_length_m": str(corridor.half_length_m),
            "target_half_width_m": str(corridor.half_width_m),
        }
    attributes: dict[str, str | None] = {
        "target_label": target.label,
        "target_kind": target.kind,
        "target_geometry": geometry,
        "target_center_x_m": str(target.point.x_m),
        "target_center_y_m": str(target.point.elevation_m),
        "target_center_z_m": str(target.point.right_m),
        "target_point_source_frame": target.point.source_frame,
        "target_elevation_source": target.elevation_source,
        "target_ground_source": target.ground_source,
        "target_frame": target.frame,
        "target_units": target.units,
    }
    attributes.update(dimensions)
    return attributes


def _attributes(study: GroundStudyProjection) -> dict[str, str | None]:
    attributes: dict[str, str | None] = {
        "request_id": study.request_id,
        "study_status": study.status.value,
        "result_status": study.result_status.value,
        "termination_reason": study.termination_reason.value,
        "solver_eligible": str(study.solver_eligibility.eligible).lower(),
        "solver_eligibility_reasons": ",".join(
            item.value for item in study.solver_eligibility.reasons
        ),
        "model_id": study.model_id,
        "model_version": study.model_version,
        "surface_id": study.surface_id,
        "frame": study.frame.value,
        "request_context_sha256": study.request_sha256,
        "result_sha256": study.result_sha256,
        "study_sha256": hashlib.sha256(study.to_json().encode("utf-8")).hexdigest(),
        "calibration_id": study.calibration.calibration_id,
        "calibration_kind": study.calibration.kind.value,
        "calibration_source": study.calibration.source,
        "calibration_confidence": str(study.calibration.confidence),
        "producer": study.provenance.producer,
        "producer_version": study.provenance.producer_version,
        "source_revision": study.provenance.source_revision,
        "source_input_sha256": study.provenance.input_sha256,
        "final_target_unavailable_reason": (
            None
            if study.final_target_unavailable_reason is None
            else study.final_target_unavailable_reason.value
        ),
    }
    attributes.update(_profile_attributes(study))
    attributes.update(_profile_warning_attributes(study))
    attributes.update(_evidence_attributes(study))
    attributes.update(_target_attributes(study))
    return attributes


def _row(sample: GroundStudySample) -> ScalarEnsembleRow:
    study = sample.study
    return ScalarEnsembleRow(
        scalar_ensemble_row_id(sample.trial_index, sample.series_id),
        sample.trial_index,
        study.status.value,
        _values(study),
        sample.series_id,
        _attributes(study),
    )


def _bounded_samples(
    samples: Iterable[GroundStudySample],
    max_rows: int,
) -> tuple[GroundStudySample, ...]:
    retained: list[GroundStudySample] = []
    for sample in samples:
        require(
            len(retained) < max_rows,
            f"sample row exceeds max_rows {max_rows}",
        )
        require(
            type(sample) is GroundStudySample,
            "samples must use exact GroundStudySample records",
        )
        retained.append(sample)
    require(bool(retained), "samples must be nonempty")
    return tuple(retained)


def build_ground_study_scalar_dataset(
    samples: Iterable[GroundStudySample],
    source_provenance: str,
    *,
    result_id: str | None = None,
    max_rows: int = MAX_GROUND_STUDY_SCALAR_ROWS,
) -> ScalarEnsembleDataset:
    """Build a bounded deterministic dataset without inferring trial identity."""
    require(
        type(max_rows) is int and 1 <= max_rows <= MAX_GROUND_STUDY_SCALAR_ROWS,
        f"max_rows must be within [1, {MAX_GROUND_STUDY_SCALAR_ROWS}]",
    )
    require(
        isinstance(source_provenance, str) and bool(source_provenance.strip()),
        "source_provenance must be nonempty",
    )
    require(isinstance(samples, Iterable), "samples must be iterable")
    if result_id is not None:
        require(
            isinstance(result_id, str) and bool(result_id.strip()),
            "result_id must be nonempty when provided",
        )
    retained = _bounded_samples(samples, max_rows)
    rows = tuple(
        _row(sample)
        for sample in sorted(
            retained, key=lambda item: (item.series_id, item.trial_index)
        )
    )
    require(
        len({row.row_id for row in rows}) == len(rows),
        "sample identities must be unique",
    )
    return ScalarEnsembleDataset(
        SCALAR_ENSEMBLE_SCHEMA_VERSION,
        result_id or f"ground-study:{source_provenance}",
        ScalarEnsembleProvenance(
            GROUND_STUDY_SCALAR_ADAPTER_ID,
            GROUND_STUDY_SCHEMA_VERSION,
            source_provenance,
        ),
        GROUND_STUDY_STAGES,
        GROUND_STUDY_CATEGORIES,
        GROUND_STUDY_VARIABLES,
        GROUND_STUDY_COHORTS,
        rows,
    )


__all__ = [
    "GROUND_STUDY_CATEGORIES",
    "GROUND_STUDY_COHORTS",
    "GROUND_STUDY_SCALAR_ADAPTER_ID",
    "GROUND_STUDY_STAGES",
    "GROUND_STUDY_VARIABLES",
    "MAX_GROUND_STUDY_SCALAR_ROWS",
    "GroundStudySample",
    "build_ground_study_scalar_dataset",
]
