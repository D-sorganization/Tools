"""Ground-study projection into the shared scalar-ensemble contract."""

import hashlib
from dataclasses import replace

import pytest

from rate_of_closure.variation.ground_study_scalar_adapter import (
    GroundStudySample,
    build_ground_study_scalar_dataset,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.ground.contract_types import (
    GroundResultStatus,
    GroundTerminationReason,
)
from shared.python.swing_sim.ground.profile_binding import (
    PROFILE_ILLUSTRATIVE_WARNING,
    ProfileOperatingCondition,
    SurfacePlacement,
    bind_material_profile,
)
from shared.python.swing_sim.ground.profile_types import GroundEvidenceKind
from shared.python.swing_sim.ground.result_types import GroundTermination
from shared.python.swing_sim.ground.study_projection import project_ground_study
from shared.python.swing_sim.ground.tests._support import (
    _failed_result,
    _request,
    _result,
)
from shared.python.swing_sim.ground.tests.test_profile_contract import _profile
from shared.python.swing_sim.ground.tests.test_study_projection import (
    _bound_surface,
    _qualified_request,
    _target,
)
from shared.python.swing_sim.ground.tests.test_study_state import (
    _partial_bounce_result,
)
from shared.python.swing_sim.ground.unavailable_types import (
    GroundUnavailableField,
    GroundUnavailableFieldId,
    GroundUnavailableReason,
)
from shared.python.swing_sim.solver.spatial_targets import TargetPoint

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe, pytest.mark.contract]


def _complete_sample(series_id: str, trial_index: int) -> GroundStudySample:
    missed = replace(_target(), point=TargetPoint(500.0, 0.0, 0.0))
    study = project_ground_study(
        _qualified_request(),
        _result(),
        bound_surface=_bound_surface(),
        target=missed,
    )
    return GroundStudySample(series_id, trial_index, study)


def _partial_sample(series_id: str, trial_index: int) -> GroundStudySample:
    study = project_ground_study(
        _request(),
        _partial_bounce_result(0.25),
        target=_target(),
    )
    return GroundStudySample(series_id, trial_index, study)


def test_dataset_preserves_explicit_identity_metrics_and_target_miss() -> None:
    complete_sample = _complete_sample("driver/base", 2)
    partial_sample = _partial_sample("driver/base", 1)
    dataset = build_ground_study_scalar_dataset(
        (complete_sample, partial_sample),
        "seed:17/config:ground-v1",
        result_id="ground-variation-17",
    )

    assert [row.row_id for row in dataset.rows] == [
        "series:driver%2Fbase/trial:1",
        "series:driver%2Fbase/trial:2",
    ]
    partial, complete = dataset.rows
    assert partial.cohort == "censored"
    assert partial.value("total_distance_m") is not None
    assert partial.value("first_target_miss_distance_m") is not None
    assert partial.value("final_target_miss_distance_m") is None
    assert partial.attributes is not None
    assert partial.attributes["final_target_unavailable_reason"] == "endpoint_airborne"
    assert complete.cohort == "complete"
    assert complete.value("total_distance_m") == pytest.approx(228.0111017034039)
    assert complete.value("bounce_count") == 1.0
    assert complete.value("final_target_accepted") == 0.0
    assert complete.value("final_target_miss_distance_m") is not None
    assert complete.attributes is not None
    assert complete.attributes["result_sha256"] == complete_sample.study.result_sha256
    assert (
        complete.attributes["request_context_sha256"]
        == complete_sample.study.request_sha256
    )
    assert complete.attributes["profile_qualification_status"] == "qualified"
    assert complete.attributes["profile_model_use_status"] == "calibrated"
    assert complete.attributes["target_center_x_m"] == "500.0"
    assert complete.attributes["target_geometry"] == "surface_circle"
    assert complete.attributes["target_radius_m"] == "1.0"
    assert (
        complete.attributes["study_sha256"]
        == hashlib.sha256(complete_sample.study.to_json().encode("utf-8")).hexdigest()
    )
    assert complete.attributes["solver_eligible"] == "true"


def test_dataset_is_wire_deterministic_and_allows_same_trial_across_series() -> None:
    left = _complete_sample("iron", 3)
    right = _complete_sample("driver", 3)

    forward = build_ground_study_scalar_dataset(
        (left, right), "fixture:ordered", result_id="ground-order"
    )
    reverse = build_ground_study_scalar_dataset(
        (right, left), "fixture:ordered", result_id="ground-order"
    )

    assert forward.to_wire() == reverse.to_wire()
    assert [row.row_id for row in forward.rows] == [
        "series:driver/trial:3",
        "series:iron/trial:3",
    ]


def test_targetless_complete_study_keeps_physics_and_null_target_values() -> None:
    study = project_ground_study(
        _qualified_request(),
        _result(),
        bound_surface=_bound_surface(),
    )
    dataset = build_ground_study_scalar_dataset(
        (GroundStudySample("targetless", 0, study),),
        "fixture:targetless",
    )

    row = dataset.rows[0]
    assert row.cohort == "complete"
    assert row.value("total_distance_m") == pytest.approx(228.0111017034039)
    assert row.value("first_target_miss_distance_m") is None
    assert row.value("final_target_miss_distance_m") is None


def test_unqualified_complete_study_is_numeric_but_not_solver_eligible() -> None:
    study = project_ground_study(_request(), _result())
    dataset = build_ground_study_scalar_dataset(
        (GroundStudySample("unqualified", 0, study),),
        "fixture:unqualified",
    )

    row = dataset.rows[0]
    assert row.cohort == "complete"
    assert row.value("total_distance_m") == pytest.approx(228.0111017034039)
    assert row.attributes is not None
    assert row.attributes["solver_eligible"] == "false"
    assert row.attributes["solver_eligibility_reasons"] == "missing_profile_binding"


def test_profile_warnings_remain_separate_indexed_attributes() -> None:
    bound = bind_material_profile(
        _profile(evidence_kind=GroundEvidenceKind.ENGINEERING_ESTIMATE),
        SurfacePlacement(
            "firm-fairway",
            0.0,
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        ProfileOperatingCondition("fairway", 290.0, 0.24),
    )
    study = project_ground_study(
        replace(_request(), surface=bound.surface),
        _result(),
        bound_surface=bound,
    )
    row = build_ground_study_scalar_dataset(
        (GroundStudySample("illustrative", 0, study),),
        "fixture:profile-warning",
    ).rows[0]

    assert row.attributes is not None
    assert row.attributes["profile_warning.0"] == PROFILE_ILLUSTRATIVE_WARNING


def test_failed_study_retains_cohort_and_nulls_every_scalar() -> None:
    study = project_ground_study(_qualified_request(), _failed_result())
    dataset = build_ground_study_scalar_dataset(
        (GroundStudySample("failure", 0, study),),
        "fixture:failure",
    )

    row = dataset.rows[0]
    assert row.cohort == "failed"
    assert all(value is None for value in row.values.values())
    assert row.attributes is not None
    assert row.attributes["study_status"] == "failed"
    assert row.attributes["solver_eligible"] == "false"
    assert row.attributes["calibration_confidence"] == str(study.calibration.confidence)
    assert row.attributes["warning.0.code"] == "LITERATURE_CALIBRATION"


def test_unavailable_study_retains_typed_unavailability_evidence() -> None:
    result = replace(
        _failed_result(),
        status=GroundResultStatus.UNAVAILABLE,
        termination=GroundTermination(
            GroundTerminationReason.UNAVAILABLE_INPUT,
            _failed_result().termination.time_s,
            False,
        ),
        unavailable_fields=(
            GroundUnavailableField(
                GroundUnavailableFieldId.PHYSICAL_CONTACT_BRACKET,
                GroundUnavailableReason.NO_PHYSICAL_CONTACT,
                "flight event detector",
            ),
        ),
    )
    study = project_ground_study(_qualified_request(), result, target=_target())
    dataset = build_ground_study_scalar_dataset(
        (GroundStudySample("unavailable", 0, study),),
        "fixture:unavailable",
    )

    row = dataset.rows[0]
    assert row.cohort == "unavailable"
    assert all(value is None for value in row.values.values())
    assert row.attributes is not None
    assert row.attributes["unavailable.0.field_id"] == "physical_contact_bracket"
    assert row.attributes["unavailable.0.reason"] == "no_physical_contact"
    assert row.attributes["unavailable.0.provenance"] == "flight event detector"
    assert row.attributes["warning.0.code"] == "LITERATURE_CALIBRATION"
    assert row.attributes["warning.0.severity"] == "info"
    assert (
        row.attributes["warning.0.message"]
        == "Validate against measured turf before decision use."
    )


def test_study_and_target_evidence_bind_target_derived_scalars() -> None:
    near_target = replace(_target(), label="same", point=TargetPoint(10.0, 0.0, 0.0))
    far_target = replace(_target(), label="same", point=TargetPoint(500.0, 0.0, 0.0))
    studies = tuple(
        project_ground_study(
            _qualified_request(),
            _result(),
            bound_surface=_bound_surface(),
            target=target,
        )
        for target in (near_target, far_target)
    )
    dataset = build_ground_study_scalar_dataset(
        tuple(
            GroundStudySample("target", index, study)
            for index, study in enumerate(studies)
        ),
        "fixture:target-binding",
    )

    near, far = dataset.rows
    assert near.value("final_target_miss_distance_m") != far.value(
        "final_target_miss_distance_m"
    )
    assert near.attributes is not None and far.attributes is not None
    assert near.attributes["target_center_x_m"] == "10.0"
    assert far.attributes["target_center_x_m"] == "500.0"
    assert near.attributes["study_sha256"] != far.attributes["study_sha256"]


def test_builder_rejects_duplicate_identity_and_overflow_without_truncation() -> None:
    sample = _complete_sample("series", 0)
    with pytest.raises(ContractViolationError, match="unique"):
        build_ground_study_scalar_dataset((sample, sample), "fixture:duplicate")
    with pytest.raises(ContractViolationError, match="max_rows"):
        build_ground_study_scalar_dataset(
            (sample, _complete_sample("series", 1)),
            "fixture:bounded",
            max_rows=1,
        )


def test_builder_rejects_empty_or_unidentified_datasets() -> None:
    with pytest.raises(ContractViolationError, match="samples must be nonempty"):
        build_ground_study_scalar_dataset((), "fixture:empty")

    sample = _complete_sample("series", 0)
    with pytest.raises(ContractViolationError, match="source_provenance"):
        build_ground_study_scalar_dataset((sample,), " ")
    with pytest.raises(ContractViolationError, match="result_id"):
        build_ground_study_scalar_dataset(
            (sample,),
            "fixture:result",
            result_id=" ",
        )
    with pytest.raises(ContractViolationError, match="samples must be iterable"):
        build_ground_study_scalar_dataset(None, "fixture:not-iterable")


@pytest.mark.parametrize("series_id", ["", "   "])
def test_sample_identity_fails_closed(series_id: str) -> None:
    with pytest.raises(ContractViolationError, match="series_id"):
        GroundStudySample(series_id, 0, _complete_sample("valid", 0).study)
