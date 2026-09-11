"""Deterministic versioned study wire contract (IA-T6, #5075)."""

from __future__ import annotations

import json

import pytest

from shared.python.swing_sim.impact_studies import (
    STUDY_WIRE_FORMAT,
    CompletenessChecks,
    EvidenceTier,
    ImpactStudyV1,
    InvalidCase,
    MetricRecord,
    Provenance,
    parse_study_wire,
    serialize_study_wire,
)


def _provenance() -> Provenance:
    return Provenance(
        code_id="swing_sim.impact_interval/1@6b27c4f05",
        data_ids=("sha256:abc123",),
        calibration_id="cal-2026-09-01",
        coordinate_frame="lab_right_handed_z_up",
    )


def _study(
    tier: EvidenceTier = EvidenceTier.MEASURED_VALIDATED,
    *,
    with_acoustics: bool = True,
) -> ImpactStudyV1:
    return ImpactStudyV1(
        study_id="study-0001",
        model_tier=tier,
        provenance=_provenance(),
        launch_metrics=(MetricRecord(name="ball_speed_mps", value=52.1, unit="m/s"),),
        contact_metrics=(
            MetricRecord(
                name="contact_time_ms", value=0.45, unit="ms", uncertainty=0.01
            ),
        ),
        vibration_metrics=(
            MetricRecord(name="head_mode_1_hz", value=8100.0, unit="Hz"),
        ),
        acoustic_metrics=(
            (MetricRecord(name="peak_spl_db", value=94.2, unit="dB"),)
            if with_acoustics
            else None
        ),
        completeness=CompletenessChecks(
            energy_closure_residual_fraction=0.002,
            convergence_demonstrated=True,
        ),
        invalid_cases=(InvalidCase(case_id="ecc-07", reason="no separation found"),),
        v1_coupling_report='{"report_format":"golf_club.impact_coupling_report/1"}',
    )


def test_wire_format_is_pinned_v1() -> None:
    assert STUDY_WIRE_FORMAT == "swing_sim.impact_study/1"
    assert _study().schema_format == STUDY_WIRE_FORMAT


@pytest.mark.contract
def test_serialization_is_byte_identical_for_identical_inputs() -> None:
    first = serialize_study_wire(_study())
    second = serialize_study_wire(_study())
    assert first == second
    assert json.loads(first)["schema_format"] == STUDY_WIRE_FORMAT


@pytest.mark.contract
def test_round_trip_preserves_every_section() -> None:
    study = _study()
    parsed = parse_study_wire(serialize_study_wire(study))
    assert parsed == study
    assert parsed.acoustic_metrics == study.acoustic_metrics
    assert parsed.invalid_cases == study.invalid_cases
    assert parsed.v1_coupling_report == study.v1_coupling_report


@pytest.mark.contract
def test_unavailable_acoustics_serializes_as_absent_not_zero() -> None:
    payload = json.loads(serialize_study_wire(_study(with_acoustics=False)))
    assert "acoustic_metrics" not in payload


def test_finite_values_are_enforced() -> None:
    with pytest.raises(ValueError, match="finite"):
        ImpactStudyV1(
            study_id="study-0002",
            model_tier=EvidenceTier.ILLUSTRATIVE,
            provenance=Provenance(code_id="engine@1", data_ids=()),
            launch_metrics=(
                MetricRecord(name="speed", value=float("nan"), unit="m/s"),
            ),
            contact_metrics=(),
            vibration_metrics=(),
            acoustic_metrics=None,
            completeness=CompletenessChecks(
                energy_closure_residual_fraction=None,
                convergence_demonstrated=False,
            ),
            invalid_cases=(),
            v1_coupling_report=None,
        )


def test_negative_uncertainty_is_rejected() -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        MetricRecord(name="x", value=1.0, unit="m", uncertainty=-0.1)


def test_measured_tier_requires_calibration_and_data() -> None:
    with pytest.raises(ValueError, match="calibration"):
        ImpactStudyV1(
            study_id="study-0003",
            model_tier=EvidenceTier.MEASURED_VALIDATED,
            provenance=Provenance(code_id="engine@1", data_ids=("sha256:abc",)),
            launch_metrics=(),
            contact_metrics=(),
            vibration_metrics=(),
            acoustic_metrics=None,
            completeness=CompletenessChecks(
                energy_closure_residual_fraction=None,
                convergence_demonstrated=True,
            ),
            invalid_cases=(),
            v1_coupling_report=None,
        )


def test_measured_tier_requires_data_ids() -> None:
    with pytest.raises(ValueError, match="data"):
        ImpactStudyV1(
            study_id="study-0004",
            model_tier=EvidenceTier.MEASURED_VALIDATED,
            provenance=Provenance(code_id="engine@1", data_ids=(), calibration_id="c1"),
            launch_metrics=(),
            contact_metrics=(),
            vibration_metrics=(),
            acoustic_metrics=None,
            completeness=CompletenessChecks(
                energy_closure_residual_fraction=None,
                convergence_demonstrated=True,
            ),
            invalid_cases=(),
            v1_coupling_report=None,
        )


def test_verified_tier_requires_demonstrated_convergence() -> None:
    with pytest.raises(ValueError, match="convergence"):
        ImpactStudyV1(
            study_id="study-0005",
            model_tier=EvidenceTier.VERIFIED,
            provenance=Provenance(code_id="engine@1", data_ids=()),
            launch_metrics=(),
            contact_metrics=(),
            vibration_metrics=(),
            acoustic_metrics=None,
            completeness=CompletenessChecks(
                energy_closure_residual_fraction=None,
                convergence_demonstrated=False,
            ),
            invalid_cases=(),
            v1_coupling_report=None,
        )


def test_illustrative_tier_has_no_extra_requirements() -> None:
    study = ImpactStudyV1(
        study_id="study-0006",
        model_tier=EvidenceTier.ILLUSTRATIVE,
        provenance=Provenance(code_id="engine@1", data_ids=()),
        launch_metrics=(),
        contact_metrics=(),
        vibration_metrics=(),
        acoustic_metrics=None,
        completeness=CompletenessChecks(
            energy_closure_residual_fraction=None,
            convergence_demonstrated=False,
        ),
        invalid_cases=(),
        v1_coupling_report=None,
    )
    assert parse_study_wire(serialize_study_wire(study)) == study


def test_empty_acoustic_tuple_is_refused_in_study_construction() -> None:
    with pytest.raises(
        ValueError, match="acoustic_metrics must be None or a non-empty tuple"
    ):
        ImpactStudyV1(
            study_id="study-0007",
            model_tier=EvidenceTier.ILLUSTRATIVE,
            provenance=Provenance(code_id="engine@1", data_ids=()),
            launch_metrics=(),
            contact_metrics=(),
            vibration_metrics=(),
            acoustic_metrics=(),  # empty tuple must be refused; use None for absence
            completeness=CompletenessChecks(
                energy_closure_residual_fraction=None,
                convergence_demonstrated=False,
            ),
            invalid_cases=(),
            v1_coupling_report=None,
        )


def test_wire_with_empty_acoustic_metrics_is_refused() -> None:
    base = json.loads(serialize_study_wire(_study(with_acoustics=False)))
    base["acoustic_metrics"] = []
    with pytest.raises(ValueError, match="acoustic_metrics"):
        parse_study_wire(json.dumps(base))


def test_wire_with_unknown_top_level_field_is_refused() -> None:
    base = json.loads(serialize_study_wire(_study()))
    base["unexpected_field"] = "malicious_or_unknown"
    with pytest.raises(ValueError, match="unknown field.*unexpected_field"):
        parse_study_wire(json.dumps(base))


def test_wire_with_unknown_provenance_field_is_refused() -> None:
    base = json.loads(serialize_study_wire(_study()))
    base["provenance"]["unexpected_subfield"] = 42
    with pytest.raises(ValueError, match="unknown field.*unexpected_subfield"):
        parse_study_wire(json.dumps(base))


def test_wire_with_unknown_completeness_field_is_refused() -> None:
    base = json.loads(serialize_study_wire(_study()))
    base["completeness"]["unexpected_subfield"] = 42
    with pytest.raises(ValueError, match="unknown field.*unexpected_subfield"):
        parse_study_wire(json.dumps(base))


def test_wire_with_unknown_metric_field_is_refused() -> None:
    base = json.loads(serialize_study_wire(_study()))
    base["launch_metrics"][0]["extra_metric_field"] = "bad"
    with pytest.raises(ValueError, match="unknown field.*extra_metric_field"):
        parse_study_wire(json.dumps(base))


def test_wire_with_unknown_invalid_case_field_is_refused() -> None:
    base = json.loads(serialize_study_wire(_study()))
    base["invalid_cases"][0]["extra_case_field"] = "bad"
    with pytest.raises(ValueError, match="unknown field.*extra_case_field"):
        parse_study_wire(json.dumps(base))


def test_non_string_v1_coupling_report_is_refused() -> None:
    with pytest.raises((TypeError, ValueError), match="v1_coupling_report"):
        ImpactStudyV1(
            study_id="study-0008",
            model_tier=EvidenceTier.ILLUSTRATIVE,
            provenance=Provenance(code_id="engine@1", data_ids=()),
            launch_metrics=(),
            contact_metrics=(),
            vibration_metrics=(),
            acoustic_metrics=None,
            completeness=CompletenessChecks(),
            invalid_cases=(),
            v1_coupling_report={"report_format": "golf_club.impact_coupling_report/1"},  # type: ignore[arg-type]
        )


def test_wire_with_dict_v1_coupling_report_is_refused() -> None:
    base = json.loads(serialize_study_wire(_study()))
    base["v1_coupling_report"] = {"report_format": "golf_club.impact_coupling_report/1"}
    with pytest.raises(ValueError, match="v1_coupling_report"):
        parse_study_wire(json.dumps(base))


def test_invalid_json_v1_coupling_report_is_refused() -> None:
    with pytest.raises(ValueError, match="v1_coupling_report.*JSON"):
        ImpactStudyV1(
            study_id="study-0009",
            model_tier=EvidenceTier.ILLUSTRATIVE,
            provenance=Provenance(code_id="engine@1", data_ids=()),
            launch_metrics=(),
            contact_metrics=(),
            vibration_metrics=(),
            acoustic_metrics=None,
            completeness=CompletenessChecks(),
            invalid_cases=(),
            v1_coupling_report="not-valid-json",
        )


def test_unrecognized_format_v1_coupling_report_is_refused() -> None:
    with pytest.raises(ValueError, match="v1_coupling_report format"):
        ImpactStudyV1(
            study_id="study-0010",
            model_tier=EvidenceTier.ILLUSTRATIVE,
            provenance=Provenance(code_id="engine@1", data_ids=()),
            launch_metrics=(),
            contact_metrics=(),
            vibration_metrics=(),
            acoustic_metrics=None,
            completeness=CompletenessChecks(),
            invalid_cases=(),
            v1_coupling_report='{"format":"unrecognized/v99"}',
        )


def test_uncertainty_convention_and_confidence_level_validation() -> None:
    # Convention without uncertainty is rejected
    with pytest.raises(ValueError, match="uncertainty_convention requires uncertainty"):
        MetricRecord(
            name="v",
            value=10.0,
            unit="m/s",
            uncertainty=None,
            uncertainty_convention="coverage_interval",
        )

    # Confidence level without uncertainty is rejected
    with pytest.raises(ValueError, match="confidence_level requires uncertainty"):
        MetricRecord(
            name="v", value=10.0, unit="m/s", uncertainty=None, confidence_level=0.95
        )

    # Invalid confidence level bounds
    with pytest.raises(ValueError, match="confidence_level must be in \\(0, 1\\)"):
        MetricRecord(
            name="v", value=10.0, unit="m/s", uncertainty=0.5, confidence_level=1.5
        )
    with pytest.raises(ValueError, match="confidence_level must be in \\(0, 1\\)"):
        MetricRecord(
            name="v", value=10.0, unit="m/s", uncertainty=0.5, confidence_level=0.0
        )

    # Default convention is one_sigma
    m_default = MetricRecord(name="v", value=10.0, unit="m/s", uncertainty=0.5)
    assert m_default.uncertainty_convention == "one_sigma"
    assert m_default.confidence_level is None

    # Explicit convention and confidence level
    m_ci = MetricRecord(
        name="v",
        value=10.0,
        unit="m/s",
        uncertainty=0.5,
        uncertainty_convention="coverage_interval",
        confidence_level=0.95,
    )
    assert m_ci.uncertainty_convention == "coverage_interval"
    assert m_ci.confidence_level == 0.95


def test_uncertainty_convention_round_trip() -> None:
    metric = MetricRecord(
        name="ball_speed_mps",
        value=52.1,
        unit="m/s",
        uncertainty=0.3,
        uncertainty_convention="coverage_interval",
        confidence_level=0.95,
    )
    study = ImpactStudyV1(
        study_id="study-ci-001",
        model_tier=EvidenceTier.ILLUSTRATIVE,
        provenance=Provenance(code_id="engine@1", data_ids=()),
        launch_metrics=(metric,),
        contact_metrics=(),
        vibration_metrics=(),
        acoustic_metrics=None,
        completeness=CompletenessChecks(),
        invalid_cases=(),
        v1_coupling_report=None,
    )
    wire = serialize_study_wire(study)
    parsed = parse_study_wire(wire)
    assert parsed.launch_metrics[0].uncertainty_convention == "coverage_interval"
    assert parsed.launch_metrics[0].confidence_level == 0.95


def test_paired_control_cannot_be_self() -> None:
    with pytest.raises(ValueError, match="paired_control_id cannot equal study_id"):
        ImpactStudyV1(
            study_id="study-self-control",
            model_tier=EvidenceTier.ILLUSTRATIVE,
            provenance=Provenance(
                code_id="engine@1",
                data_ids=(),
                paired_control_id="study-self-control",
            ),
            launch_metrics=(),
            contact_metrics=(),
            vibration_metrics=(),
            acoustic_metrics=None,
            completeness=CompletenessChecks(),
            invalid_cases=(),
            v1_coupling_report=None,
        )


def test_qualification_evidence_verification_for_measured_validated_tier() -> None:
    from shared.python.swing_sim.impact_studies import verify_study_qualification

    study = _study(EvidenceTier.MEASURED_VALIDATED)
    # Without a resolver, MEASURED_VALIDATED refuses validation because
    # metadata labels are not evidence authentication.
    with pytest.raises(ValueError, match="requires an evidence resolver"):
        verify_study_qualification(study)

    # With a resolver that confirms calibration_id and data_ids
    valid_ids = {"cal-2026-09-01", "sha256:abc123"}
    verify_study_qualification(study, resolver=lambda i: i in valid_ids)

    # With an unresolvable calibration_id
    with pytest.raises(ValueError, match="unresolvable calibration_id"):
        verify_study_qualification(study, resolver=lambda i: i == "sha256:abc123")

    # With an unresolvable data_id
    with pytest.raises(ValueError, match="unresolvable data_id"):
        verify_study_qualification(study, resolver=lambda i: i == "cal-2026-09-01")
