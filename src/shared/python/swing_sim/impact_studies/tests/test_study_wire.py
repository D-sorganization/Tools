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
