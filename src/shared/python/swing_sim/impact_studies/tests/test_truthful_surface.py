"""Truthful application surface contract (IA-T6, #5075).

The surface must distinguish illustrative, verified and measured/validated
results, show ball launch separately from sound/feel, and refuse
unavailable acoustics rather than fabricate zeros.
"""

from __future__ import annotations

import pytest

from shared.python.swing_sim.impact_studies import (
    AcousticsUnavailableError,
    CompletenessChecks,
    EvidenceTier,
    ImpactStudyV1,
    MetricRecord,
    Provenance,
    acoustic_metrics_or_raise,
    acoustics_available,
    study_statement,
)


def _study(
    tier: EvidenceTier,
    *,
    with_acoustics: bool = False,
) -> ImpactStudyV1:
    completeness = CompletenessChecks(
        energy_closure_residual_fraction=0.001,
        convergence_demonstrated=tier is not EvidenceTier.ILLUSTRATIVE,
    )
    provenance = (
        Provenance(
            code_id="engine@1",
            data_ids=("sha256:abc",),
            calibration_id="cal-1",
        )
        if tier is EvidenceTier.MEASURED_VALIDATED
        else Provenance(code_id="engine@1", data_ids=())
    )
    return ImpactStudyV1(
        study_id="study-truth",
        model_tier=tier,
        provenance=provenance,
        launch_metrics=(MetricRecord(name="ball_speed_mps", value=52.1, unit="m/s"),),
        contact_metrics=(),
        vibration_metrics=(),
        acoustic_metrics=(
            (MetricRecord(name="peak_spl_db", value=94.2, unit="dB"),)
            if with_acoustics
            else None
        ),
        completeness=completeness,
        invalid_cases=(),
        v1_coupling_report=None,
    )


def test_acoustics_available_only_when_measured_record_present() -> None:
    assert acoustics_available(_study(EvidenceTier.MEASURED_VALIDATED)) is False
    assert (
        acoustics_available(
            _study(EvidenceTier.MEASURED_VALIDATED, with_acoustics=True)
        )
        is True
    )


def test_unavailable_acoustics_raise_instead_of_fabricating_zeros() -> None:
    with pytest.raises(AcousticsUnavailableError, match="no acoustic"):
        acoustic_metrics_or_raise(_study(EvidenceTier.MEASURED_VALIDATED))


def test_available_acoustics_return_the_measured_records() -> None:
    study = _study(EvidenceTier.MEASURED_VALIDATED, with_acoustics=True)
    records = acoustic_metrics_or_raise(study)
    assert records == study.acoustic_metrics
    assert all(record.value != 0.0 for record in records)


@pytest.mark.parametrize(
    ("tier", "label"),
    [
        (EvidenceTier.ILLUSTRATIVE, "illustrative"),
        (EvidenceTier.VERIFIED, "verified"),
        (EvidenceTier.MEASURED_VALIDATED, "measured/validated"),
    ],
)
def test_statement_labels_the_evidence_tier(tier: EvidenceTier, label: str) -> None:
    statement = study_statement(_study(tier))
    assert label in statement


def test_statement_shows_launch_separately_from_sound() -> None:
    study = _study(EvidenceTier.MEASURED_VALIDATED, with_acoustics=True)
    statement = study_statement(study)
    lines = statement.splitlines()
    launch_lines = [line for line in lines if "Ball launch" in line]
    sound_lines = [line for line in lines if "Sound" in line]
    assert launch_lines and sound_lines
    assert all("peak_spl_db" not in line for line in launch_lines)
    assert all("ball_speed_mps" not in line for line in sound_lines)


def test_statement_reports_absent_acoustics_explicitly() -> None:
    statement = study_statement(_study(EvidenceTier.VERIFIED))
    assert "unavailable" in statement.lower()
