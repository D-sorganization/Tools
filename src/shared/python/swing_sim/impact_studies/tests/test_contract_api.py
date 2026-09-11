"""Pinned downstream-facing API for impact_studies (IA-T6, #5075)."""

from __future__ import annotations

import pytest

import shared.python.swing_sim.impact_studies as studies

EXPECTED_PUBLIC_API = {
    "AcousticsUnavailableError",
    "CompletenessChecks",
    "EvidenceResolver",
    "EvidenceTier",
    "ImpactStudyV1",
    "InvalidCase",
    "MetricRecord",
    "Provenance",
    "STUDY_WIRE_FORMAT",
    "acoustic_metrics_or_raise",
    "acoustics_available",
    "parse_study_wire",
    "serialize_study_wire",
    "study_statement",
    "verify_study_qualification",
}


@pytest.mark.contract
def test_public_api_is_explicit_and_pinned() -> None:
    assert set(studies.__all__) == EXPECTED_PUBLIC_API
    for symbol in studies.__all__:
        assert getattr(studies, symbol) is not None
