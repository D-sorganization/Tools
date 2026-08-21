"""Registered execution tests for the shared rotating-base provider."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from shared.python.swing_sim.rotating_base import (
    MODEL_TIER,
    RotatingBaseRunRequest,
    load_qualified_study,
    run_registered_case,
)

FIXTURE = (
    Path(__file__).parent / "fixtures" / "rotating_base_torso_velocity_study_v1.json"
)
pytestmark = [pytest.mark.contract, pytest.mark.scientific]


def test_registered_request_rejects_out_of_design_values() -> None:
    with pytest.raises(ValueError, match="torso_profile"):
        RotatingBaseRunRequest(
            torso_profile="coast",
            matching_rule="relative_club_rate",
            initial_torso_rate_rad_s=3.5,
        )
    with pytest.raises(ValueError, match="initial_torso_rate_rad_s"):
        RotatingBaseRunRequest(
            torso_profile="accelerate",
            matching_rule="relative_club_rate",
            initial_torso_rate_rad_s=4.0,
        )


def test_registered_full_run_reproduces_pinned_case_and_traces() -> None:
    authority = load_qualified_study(FIXTURE)
    expected = authority.study.cases[0]
    request = RotatingBaseRunRequest(
        torso_profile=expected.torso_profile,
        matching_rule=expected.matching_rule,
        initial_torso_rate_rad_s=expected.initial_torso_rate_rad_s,
    )

    result = run_registered_case(request)

    assert result.model_tier == MODEL_TIER
    assert result.case.case_index == expected.case_index
    assert result.case.valid == expected.valid
    assert result.case.exclusion_reasons == expected.exclusion_reasons
    for field in fields(expected.metrics):
        actual_value = getattr(result.case.metrics, field.name)
        expected_value = getattr(expected.metrics, field.name)
        assert actual_value == pytest.approx(expected_value, abs=1e-10)
    assert result.trace.time_s.shape == (241,)
    assert result.trace.force_on_club_n.shape == (241, 2, 2)
    assert result.trace.clubhead_speed_m_s[-1] == pytest.approx(
        expected.metrics.impact_speed_m_s, abs=1e-10
    )
    assert np.all(np.isfinite(result.trace.contact_power_on_club_w))
    assert np.all(np.isfinite(result.trace.force_generated_couple_nm))
    with pytest.raises(ValueError, match="read-only"):
        result.trace.clubhead_speed_m_s[0] = 0.0
