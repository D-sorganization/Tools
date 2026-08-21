"""Registered execution tests for the shared rotating-base provider."""

from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pytest

from shared.python.swing_sim.rotating_base import (
    MODEL_TIER,
    RotatingBaseRunRequest,
    load_embedded_qualified_study,
    registered_metric_matches_authority,
    registered_run_json,
    run_registered_case,
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
    authority = load_embedded_qualified_study()
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
        assert registered_metric_matches_authority(
            field.name, actual_value, expected_value
        )
    assert result.trace.time_s.shape == (241,)
    assert result.trace.force_on_club_n.shape == (241, 2, 2)
    assert registered_metric_matches_authority(
        "impact_speed_m_s",
        result.trace.clubhead_speed_m_s[-1],
        expected.metrics.impact_speed_m_s,
    )
    assert np.all(np.isfinite(result.trace.contact_power_on_club_w))
    assert np.all(np.isfinite(result.trace.force_generated_couple_nm))
    with pytest.raises(ValueError, match="read-only"):
        result.trace.clubhead_speed_m_s[0] = 0.0
    with pytest.raises(ValueError, match="request and case identities"):
        type(result)(
            request=RotatingBaseRunRequest(
                torso_profile="accelerate",
                matching_rule="relative_club_rate",
                initial_torso_rate_rad_s=3.5,
            ),
            case=result.case,
            trace=result.trace,
        )
    payload = json.loads(registered_run_json(result))
    assert payload["schema_id"] == "swing-sim/rotating-base-run-result"
    assert payload["source_revision"] == result.source_revision
    assert payload["request"]["case_index"] == expected.case_index
    assert payload["case"]["exclusion_reasons"] == list(expected.exclusion_reasons)
    assert payload["boundaries"] == {
        "coaching_recommendation": "unsupported",
        "coordinate_semantics": "nonanatomical_model_coordinate",
        "human_validation": "unavailable",
    }
