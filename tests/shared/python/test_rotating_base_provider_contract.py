"""Contracts for the shared rotating-base transfer provider."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from shared.python.swing_sim.rotating_base import (
    EXPECTED_UPSTREAM_SOURCE_REVISION,
    RotatingBaseProviderResult,
)


def _case(*, index: int, valid: bool) -> dict[str, object]:
    return {
        "case_index": index,
        "torso_profile": "accelerate",
        "matching_rule": "relative_club_rate",
        "initial_torso_rate_rad_s": 3.5,
        "initial_club_rate_rad_s": 4.5,
        "final_torso_rate_rad_s": 4.1,
        "impact_speed_m_s": 39.2,
        "clubhead_speed_gain_m_s": 8.2,
        "contact_work_on_club_j": 12.1,
        "braking_grip_work_j": 1.2,
        "force_couple_work_j": -0.3,
        "negative_along_path_impulse_ns": 0.4,
        "bilateral_wrist_work_j": -0.8,
        "total_control_work_j": 14.0,
        "distal_energy_gain_j": 10.3,
        "peak_grip_force_n": 74.0,
        "maximum_constraint_residual_m": 1e-10,
        "maximum_velocity_constraint_residual_m_s": 2e-10,
        "maximum_contact_power_identity_residual_w": 3e-10,
        "work_energy_closure_j": 2e-4,
        "valid": valid,
        "exclusion_reasons": [] if valid else ["registered_limit_exceeded"],
    }


def _payload() -> dict[str, object]:
    return {
        "schema_id": "swing-sim/rotating-base-provider-result",
        "schema_version": 1,
        "source_revision": EXPECTED_UPSTREAM_SOURCE_REVISION,
        "study": {
            "schema_version": "rotating-base-torso-velocity-study-v1",
            "study_id": "registered-rotating-base-two-hand-torso-velocity-grid",
            "model_tier": "planar_rotating_base_two_hand_compliant_club",
            "attempted_case_count": 2,
            "valid_case_count": 1,
            "matching_rules": {
                "relative_club_rate": "relative rate fixed",
                "absolute_club_rate": "absolute rate fixed",
            },
            "cases": [_case(index=0, valid=True), _case(index=1, valid=False)],
            "same_state_killswitch": {
                "branch_time_s": 0.03,
                "pre_branch_state_max_abs_difference": 0.0,
                "channels": {
                    name: {
                        "pre_branch_state_max_abs_difference": 0.0,
                        "delivery_speed_difference_m_s": value,
                        "post_branch_contact_work_difference_j": -value,
                    }
                    for name, value in {
                        "torso": 1.2,
                        "bilateral_arm": -1.3,
                        "bilateral_wrist": 1.8,
                    }.items()
                },
            },
            "claims": {
                "universal_high_torso_velocity_strategy": "not_supported",
                "human_coaching_strategy": "unsupported",
            },
            "limitations": [
                "Planar reduced coordinates are not anatomical torso observables."
            ],
        },
    }


def test_provider_result_retains_design_adverse_rows_and_boundaries() -> None:
    result = RotatingBaseProviderResult.from_mapping(_payload())

    assert result.source_revision == EXPECTED_UPSTREAM_SOURCE_REVISION
    assert result.study.attempted_case_count == 2
    assert result.study.valid_case_count == 1
    assert result.study.cases[1].exclusion_reasons == ("registered_limit_exceeded",)
    assert result.study.same_state_killswitch.channel_names == (
        "torso",
        "bilateral_arm",
        "bilateral_wrist",
    )
    assert result.study.human_coaching_supported is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(source_revision="untrusted"),
            "source_revision",
        ),
        (
            lambda payload: payload["study"].update(valid_case_count=2),
            "valid_case_count",
        ),
        (
            lambda payload: payload["study"]["cases"][1].update(exclusion_reasons=[]),
            "invalid case",
        ),
        (
            lambda payload: payload["study"]["same_state_killswitch"]["channels"].pop(
                "bilateral_wrist"
            ),
            "killswitch channels",
        ),
        (
            lambda payload: payload["study"]["claims"].update(
                human_coaching_strategy="supported"
            ),
            "human coaching",
        ),
    ],
)
def test_provider_result_fails_closed_on_semantic_drift(
    mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    payload = _payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        RotatingBaseProviderResult.from_mapping(payload)
