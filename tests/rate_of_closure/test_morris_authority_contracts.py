"""Strict versioned wire contracts for the Rate Morris authority."""

from __future__ import annotations

import pytest

from rate_of_closure.application.morris.contracts import (
    MORRIS_JOB_SCHEMA_ID,
    MORRIS_REQUEST_SCHEMA_ID,
    MorrisJobEnvelope,
    parse_morris_request,
)
from rate_of_closure.simulation import BallSupportMode, ContactMode
from shared.python.contracts import (
    ContractLevel,
    get_contract_level,
    set_contract_level,
)
from shared.python.swing_sim.run_config import SwingRunMode
from shared.python.swing_sim.variation import variable_registry

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def request_document() -> dict[str, object]:
    """Return one minimal valid v1 request document."""
    key = "swing_sim.swing.yaw_deg"
    return {
        "schema_id": MORRIS_REQUEST_SCHEMA_ID,
        "schema_version": 1,
        "request_id": "request-17",
        "base": {
            "club_name": "Driver 10.5°",
            "support_mode": "tee",
            "tee_height_m": 0.0381,
            "plane_yaw_deg": 0.0,
            "plane_side_tilt_deg": 0.0,
            "plane_forward_tilt_deg": 0.0,
            "pendulum_m1_kg": 7.5,
            "pendulum_l1_m": 0.75,
            "pendulum_lc1_m": 0.3375,
            "pendulum_i1_kg_m2": 1.210546875,
            "pendulum_m2_kg": 0.35,
            "pendulum_l2_m": 1.0,
            "pendulum_lc2_m": 0.7557142857142858,
            "pendulum_i2_kg_m2": 0.2877354761904762,
            "damping_shoulder": 0.4,
            "damping_wrist": 0.25,
            "swing_duration_s": 0.05,
            "flight_model": "waterloo_penner",
            "impact_offset_toe_mm": 0.0,
            "impact_offset_high_mm": 0.0,
        },
        "factors": [
            {
                "spec_id": "yaw",
                "variable_key": key,
                "lower": -2.0,
                "upper": 2.0,
                "unit": variable_registry()[key].unit,
            }
        ],
        "trajectories": 2,
        "levels": 4,
        "seed": 17,
        "minimum_effects": 2,
        "worker_count": 1,
    }


def test_request_reconstructs_pinned_simulation_authority() -> None:
    request = parse_morris_request(request_document())
    config = request.base_config()

    assert config.source_kind == "double_pendulum"
    assert config.contact_mode is ContactMode.FIXED_BALL_CONTACT
    assert config.swing_run_config.mode is SwingRunMode.PASSIVE
    assert not config.swing_run_config.joint_locks.has_locks
    assert config.torque_library is None
    assert config.impact_time_s is None
    assert config.impact_time_offset_s == 0.0
    assert config.ball_setup.support_mode is BallSupportMode.TEE
    assert request.design().factors[0].variable_key == "swing_sim.swing.yaw_deg"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.update(extra=True), "fields"),
        (lambda item: item.pop("seed"), "fields"),
        (lambda item: item.update(schema_version=2), "version"),
        (lambda item: item.update(trajectories=100_001), "sample"),
        (lambda item: item.update(worker_count=33), "worker"),
        (
            lambda item: item["factors"][0].update(unit="rad"),  # type: ignore[index,union-attr]
            "unit",
        ),
        (
            lambda item: item["factors"].append(dict(item["factors"][0])),  # type: ignore[index,union-attr]
            "unique",
        ),
        (
            lambda item: item["base"].update(support_mode="ground"),  # type: ignore[union-attr]
            "tee_height_m",
        ),
    ],
)
def test_request_rejects_schema_resource_unit_and_tee_violations(
    mutation: object, message: str
) -> None:
    document = request_document()
    mutation(document)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError), match=message):
        parse_morris_request(document)


def test_job_envelope_is_exact_and_never_exposes_partial_report() -> None:
    envelope = MorrisJobEnvelope.running("job-1", "request-17", (1, 4), True)
    assert envelope.to_json_dict() == {
        "schema_id": MORRIS_JOB_SCHEMA_ID,
        "schema_version": 1,
        "job_id": "job-1",
        "request_id": "request-17",
        "status": "running",
        "completed_samples": 1,
        "total_samples": 4,
        "cancel_requested": True,
        "report": None,
        "error": None,
    }


def _reverse_factor(document: dict[str, object]) -> None:
    factor = document["factors"][0]  # type: ignore[index]
    factor.update(lower=2.0, upper=-2.0)


def _ground_tee_factor(document: dict[str, object]) -> None:
    document["base"].update(support_mode="ground", tee_height_m=0.0)  # type: ignore[union-attr]
    document["factors"] = [
        {
            "spec_id": "tee",
            "variable_key": "swing_sim.ball_setup.tee_height_m",
            "lower": 0.01,
            "upper": 0.05,
            "unit": "m",
        }
    ]


@pytest.mark.parametrize("level", [ContractLevel.WARN, ContractLevel.OFF])
@pytest.mark.parametrize(
    "mutation",
    [
        _reverse_factor,
        lambda item: item["base"].update(pendulum_m1_kg=-7.5),  # type: ignore[union-attr]
        _ground_tee_factor,
        lambda item: item["base"].update(pendulum_lc1_m=0.9),  # type: ignore[union-attr]
        lambda item: item["base"].update(damping_wrist=-0.1),  # type: ignore[union-attr]
        lambda item: item["base"].update(swing_duration_s=0.0),  # type: ignore[union-attr]
        lambda item: item["base"].update(impact_offset_toe_mm=81.0),  # type: ignore[union-attr]
    ],
)
def test_wire_validation_is_unconditional_when_shared_dbc_is_not_enforcing(
    level: ContractLevel, mutation: object
) -> None:
    document = request_document()
    mutation(document)  # type: ignore[operator]
    original = get_contract_level()
    try:
        set_contract_level(level)
        with pytest.raises((TypeError, ValueError)):
            parse_morris_request(document)
    finally:
        set_contract_level(original)


@pytest.mark.parametrize("level", [ContractLevel.WARN, ContractLevel.OFF])
@pytest.mark.parametrize(
    ("variable_key", "lower", "upper"),
    [
        ("swing_sim.impact.delivery.impact_offset_toe_mm", -81.0, 0.0),
        ("swing_sim.impact.delivery.impact_offset_toe_mm", 0.0, 81.0),
        ("swing_sim.impact.delivery.impact_offset_high_mm", -41.0, 0.0),
        ("swing_sim.impact.delivery.impact_offset_high_mm", 0.0, 41.0),
    ],
)
def test_delivery_factor_endpoints_fail_closed_at_both_physical_bounds(
    level: ContractLevel, variable_key: str, lower: float, upper: float
) -> None:
    document = request_document()
    document["factors"] = [
        {
            "spec_id": "offset",
            "variable_key": variable_key,
            "lower": lower,
            "upper": upper,
            "unit": "mm",
        }
    ]
    original = get_contract_level()
    try:
        set_contract_level(level)
        with pytest.raises(ValueError, match="factor endpoint"):
            parse_morris_request(document)
    finally:
        set_contract_level(original)
