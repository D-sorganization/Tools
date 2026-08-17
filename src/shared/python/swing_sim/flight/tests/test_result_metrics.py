"""Analytic and degenerate trajectory tests for flight result metrics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

import pytest

from shared.python.swing_sim.flight.impact_solution_adapter import (
    _spin_axis_tilt_deg as impact_spin_axis_tilt_deg,
)
from shared.python.swing_sim.flight.result_contract import (
    AvailabilityReason,
    FlightMetricId,
    SignRule,
    ValueStatus,
    flight_metric_catalog,
)
from shared.python.swing_sim.flight.result_metrics import (
    FlightMetricInputs,
    FlightRunManifest,
    GroundModelResult,
    MetricTrajectoryPoint,
    derive_flight_metric_result,
)
from shared.python.swing_sim.flight.spin_axis_convention import spin_axis_tilt_deg

FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "ball_flight_metrics_golden_v1.json"
)


def _manifest() -> FlightRunManifest:
    return FlightRunManifest(
        model_id="analytic_fixture",
        model_version="1.0.0",
        integration_status="complete",
        termination_reason="ground_crossing",
        environment=(
            ("air_density_kg_m3", "1.225"),
            ("gravity_m_s2", "9.80665"),
        ),
        wind=(("model", "still_air"),),
        uncertainty_status="deterministic",
    )


def _analytic_inputs() -> FlightMetricInputs:
    points = (
        MetricTrajectoryPoint(0.0, (0.0, 0.0, 0.0), (10.0, 10.0, 1.0)),
        MetricTrajectoryPoint(1.0, (10.0, 5.0, 1.0), (10.0, 0.0, 1.0)),
        MetricTrajectoryPoint(2.0, (20.0, 1.0, 3.0), (10.0, -5.0, 1.0)),
        MetricTrajectoryPoint(3.0, (30.0, -1.0, 5.0), (10.0, -7.0, 1.0)),
    )
    return FlightMetricInputs(
        trajectory=points,
        spin_vector_rpm=(100.0, 2500.0, -50.0),
        target_position_m=(24.0, 0.0, 6.0),
    )


def test_analytic_landing_is_interpolated_and_metrics_are_not_aliases() -> None:
    result = derive_flight_metric_result(_analytic_inputs(), _manifest())

    assert result.scalar(FlightMetricId.FLIGHT_TIME) == pytest.approx(2.5)
    assert result.vector(FlightMetricId.LANDING_POSITION) == pytest.approx(
        (25.0, 0.0, 4.0)
    )
    assert result.scalar(FlightMetricId.CARRY_DISTANCE) == pytest.approx(
        math.hypot(25.0, 4.0)
    )
    assert result.scalar(FlightMetricId.CARRY_OFFLINE) == pytest.approx(4.0)
    assert result.scalar(FlightMetricId.CURVE) != pytest.approx(4.0)
    assert result.scalar(FlightMetricId.TARGET_RESIDUAL) == pytest.approx(
        math.sqrt(5.0)
    )
    assert result.scalar(FlightMetricId.TARGET_DOWNRANGE_RESIDUAL) == pytest.approx(1.0)
    assert result.scalar(FlightMetricId.TARGET_LATERAL_RESIDUAL) == pytest.approx(-2.0)
    assert (
        result.value(FlightMetricId.TOTAL_DISTANCE).reason
        is AvailabilityReason.GROUND_MODEL_REQUIRED
    )


def test_qualified_ground_output_is_never_inferred_from_carry() -> None:
    ground = GroundModelResult(
        model_id="qualified-ground/v1",
        total_distance_m=28.0,
        roll_distance_m=2.7,
        bounce_count=2,
        final_offline_m=4.5,
    )
    inputs = _analytic_inputs().with_ground_result(ground)
    result = derive_flight_metric_result(inputs, _manifest())

    assert result.scalar(FlightMetricId.TOTAL_DISTANCE) == 28.0
    assert result.scalar(FlightMetricId.ROLL_DISTANCE) == 2.7
    assert (
        result.value(FlightMetricId.TOTAL_DISTANCE).status
        is ValueStatus.MODEL_DEPENDENT
    )
    assert (
        result.value(FlightMetricId.TOTAL_DISTANCE).provenance == "qualified-ground/v1"
    )
    with pytest.raises(ValueError, match="bounce_count must be an integer"):
        GroundModelResult(cast(int, "invalid"), 28.0, 2.7, 1.5, 4.5)


def test_positive_spin_axis_tilt_uses_fade_right_convention() -> None:
    tilt = math.radians(10.0)
    source = _analytic_inputs()
    inputs = FlightMetricInputs(
        source.trajectory,
        (0.0, -1000.0 * math.sin(tilt), 1000.0 * math.cos(tilt)),
        source.target_position_m,
    )

    result = derive_flight_metric_result(inputs, _manifest())

    assert result.scalar(FlightMetricId.SPIN_AXIS_TILT) == pytest.approx(10.0)
    assert (
        flight_metric_catalog().definition(FlightMetricId.SPIN_AXIS_TILT).sign_rule
        is SignRule.POSITIVE_RIGHT
    )


def test_spin_axis_tilt_projects_out_gyro_spin_in_every_producer() -> None:
    spin = (500.0, -100.0, 1000.0)
    expected = spin_axis_tilt_deg(spin)
    source = _analytic_inputs()
    result = derive_flight_metric_result(
        FlightMetricInputs(source.trajectory, spin, source.target_position_m),
        _manifest(),
    )

    assert expected == pytest.approx(math.degrees(math.atan2(100.0, 1000.0)))
    assert impact_spin_axis_tilt_deg(spin) == pytest.approx(expected)
    assert result.scalar(FlightMetricId.SPIN_AXIS_TILT) == pytest.approx(expected)


def test_degenerate_trajectories_return_reasons_and_invalid_order_fails() -> None:
    one_point = FlightMetricInputs(
        trajectory=(MetricTrajectoryPoint(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),),
        spin_vector_rpm=(0.0, 0.0, 0.0),
    )
    result = derive_flight_metric_result(one_point, _manifest())

    assert (
        result.value(FlightMetricId.CARRY_DISTANCE).reason
        is AvailabilityReason.INSUFFICIENT_TRAJECTORY
    )
    assert (
        result.value(FlightMetricId.SPIN_AXIS_TILT).reason
        is AvailabilityReason.ZERO_SPIN
    )
    assert (
        result.value(FlightMetricId.TARGET_RESIDUAL).reason
        is AvailabilityReason.TARGET_NOT_CONFIGURED
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        FlightMetricInputs(
            trajectory=(
                MetricTrajectoryPoint(1.0, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
                MetricTrajectoryPoint(1.0, (1.0, 0.0, 0.0), (1.0, -1.0, 0.0)),
            ),
            spin_vector_rpm=(0.0, 1.0, 0.0),
        )


def test_python_matches_shared_parity_fixture() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = derive_flight_metric_result(_analytic_inputs(), _manifest())
    expected = fixture["analytic_case"]["expected_scalars"]

    for metric_id, value in expected.items():
        assert result.scalar(FlightMetricId(metric_id)) == pytest.approx(
            value, abs=1e-10
        )
    digest = hashlib.sha256(flight_metric_catalog().to_json().encode()).hexdigest()
    assert digest == fixture["catalog_sha256"]
    result_digest = hashlib.sha256(result.to_json().encode()).hexdigest()
    assert result_digest == fixture["analytic_result_sha256"]
