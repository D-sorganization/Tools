"""Tests for the authoritative flight-to-repeated-bounce composition facade."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

import shared.python.swing_sim.flight.ground_bounce_execution as subject
from shared.python.swing_sim.flight import (
    FlightGroundTransferError,
    FlightGroundTransferSettings,
    FlightResult,
    FlightStatePoint,
    LaunchConditions,
    TrajectoryPoint,
    execute_repeated_bounce_from_flight,
)
from shared.python.swing_sim.ground import (
    BounceTerminationReason,
    CalibrationKind,
    GroundCalibration,
    GroundFrame,
    GroundProvenance,
    GroundSurfaceProfile,
    GroundUnavailableFieldId,
    GroundUnavailableReason,
    RepeatedBounceRequestResultPair,
)

BALL_RADIUS_M = 0.02135


def _surface() -> GroundSurfaceProfile:
    return GroundSurfaceProfile(
        surface_id="composition-plane",
        provider_id="tools.flight-test",
        provider_version="1.0.0",
        frame=GroundFrame.TARGET,
        height_m=0.0,
        normal_unit=(0.0, 1.0, 0.0),
        surface_velocity_m_s=(0.0, 0.0, 0.0),
        normal_restitution=0.4,
        static_friction=0.35,
        kinetic_friction=0.25,
        rolling_resistance=0.04,
        firmness_pa=1_000_000.0,
        hardness_fraction=0.7,
        grass_height_m=0.01,
        compressibility_fraction=0.2,
        compression_damping_fraction=0.2,
        turf_density_kg_m3=180.0,
        moisture_fraction=0.3,
    )


def _settings() -> FlightGroundTransferSettings:
    return FlightGroundTransferSettings(
        request_id="flight-bounce-composition-001",
        surface=_surface(),
        calibration=GroundCalibration(
            "test-calibration", CalibrationKind.MEASURED, "test evidence", 1.0
        ),
        provenance=GroundProvenance("pytest", "1.0", "local", "a" * 64),
        max_time_s=12.0,
        output_interval_s=0.01,
        max_events=32,
    )


def _state(
    time_s: float,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
) -> FlightStatePoint:
    return FlightStatePoint(
        time_s,
        np.array(position),
        np.array(velocity),
        np.array((1.0, 2.0, 3.0)),
    )


def _crossing_result() -> FlightResult:
    return FlightResult(
        (
            _state(0.0, (0.0, 0.0, 0.0), (10.0, 0.0, 5.0)),
            _state(0.1, (1.0, -1.0, 0.05), (10.0, -1.0, 1.0)),
            _state(0.2, (2.0, -2.0, 0.03), (10.0, -1.0, -2.0)),
            _state(0.3, (3.0, -3.0, -0.001), (10.0, -1.0, -2.0)),
        ),
        "synthetic",
    )


def _no_contact_result() -> FlightResult:
    return FlightResult(
        (
            _state(0.0, (0.0, 0.0, 0.0), (10.0, 0.0, 5.0)),
            _state(0.1, (1.0, 0.0, 0.05), (10.0, 0.0, 2.0)),
        ),
        "no-contact",
    )


def _grazing_result() -> FlightResult:
    return FlightResult(
        (
            _state(0.0, (0.0, 0.0, 0.0), (5.0, 0.0, 3.0)),
            _state(0.1, (0.5, 0.0, 0.04), (5.0, 0.0, 1.0)),
            _state(0.2, (1.0, 0.0, 0.0), (5.0, 0.0, 0.0)),
        ),
        "grazing",
    )


def _missing_angular_state_result() -> FlightResult:
    points = tuple(
        TrajectoryPoint(point.time, point.position, point.velocity)
        for point in _crossing_result().trajectory
    )
    return FlightResult(points, "missing-angular-state")


def _launch() -> LaunchConditions:
    return LaunchConditions(
        ball_speed=10.0,
        launch_angle=math.radians(20.0),
        spin_rate=300.0,
        ball_radius=BALL_RADIUS_M,
    )


def test_composer_rejects_non_exact_contract_types_before_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_transfer(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("transfer must not run")

    monkeypatch.setattr(subject, "build_ground_simulation_request", forbidden_transfer)

    class FlightResultSubclass(FlightResult):
        """Nominal subclass used to prove the exact-type boundary."""

    class LaunchConditionsSubclass(LaunchConditions):
        """Nominal subclass used to prove the exact-type boundary."""

    class TransferSettingsSubclass(FlightGroundTransferSettings):
        """Nominal subclass used to prove the exact-type boundary."""

    flight = _crossing_result()
    launch = _launch()
    settings = _settings()
    invalid_calls = (
        (cast(FlightResult, object()), launch, settings, "exact FlightResult"),
        (
            FlightResultSubclass(flight.trajectory, flight.model_name),
            launch,
            settings,
            "exact FlightResult",
        ),
        (flight, cast(LaunchConditions, object()), settings, "exact LaunchConditions"),
        (
            flight,
            LaunchConditionsSubclass(10.0, math.radians(20.0)),
            settings,
            "exact LaunchConditions",
        ),
        (
            flight,
            launch,
            cast(FlightGroundTransferSettings, object()),
            "exact FlightGroundTransferSettings",
        ),
        (
            flight,
            launch,
            TransferSettingsSubclass(**settings.__dict__),
            "exact FlightGroundTransferSettings",
        ),
    )
    for bad_flight, bad_launch, bad_settings, message in invalid_calls:
        with pytest.raises(ValueError, match=message):
            execute_repeated_bounce_from_flight(
                bad_flight,
                bad_launch,
                bad_settings,
            )
    assert calls == 0


@pytest.mark.parametrize("capture_speed_m_s", [0.0, -0.1, math.inf, math.nan, True])
def test_composer_validates_callback_and_capture_before_transfer(
    monkeypatch: pytest.MonkeyPatch,
    capture_speed_m_s: float,
) -> None:
    calls = 0

    def forbidden_transfer(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("transfer must not run")

    monkeypatch.setattr(subject, "build_ground_simulation_request", forbidden_transfer)
    with pytest.raises(ValueError, match="is_cancelled must be callable or None"):
        execute_repeated_bounce_from_flight(
            _crossing_result(),
            _launch(),
            _settings(),
            capture_speed_m_s=cast(float, capture_speed_m_s),
            is_cancelled=cast(Any, 1),
        )
    with pytest.raises(
        ValueError, match="capture_speed_m_s must be finite and positive"
    ):
        execute_repeated_bounce_from_flight(
            _crossing_result(),
            _launch(),
            _settings(),
            capture_speed_m_s=cast(float, capture_speed_m_s),
        )
    assert calls == 0


def test_composer_preserves_request_result_identity_and_digests() -> None:
    pair = execute_repeated_bounce_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
    )

    assert type(pair) is RepeatedBounceRequestResultPair
    assert pair.request.ground_request.request_id == _settings().request_id
    assert pair.request.ground_request.surface.surface_id == _surface().surface_id
    assert pair.result.request_id == pair.request.request_id
    assert pair.result.surface_id == pair.request.surface_id
    assert pair.result.frame is pair.request.frame
    assert pair.result.request_fingerprint_sha256 == pair.request.ground_request_sha256
    assert pair.execution_input_sha256 == pair.request.execution_input_sha256


def test_composer_carries_capture_threshold_into_execution() -> None:
    bounce = execute_repeated_bounce_from_flight(
        _crossing_result(), _launch(), _settings(), capture_speed_m_s=0.05
    )
    captured = execute_repeated_bounce_from_flight(
        _crossing_result(), _launch(), _settings(), capture_speed_m_s=3.0
    )

    assert bounce.request.capture_speed_m_s == pytest.approx(0.05)
    assert captured.request.capture_speed_m_s == pytest.approx(3.0)
    assert bounce.result.impacts[0].effective_restitution == pytest.approx(0.4)
    assert captured.result.impacts[0].effective_restitution == pytest.approx(0.0)
    assert captured.result.termination.reason is BounceTerminationReason.SETTLED_TO_SKID
    assert bounce.execution_input_sha256 != captured.execution_input_sha256


def test_composer_preserves_preflight_cancellation_as_a_valid_pair() -> None:
    pair = execute_repeated_bounce_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
        is_cancelled=lambda: True,
    )

    assert type(pair) is RepeatedBounceRequestResultPair
    assert pair.result.termination.reason is BounceTerminationReason.CANCELLED
    assert pair.result.trajectory == ()
    assert pair.result.events == ()
    assert pair.result.impacts == ()
    assert pair.result.airborne_segments == ()
    assert pair.result.handoff_state is None


@pytest.mark.parametrize(
    ("case", "message", "field_id", "reason"),
    [
        (
            "no-contact",
            "flight trajectory has no descending physical contact crossing",
            GroundUnavailableFieldId.PHYSICAL_CONTACT_BRACKET,
            GroundUnavailableReason.NO_PHYSICAL_CONTACT,
        ),
        (
            "grazing",
            "physical contact bracket must be strictly incoming, not grazing",
            GroundUnavailableFieldId.PHYSICAL_CONTACT_BRACKET,
            GroundUnavailableReason.SOURCE_OUT_OF_BOUNDS,
        ),
        (
            "missing-angular-state",
            "flight trajectory does not propagate terminal angular velocity",
            GroundUnavailableFieldId.TERMINAL_ANGULAR_VELOCITY,
            GroundUnavailableReason.SOURCE_DOES_NOT_PROPAGATE,
        ),
    ],
)
def test_composer_preserves_typed_transfer_failures_without_execution(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
    field_id: GroundUnavailableFieldId,
    reason: GroundUnavailableReason,
) -> None:
    calls = 0

    def forbidden_execution(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("bounce execution must not run")

    monkeypatch.setattr(subject, "execute_repeated_bounce_request", forbidden_execution)
    result = {
        "no-contact": _no_contact_result,
        "grazing": _grazing_result,
        "missing-angular-state": _missing_angular_state_result,
    }[case]()
    with pytest.raises(FlightGroundTransferError) as error:
        execute_repeated_bounce_from_flight(
            result,
            _launch(),
            _settings(),
        )

    assert str(error.value) == message
    assert error.value.field_id is field_id
    assert error.value.reason is reason
    assert calls == 0
