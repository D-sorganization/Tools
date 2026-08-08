"""Physical flight-to-ground transfer tests for issue #4269."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from shared.python.swing_sim.ball_setup import BallSetup, BallSupportMode
from shared.python.swing_sim.flight import (
    FlightGroundTransferError,
    FlightGroundTransferSettings,
    FlightResult,
    FlightStatePoint,
    LaunchConditions,
    MacDonaldHanzelyModel,
    SurfaceFlightSimulationSettings,
    TrajectoryPoint,
    WaterlooPennerModel,
    build_ground_simulation_request,
    launch_relative_surface,
)
from shared.python.swing_sim.ground import (
    CalibrationKind,
    GroundCalibration,
    GroundFrame,
    GroundProvenance,
    GroundSurfaceProfile,
    GroundUnavailableFieldId,
    GroundUnavailableReason,
)

BALL_RADIUS_M = 0.02135


def _surface(height_m: float = 0.0) -> GroundSurfaceProfile:
    return GroundSurfaceProfile(
        surface_id="test-plane",
        provider_id="tools.flight-test",
        provider_version="1.0.0",
        frame=GroundFrame.TARGET,
        height_m=height_m,
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


def _settings(
    surface: GroundSurfaceProfile | None = None,
) -> FlightGroundTransferSettings:
    return FlightGroundTransferSettings(
        request_id="flight-ground-001",
        surface=_surface() if surface is None else surface,
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
    omega: tuple[float, float, float] = (1.0, 2.0, 3.0),
) -> FlightStatePoint:
    return FlightStatePoint(
        time_s, np.array(position), np.array(velocity), np.array(omega)
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


def _launch() -> LaunchConditions:
    return LaunchConditions(
        ball_speed=10.0,
        launch_angle=math.radians(20.0),
        spin_rate=300.0,
        ball_radius=BALL_RADIUS_M,
    )


def _assert_sequence_rejected(result: FlightResult, message: str) -> None:
    with pytest.raises(FlightGroundTransferError, match=message) as error:
        build_ground_simulation_request(result, _launch(), _settings())
    assert error.value.field_id is GroundUnavailableFieldId.PHYSICAL_CONTACT_BRACKET
    assert error.value.reason is GroundUnavailableReason.SOURCE_OUT_OF_BOUNDS


def test_native_models_propagate_full_terminal_angular_velocity() -> None:
    launch = LaunchConditions(
        ball_speed=30.0,
        launch_angle=math.radians(20.0),
        spin_rate=2400.0,
        spin_axis=(0.2, -0.9, 0.3872983346207417),
    )
    waterloo = WaterlooPennerModel().simulate(launch)
    macdonald = MacDonaldHanzelyModel(decay=0.07).simulate(launch)

    assert isinstance(waterloo.trajectory[-1], FlightStatePoint)
    np.testing.assert_allclose(
        waterloo.trajectory[-1].angular_velocity_rad_s,
        launch.get_spin_vector(),
    )
    expected_decay = math.exp(-0.07 * macdonald.flight_time)
    np.testing.assert_allclose(
        macdonald.trajectory[-1].angular_velocity_rad_s,
        launch.get_spin_vector() * expected_decay,
    )


def test_transfer_preserves_descending_sphere_contact_and_frame_rotation() -> None:
    request = build_ground_simulation_request(
        _crossing_result(), _launch(), _settings()
    )

    separated = request.last_separated_state
    penetrating = request.first_penetrating_state
    assert separated.time_s == pytest.approx(0.2)
    assert penetrating.time_s == pytest.approx(0.3)
    assert separated.position_m == pytest.approx((2.0, 0.03, 2.0))
    assert penetrating.position_m == pytest.approx((3.0, -0.001, 3.0))
    assert separated.velocity_m_s == pytest.approx((10.0, -2.0, 1.0))
    assert separated.angular_velocity_rad_s == pytest.approx((1.0, 3.0, -2.0))
    assert request.ball_radius_m == BALL_RADIUS_M
    assert request.ball_mass_kg == _launch().ball_mass


def test_initial_penetration_does_not_retrigger_before_later_contact() -> None:
    request = build_ground_simulation_request(
        _crossing_result(), _launch(), _settings()
    )
    assert request.last_separated_state.time_s > 0.0
    assert request.first_penetrating_state.time_s > request.last_separated_state.time_s


def test_transfer_fails_closed_without_a_descending_crossing() -> None:
    result = FlightResult(
        (
            _state(0.0, (0.0, 0.0, 0.0), (5.0, 0.0, 3.0)),
            _state(0.1, (0.5, 0.0, 0.05), (5.0, 0.0, 2.0)),
            _state(0.2, (1.0, 0.0, 0.06), (5.0, 0.0, 1.0)),
        ),
        "no-crossing",
    )
    with pytest.raises(FlightGroundTransferError, match="physical contact crossing"):
        build_ground_simulation_request(result, _launch(), _settings())


def test_transfer_rejects_grazing_contact() -> None:
    result = FlightResult(
        (
            _state(0.0, (0.0, 0.0, 0.0), (5.0, 0.0, 3.0)),
            _state(0.1, (0.5, 0.0, 0.04), (5.0, 0.0, 1.0)),
            _state(0.2, (1.0, 0.0, 0.0), (5.0, 0.0, 0.0)),
        ),
        "grazing",
    )
    with pytest.raises(FlightGroundTransferError, match="strictly incoming"):
        build_ground_simulation_request(result, _launch(), _settings())


def test_transfer_rejects_missing_angular_state_and_ambiguous_origin() -> None:
    base_points = tuple(
        TrajectoryPoint(point.time, point.position, point.velocity)
        for point in _crossing_result().trajectory
    )
    with pytest.raises(FlightGroundTransferError) as missing:
        build_ground_simulation_request(
            FlightResult(base_points, "missing-spin"), _launch(), _settings()
        )
    assert missing.value.field_id is GroundUnavailableFieldId.TERMINAL_ANGULAR_VELOCITY

    shifted = list(_crossing_result().trajectory)
    shifted[0] = _state(0.0, (0.01, 0.0, 0.0), (10.0, 0.0, 5.0))
    with pytest.raises(FlightGroundTransferError, match="launch origin"):
        build_ground_simulation_request(
            FlightResult(tuple(shifted), "shifted"), _launch(), _settings()
        )


@pytest.mark.parametrize(
    ("time_s", "position"),
    [
        (5e-10, (0.0, 0.0, 0.0)),
        (0.0, (5e-10, 0.0, 0.0)),
    ],
)
def test_transfer_requires_exact_launch_origin(
    time_s: float,
    position: tuple[float, float, float],
) -> None:
    points = list(_crossing_result().trajectory)
    points[0] = _state(time_s, position, (10.0, 0.0, 5.0))

    _assert_sequence_rejected(FlightResult(tuple(points), "origin-drift"), "exact")


def test_transfer_rejects_reversed_and_duplicate_time_sequences() -> None:
    reversed_points = list(_crossing_result().trajectory)
    reversed_points[1], reversed_points[2] = reversed_points[2], reversed_points[1]
    _assert_sequence_rejected(
        FlightResult(tuple(reversed_points), "reversed"), "strictly increasing"
    )

    duplicate_points = list(_crossing_result().trajectory)
    object.__setattr__(duplicate_points[2], "time", duplicate_points[1].time)
    _assert_sequence_rejected(
        FlightResult(tuple(duplicate_points), "duplicate"), "strictly increasing"
    )


@pytest.mark.parametrize("bad_time", [float("nan"), float("inf"), -0.1])
def test_transfer_rejects_nonfinite_and_negative_sample_times(
    bad_time: float,
) -> None:
    points = list(_crossing_result().trajectory)
    object.__setattr__(points[2], "time", bad_time)

    _assert_sequence_rejected(
        FlightResult(tuple(points), "invalid-time"), "finite and nonnegative"
    )


def test_launch_relative_tee_surface_uses_ball_bottom_height() -> None:
    setup = BallSetup(BallSupportMode.TEE, 0.04)
    launch = LaunchConditions(
        ball_speed=10.0,
        launch_angle=math.radians(20.0),
        spin_rate=300.0,
        ball_radius=BALL_RADIUS_M,
        ball_setup=setup,
    )
    surface = _surface()
    shifted = launch_relative_surface(surface, launch.ball_radius, setup)
    assert shifted.height_m == pytest.approx(-(BALL_RADIUS_M + 0.04))
    result = FlightResult(
        (
            _state(0.0, (0.0, 0.0, 0.0), (5.0, 0.0, 2.0)),
            _state(0.1, (0.5, 0.0, -0.03), (5.0, 0.0, -1.0)),
            _state(0.2, (1.0, 0.0, -0.041), (5.0, 0.0, -1.0)),
        ),
        "tee-origin",
    )
    request = build_ground_simulation_request(result, launch, _settings(surface))
    assert request.surface.height_m == pytest.approx(-(BALL_RADIUS_M + 0.04))
    assert request.first_penetrating_state.position_m[1] == pytest.approx(-0.041)


def test_native_tee_flight_terminates_at_launch_relative_ground() -> None:
    setup = BallSetup(BallSupportMode.TEE, 0.0381)
    launch = LaunchConditions(
        ball_speed=30.0,
        launch_angle=math.radians(20.0),
        spin_rate=2400.0,
        ball_setup=setup,
    )

    result = WaterlooPennerModel().simulate(launch)

    assert result.flight_time > 0.0
    assert result.trajectory[-1].position[2] == pytest.approx(-setup.tee_height_m)


def test_surface_aware_flight_on_lowered_slope_builds_contact_request() -> None:
    terrain = replace(_surface(-0.15), normal_unit=(0.0, 0.8, 0.6))
    launch = LaunchConditions(
        ball_speed=30.0,
        launch_angle=math.radians(20.0),
        spin_rate=2400.0,
    )
    relative = launch_relative_surface(terrain, launch.ball_radius, launch.ball_setup)
    simulation = SurfaceFlightSimulationSettings(
        launch_relative_surface=relative,
        max_time_s=10.0,
        output_interval_s=0.01,
    )

    result = WaterlooPennerModel().simulate_to_surface(launch, simulation)
    request = build_ground_simulation_request(result, launch, _settings(terrain))

    terminal = request.first_penetrating_state
    assert request.surface == relative
    assert request.surface.signed_gap_m(terminal, launch.ball_radius) == pytest.approx(
        0.0, abs=1e-9
    )
    assert terminal.position_m[1] < -0.1
    assert request.last_separated_state.time_s < terminal.time_s


def test_surface_aware_flight_honors_raised_terrain_with_tee_clearance() -> None:
    setup = BallSetup(BallSupportMode.TEE, 0.04)
    terrain = _surface(0.01)
    launch = LaunchConditions(
        ball_speed=25.0,
        launch_angle=math.radians(18.0),
        spin_rate=2000.0,
        ball_setup=setup,
    )
    relative = launch_relative_surface(terrain, launch.ball_radius, setup)
    simulation = SurfaceFlightSimulationSettings(relative)

    result = WaterlooPennerModel().simulate_to_surface(launch, simulation)
    request = build_ground_simulation_request(result, launch, _settings(terrain))

    assert request.surface == relative
    assert request.first_penetrating_state.position_m[1] == pytest.approx(-0.03)


def test_transfer_uses_signed_gap_for_arbitrary_surface_normal() -> None:
    inverse_root_two = math.sqrt(0.5)
    surface = replace(_surface(), normal_unit=(0.0, inverse_root_two, inverse_root_two))
    result = FlightResult(
        (
            _state(0.0, (0.0, 0.0, 0.0), (5.0, 0.0, 2.0)),
            _state(0.1, (0.5, -0.01, 0.08), (5.0, 1.0, -2.0)),
            _state(0.2, (1.0, 0.1, -0.1), (5.0, 1.0, -2.0)),
        ),
        "sloped-plane",
    )

    request = build_ground_simulation_request(result, _launch(), _settings(surface))

    assert request.last_separated_state.time_s == pytest.approx(0.1)
    assert request.first_penetrating_state.time_s == pytest.approx(0.2)
    assert request.surface.height_m == pytest.approx(-BALL_RADIUS_M)


@pytest.mark.parametrize("terrain_height_m", [-0.15, 0.2])
def test_launch_relative_surface_preserves_terrain_profile(
    terrain_height_m: float,
) -> None:
    setup = BallSetup(BallSupportMode.TEE, 0.03)
    base = replace(
        _surface(terrain_height_m),
        normal_unit=(0.0, 0.8, 0.6),
        surface_velocity_m_s=(0.1, 0.06, -0.08),
    )

    shifted = launch_relative_surface(base, BALL_RADIUS_M, setup)

    assert shifted.height_m == pytest.approx(
        terrain_height_m - BALL_RADIUS_M - setup.tee_height_m
    )
    assert shifted.normal_unit == base.normal_unit
    assert shifted.surface_velocity_m_s == base.surface_velocity_m_s
    assert shifted.static_friction == base.static_friction
    assert shifted.firmness_pa == base.firmness_pa
