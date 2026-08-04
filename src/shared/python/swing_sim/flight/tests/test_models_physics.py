"""Physics sanity tests for the ported flight models.

Ballistic limit, carry monotonicity, ground-event termination, and
cross-model carry spread for a standard driver launch.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest

from shared.python.swing_sim.flight import (
    FlightModelRegistry,
    LaunchConditions,
    WaterlooPennerModel,
)

# Standard driver launch: ~74 m/s (165 mph) ball speed, 12 deg, 2600 RPM.
DRIVER_LAUNCH = LaunchConditions(
    ball_speed=74.0,
    launch_angle=math.radians(12.0),
    spin_rate=2600.0,
)


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    """Prevent cross-test pollution of the class-level model dict."""
    FlightModelRegistry.reset()
    yield
    FlightModelRegistry.reset()


def _vacuum_range(speed: float, angle: float, g: float) -> float:
    """Analytic drag-free projectile range for a ground-level launch."""
    return speed * speed * math.sin(2.0 * angle) / g


@pytest.mark.physics
@pytest.mark.unit
def test_no_spin_low_speed_approaches_vacuum_projectile() -> None:
    """A slow spinless launch stays within a small drag margin of vacuum."""
    speed, angle = 10.0, math.radians(30.0)
    launch = LaunchConditions(ball_speed=speed, launch_angle=angle, spin_rate=0.0)
    result = WaterlooPennerModel().simulate(launch)

    expected = _vacuum_range(speed, angle, launch.gravity)
    assert result.carry_distance <= expected  # drag only removes range
    assert result.carry_distance > 0.90 * expected  # small drag margin
    expected_h = (speed * math.sin(angle)) ** 2 / (2.0 * launch.gravity)
    assert result.max_height == pytest.approx(expected_h, rel=0.10)


@pytest.mark.physics
@pytest.mark.unit
def test_carry_is_monotonic_in_ball_speed() -> None:
    model = WaterlooPennerModel()
    carries = []
    for speed in (40.0, 50.0, 60.0, 70.0, 80.0):
        launch = LaunchConditions(
            ball_speed=speed, launch_angle=math.radians(12.0), spin_rate=2600.0
        )
        carries.append(model.simulate(launch).carry_distance)
    assert all(b > a for a, b in zip(carries, carries[1:], strict=False))


@pytest.mark.physics
@pytest.mark.unit
def test_ground_event_terminates_trajectory() -> None:
    result = WaterlooPennerModel().simulate(DRIVER_LAUNCH, max_time=60.0)
    final = result.trajectory[-1]
    # Terminal event: final height at the ground, not at max_time.
    assert final.position[2] == pytest.approx(0.0, abs=1e-6)
    assert result.flight_time < 60.0
    assert result.landing_angle > 0.0  # descending at landing
    # Interior of the trajectory is strictly above ground.
    heights = result.to_position_array()[1:-1, 2]
    assert np.all(heights > -1e-9)


@pytest.mark.physics
@pytest.mark.unit
def test_backspin_increases_peak_height() -> None:
    model = WaterlooPennerModel()
    no_spin = LaunchConditions(
        ball_speed=74.0, launch_angle=math.radians(12.0), spin_rate=0.0
    )
    assert model.simulate(DRIVER_LAUNCH).max_height > model.simulate(no_spin).max_height


@pytest.mark.physics
@pytest.mark.unit
def test_cross_model_carry_spread_for_driver_launch() -> None:
    """All 7 models land a standard driver launch in a plausible carry band."""
    carries: dict[str, float] = {}
    for model in FlightModelRegistry.get_all_models():
        result = model.simulate(DRIVER_LAUNCH, max_time=20.0)
        carries[model.name] = result.carry_distance
    assert len(carries) == 7
    for name, carry in carries.items():
        assert 150.0 <= carry <= 320.0, f"{name} carry {carry:.1f} m out of band"


@pytest.mark.physics
@pytest.mark.unit
def test_pure_backspin_launch_stays_on_target_line() -> None:
    """Zero azimuth + pure backspin gives no lateral deviation."""
    result = WaterlooPennerModel().simulate(DRIVER_LAUNCH)
    assert abs(result.lateral_deviation) < 1e-6


@pytest.mark.unit
def test_launch_conditions_validation() -> None:
    with pytest.raises(ValueError, match="ball_speed"):
        LaunchConditions(ball_speed=-1.0, launch_angle=0.2)
    with pytest.raises(ValueError, match="launch_angle"):
        LaunchConditions(ball_speed=70.0, launch_angle=12.0)  # degrees passed
    with pytest.raises(ValueError, match="spin_rate"):
        LaunchConditions(ball_speed=70.0, launch_angle=0.2, spin_rate=-5.0)
    with pytest.raises(ValueError, match="unit vector"):
        LaunchConditions(ball_speed=70.0, launch_angle=0.2, spin_axis=(0.0, -2.0, 0.0))


@pytest.mark.unit
def test_from_imperial_converts_units() -> None:
    launch = LaunchConditions.from_imperial(
        ball_speed_mph=165.0, launch_angle_deg=12.0, spin_rate_rpm=2600.0
    )
    assert launch.ball_speed == pytest.approx(165.0 * 0.44704)
    assert launch.launch_angle == pytest.approx(math.radians(12.0))
    assert launch.spin_rate == 2600.0
