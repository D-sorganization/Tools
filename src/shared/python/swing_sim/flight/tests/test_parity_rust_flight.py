"""Rust <-> Python ball-flight parity (graceful skip without the wheel).

The ``tools_core`` Rust kernel and the Python Waterloo/Penner model share
the same quadratic drag law and defaults (cd0=0.21, cd1=0.05, cd2=0.02),
so a zero-spin trajectory must agree to integrator tolerance (fixed-step
RK4 vs adaptive RK45 + dt-quantised ground stop). Lift laws differ
(quadratic-polynomial vs Penner power fit), so spinning shots are only
compared coarsely.

Skips cleanly when ``tools_core`` is absent or is an older wheel that
does not expose ``simulate_trajectory``.
"""

from __future__ import annotations

import math

import pytest

from shared.python.swing_sim.flight import (
    LaunchConditions,
    WaterlooPennerModel,
    is_rust_available,
    simulate_trajectory_rust,
)

pytestmark = pytest.mark.skipif(
    not is_rust_available(),
    reason=(
        "tools_core wheel absent or too old (no simulate_trajectory); "
        "Rust ball-flight fast path unavailable on this interpreter"
    ),
)


@pytest.mark.parity
@pytest.mark.physics
def test_zero_spin_trajectory_matches_penner_within_tolerance() -> None:
    launch = LaunchConditions(
        ball_speed=70.0, launch_angle=math.radians(12.0), spin_rate=0.0
    )
    rust = simulate_trajectory_rust(launch, max_time=20.0, dt=0.005)
    python = WaterlooPennerModel().simulate(launch, max_time=20.0, dt=0.005)

    assert rust.carry_distance == pytest.approx(python.carry_distance, rel=0.02)
    assert rust.max_height == pytest.approx(python.max_height, rel=0.02)
    assert rust.flight_time == pytest.approx(python.flight_time, rel=0.02)
    assert rust.landing_angle == pytest.approx(python.landing_angle, abs=2.0)


@pytest.mark.parity
@pytest.mark.physics
def test_spinning_driver_shot_agrees_coarsely() -> None:
    """Different lift laws: expect the same plausible band, not tight parity.

    The Rust kernel uses a quadratic-polynomial Cl (cl1=0.38) with 0.08 1/s
    spin decay; Penner uses a capped power-law fit — the Rust carry is
    systematically shorter for a spinning driver shot.
    """
    launch = LaunchConditions(
        ball_speed=74.0, launch_angle=math.radians(12.0), spin_rate=2600.0
    )
    rust = simulate_trajectory_rust(launch, max_time=20.0, dt=0.005)
    python = WaterlooPennerModel().simulate(launch, max_time=20.0, dt=0.005)

    assert 150.0 <= rust.carry_distance <= 320.0
    assert 150.0 <= python.carry_distance <= 320.0
    assert rust.carry_distance < python.carry_distance  # weaker lift law
    # Backspin lifts both above the no-spin trajectory of the same launch.
    no_spin = LaunchConditions(
        ball_speed=74.0, launch_angle=math.radians(12.0), spin_rate=0.0
    )
    assert rust.max_height > simulate_trajectory_rust(no_spin).max_height


@pytest.mark.parity
@pytest.mark.unit
def test_rust_result_is_flight_frame() -> None:
    """Facade output uses the flight frame: height on z, lateral on y."""
    launch = LaunchConditions(
        ball_speed=70.0, launch_angle=math.radians(12.0), spin_rate=0.0
    )
    result = simulate_trajectory_rust(launch)
    pos = result.to_position_array()
    assert pos[:, 2].max() > 1.0  # height accumulates on z (flight up)
    assert abs(pos[-1, 1]) < 1e-6  # no lateral deviation without spin


@pytest.mark.parity
@pytest.mark.unit
def test_rust_wrapper_validates_preconditions() -> None:
    good = LaunchConditions(ball_speed=70.0, launch_angle=0.2)
    with pytest.raises(ValueError, match="max_time"):
        simulate_trajectory_rust(good, max_time=0.0)
    with pytest.raises(ValueError, match="dt"):
        simulate_trajectory_rust(good, dt=-0.01)
    zero_speed = LaunchConditions(ball_speed=0.0, launch_angle=0.2)
    with pytest.raises(ValueError, match="ball_speed"):
        simulate_trajectory_rust(zero_speed)
