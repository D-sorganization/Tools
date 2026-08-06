"""Standalone ball-flight exploration: launch entry -> flight, no swing.

Logic layer of the V2 Flight Explorer (epic #4120): build
:class:`~shared.python.swing_sim.flight.types.LaunchConditions` either
directly from launch-monitor ball numbers (ball speed, launch angle,
launch direction, spin, spin-axis tilt) or from club delivery numbers run through
``swing_sim.impact.delivery`` and the rigid-body impact model, then
integrate the flight with any of the 7 literature models in
``swing_sim.flight`` and return an app-frame trajectory + metrics.

Sign conventions match the simulation session (app frame: x target,
y up, z right): launch direction and lateral are + right of target; spin-axis
tilt is + fade-side (curves right), matching the D-plane diagnostics in
``swing_sim.impact.delivery``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from rate_of_closure._contracts import ensure, require
from rate_of_closure.model import MPH_PER_MPS
from rate_of_closure.simulation.session import BALL_POSITION_M
from shared.python.swing_sim.flight import (
    LaunchConditions,
    LaunchDirection,
    LaunchDirectionConvention,
    WindScenario,
    derive_launch_conditions,
    from_flight_frame,
    launch_direction_to_flight_azimuth,
    to_flight_frame,
)
from shared.python.swing_sim.flight import simulate as flight_simulate
from shared.python.swing_sim.impact import (
    DeliveryParameters,
    ImpactModelType,
    ImpactSolverAPI,
    derive_delivery,
)

__all__ = [
    "EXPLORER_METRIC_KEYS",
    "FlightExploration",
    "WindComparison",
    "compare_wind",
    "explore_flight",
    "launch_from_delivery",
    "launch_from_direct",
]

#: Metric keys every exploration reports, in display order.
EXPLORER_METRIC_KEYS: tuple[str, ...] = (
    "ball_speed_mph",
    "launch_angle_deg",
    "launch_direction_deg",
    # Deprecated persistence alias retained for lossless older exports.
    "launch_azimuth_deg",
    "spin_rpm",
    "carry_m",
    "max_height_m",
    "flight_time_s",
    "landing_angle_deg",
    "lateral_m",
)


@dataclass(frozen=True)
class FlightExploration:
    """One standalone flight run (app frame).

    Attributes:
        launch: The flight-frame launch conditions used.
        model_name: The flight-model registry name used.
        times: (N,) trajectory sample times [s].
        positions: (N, 3) app-frame ball positions from the tee.
        metrics: Launch + flight summary keyed by
            :data:`EXPLORER_METRIC_KEYS`.
    """

    launch: LaunchConditions
    model_name: str
    times: np.ndarray
    positions: np.ndarray
    metrics: dict[str, float]


@dataclass(frozen=True)
class WindComparison:
    """Common-input no-wind and selected-wind trajectories and deltas."""

    calm: FlightExploration
    wind: FlightExploration
    scenario: WindScenario
    deltas: dict[str, float]


def launch_from_direct(
    ball_speed_mph: float,
    launch_angle_deg: float,
    launch_direction_deg: float | None = None,
    spin_rpm: float | None = None,
    spin_axis_tilt_deg: float | None = None,
    *,
    azimuth_deg: float | None = None,
    direction_convention: LaunchDirectionConvention = (
        LaunchDirectionConvention.APP_NATIVE
    ),
) -> LaunchConditions:
    """Launch conditions from right-positive launch-monitor ball numbers.

    Args:
        ball_speed_mph: Ball speed [mph], > 0.
        launch_angle_deg: Launch angle above horizontal [deg].
        launch_direction_deg: Horizontal launch direction [deg]; positive
            starts right and negative starts left of the target line.
        spin_rpm: Total spin rate [rpm], >= 0.
        spin_axis_tilt_deg: Spin-axis tilt [deg]; + = fade-side (curves
            right for a right-handed player), matching the D-plane tilt
            reported by ``swing_sim.impact.delivery``.
        azimuth_deg: Deprecated app-native alias for imported/caller data.
        direction_convention: Explicit convention of ``launch_direction_deg``.

    Returns:
        Flight-frame :class:`LaunchConditions`.
    """
    require(
        math.isfinite(ball_speed_mph) and ball_speed_mph > 0.0,
        "ball_speed_mph must be finite and > 0",
        ball_speed_mph,
    )
    # Accept the historical keyword without silently choosing between conflicts.
    if launch_direction_deg is None:
        if azimuth_deg is None:
            raise TypeError("launch_direction_deg is required")
        launch_direction_deg = azimuth_deg
    elif azimuth_deg is not None and not math.isclose(
        launch_direction_deg, azimuth_deg, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("conflicting launch-direction and legacy azimuth values")
    if spin_rpm is None or spin_axis_tilt_deg is None:
        raise TypeError("spin_rpm and spin_axis_tilt_deg are required")
    flight_azimuth_deg = launch_direction_to_flight_azimuth(
        LaunchDirection(launch_direction_deg, direction_convention)
    )
    # App direction + = right; flight-frame azimuth + = left (+y): flip.
    # Fade-side tilt (+) needs a downward (-z flight) sidespin component,
    # which the legacy spin_axis_angle decomposition produces for a
    # negative angle — hence both signs flip here.
    return LaunchConditions.from_imperial(
        ball_speed_mph=ball_speed_mph,
        launch_angle_deg=launch_angle_deg,
        spin_rate_rpm=spin_rpm,
        azimuth_angle_deg=flight_azimuth_deg,
        spin_axis_angle_deg=-spin_axis_tilt_deg,
    )


def launch_from_delivery(params: DeliveryParameters) -> LaunchConditions:
    """Launch conditions from club delivery numbers via the impact model.

    Runs the delivery front-end (:func:`derive_delivery`) and the
    rigid-body COR impact solve — the same physics chain as the
    simulation session, minus the swing and the club-specific
    bulge/roll gear-effect callable (no club is selected here).

    Args:
        params: Launch-monitor-style delivery parameters.

    Returns:
        Flight-frame :class:`LaunchConditions` for the struck ball.
    """
    derived = derive_delivery(params)
    solver = ImpactSolverAPI(ImpactModelType.RIGID_BODY)
    post = solver.solve_impact(
        timestamp=0.0,
        clubhead_velocity=derived.clubhead_velocity,
        clubhead_orientation=derived.face_normal,
        impact_offset=derived.impact_offset,
        record=False,
    )
    launch: LaunchConditions = derive_launch_conditions(
        to_flight_frame(post.ball_velocity),
        to_flight_frame(post.ball_angular_velocity),
    )
    return launch


def explore_flight(
    launch: LaunchConditions, model_name: str = "waterloo_penner"
) -> FlightExploration:
    """Integrate one standalone flight and package app-frame results.

    Args:
        launch: Flight-frame launch conditions.
        model_name: A ``FlightModelType`` value string (one of the 7
            literature models).

    Returns:
        A complete :class:`FlightExploration`.
    """
    flight = flight_simulate(launch, model_name=model_name)
    times = np.array([p.time for p in flight.trajectory])
    positions = (
        from_flight_frame(flight.to_position_array()) + BALL_POSITION_M
        if len(flight.trajectory)
        else np.zeros((0, 3))
    )
    metrics = {
        "ball_speed_mph": launch.ball_speed * MPH_PER_MPS,
        "launch_angle_deg": math.degrees(launch.launch_angle),
        # Flight azimuth + = left; public launch direction + = right.
        "launch_direction_deg": -math.degrees(launch.azimuth_angle),
        "launch_azimuth_deg": -math.degrees(launch.azimuth_angle),
        "spin_rpm": launch.spin_rate,
        "carry_m": float(flight.carry_distance),
        "max_height_m": float(flight.max_height),
        "flight_time_s": float(flight.flight_time),
        "landing_angle_deg": float(flight.landing_angle),
        # Flight lateral + = left (+y flight); app lateral + = right.
        "lateral_m": -float(flight.lateral_deviation),
    }
    ensure(
        set(metrics) == set(EXPLORER_METRIC_KEYS),
        "exploration metrics must cover EXPLORER_METRIC_KEYS exactly",
    )
    return FlightExploration(
        launch=launch,
        model_name=model_name,
        times=times,
        positions=np.asarray(positions),
        metrics=metrics,
    )


def compare_wind(
    launch: LaunchConditions,
    scenario: WindScenario,
    model_name: str = "waterloo_penner",
) -> WindComparison:
    """Evaluate identical launch inputs with no wind and selected wind."""
    require(launch is not None, "launch must be provided", launch)
    require(scenario is not None, "scenario must be provided", scenario)
    calm_launch = replace(launch, wind_speed=0.0, wind_scenario=None)
    wind_launch = replace(launch, wind_speed=0.0, wind_scenario=scenario)
    calm = explore_flight(calm_launch, model_name)
    wind = explore_flight(wind_launch, model_name)
    delta_keys = (
        "carry_m",
        "max_height_m",
        "flight_time_s",
        "landing_angle_deg",
        "lateral_m",
    )
    deltas = {key: wind.metrics[key] - calm.metrics[key] for key in delta_keys}
    return WindComparison(calm=calm, wind=wind, scenario=scenario, deltas=deltas)


def explore_with_optional_wind(
    launch: LaunchConditions,
    scenario: WindScenario | None,
    model_name: str,
) -> tuple[FlightExploration, WindComparison | None]:
    """Run one flight or a common-input calm/selected-wind pair."""
    if scenario is None:
        return explore_flight(launch, model_name), None
    comparison = compare_wind(launch, scenario, model_name)
    return comparison.wind, comparison
