"""Pure request and export helpers for the PyQt wind-strategy workflow."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from rate_of_closure.model import MPH_PER_MPS
from rate_of_closure.simulation import launch_from_delivery, launch_from_direct
from rate_of_closure.variation.scalar_ensemble_io import scalar_ensemble_csv
from shared.python.swing_sim.flight import (
    LaunchConditions,
    ScalarDistribution,
    StrategyAnalysisConfig,
    StrategyAnalysisRequest,
    TargetPoint,
    WindEstimateError,
    WindStrategy,
    WindUncertaintySpec,
)
from shared.python.swing_sim.flight.direction import LaunchDirectionConvention
from shared.python.swing_sim.impact import DeliveryParameters
from shared.python.swing_sim.solver import (
    SpatialTarget,
    SurfaceCircleTolerance,
    SurfaceCorridorTolerance,
)

_UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class FlightExplorerLaunchValues:
    """Values required to build the launch currently shown in Flight Explorer."""

    direct_mode: bool
    speed_mps: float
    direct: Mapping[str, float]
    delivery: Mapping[str, float]
    direction_convention: LaunchDirectionConvention


def build_flight_explorer_launch(
    values: FlightExplorerLaunchValues,
) -> LaunchConditions:
    """Build direct or impact-derived launch conditions from current controls."""
    if values.direct_mode:
        return launch_from_direct(
            ball_speed_mph=values.speed_mps * MPH_PER_MPS,
            launch_angle_deg=values.direct["launch_angle_deg"],
            launch_direction_deg=values.direct["launch_direction_deg"],
            spin_rpm=values.direct["spin_rpm"],
            spin_axis_tilt_deg=values.direct["spin_axis_tilt_deg"],
            direction_convention=values.direction_convention,
        )
    return launch_from_delivery(
        DeliveryParameters(
            clubhead_speed_mps=values.speed_mps,
            club_path_deg=values.delivery["club_path_deg"],
            face_angle_deg=values.delivery["face_angle_deg"],
            attack_angle_deg=values.delivery["attack_angle_deg"],
            dynamic_loft_deg=values.delivery["dynamic_loft_deg"],
            impact_offset_toe_mm=values.delivery["impact_offset_toe_mm"],
            impact_offset_high_mm=values.delivery["impact_offset_high_mm"],
        )
    )


@dataclass(frozen=True)
class WindStrategyLaunchContext:
    """Current launch, canonical spatial target, and selected flight model."""

    launch: LaunchConditions
    target: SpatialTarget
    model_name: str

    def __post_init__(self) -> None:
        if self.target.kind != "landing_area":
            raise ValueError("wind strategy analysis requires a landing-area target")
        if not self.model_name.strip():
            raise ValueError("flight model name must be nonempty")


@dataclass(frozen=True)
class WindStrategySettings:
    """User-authored wind ensemble settings in displayed engineering units."""

    trials: int
    true_speed_mps: float
    true_from_bearing_deg: float
    speed_bias_mps: float
    speed_std_mps: float
    bearing_bias_deg: float
    bearing_std_deg: float
    correlation: float
    aim_gain_deg_per_mps: float
    seed: int

    def __post_init__(self) -> None:
        values = (
            self.true_speed_mps,
            self.true_from_bearing_deg,
            self.speed_bias_mps,
            self.speed_std_mps,
            self.bearing_bias_deg,
            self.bearing_std_deg,
            self.correlation,
            self.aim_gain_deg_per_mps,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("wind strategy settings must be finite")
        if not 1 <= self.trials <= 100_000:
            raise ValueError("trials must be in [1, 100000]")
        if not 0 <= self.seed <= _UINT32_MAX:
            raise ValueError("seed must be a uint32 integer")
        if self.true_speed_mps < 0.0:
            raise ValueError("true wind speed must be nonnegative")
        if self.speed_std_mps < 0.0 or self.bearing_std_deg < 0.0:
            raise ValueError("estimate standard deviations must be nonnegative")
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError("correlation must be in [-1, 1]")


def _target_radius(target: SpatialTarget) -> float:
    tolerance = target.tolerance
    if isinstance(tolerance, SurfaceCircleTolerance):
        return float(tolerance.radius_m)
    if isinstance(tolerance, SurfaceCorridorTolerance):
        return float(min(tolerance.half_length_m, tolerance.half_width_m))
    raise ValueError("landing target requires a surface tolerance")


def target_hold_note(target: SpatialTarget) -> str:
    """Describe how the core circular target-hold metric treats the target."""
    tolerance = target.tolerance
    if isinstance(tolerance, SurfaceCircleTolerance):
        return f"Target hold uses the current {tolerance.radius_m:g} m circle."
    radius = _target_radius(target)
    return (
        "Target hold uses a conservative inscribed circle "
        f"({radius:g} m radius) for the current corridor."
    )


def build_strategy_request(
    context: WindStrategyLaunchContext,
    settings: WindStrategySettings,
) -> StrategyAnalysisRequest:
    """Build one immutable shared-core request without changing its physics."""
    forward_m, _elevation_m, right_m = context.target.point.app_coordinates_m
    uncertainty = WindUncertaintySpec(
        trials=settings.trials,
        seed=settings.seed,
        true_speed_mps=ScalarDistribution(
            "fixed", settings.true_speed_mps, minimum=0.0
        ),
        true_from_bearing_deg=ScalarDistribution(
            "fixed", settings.true_from_bearing_deg
        ),
        estimate_error=WindEstimateError(
            settings.speed_bias_mps,
            settings.speed_std_mps,
            settings.bearing_bias_deg,
            settings.bearing_std_deg,
            settings.correlation,
        ),
        provenance="rate-of-closure/pyqt/user-declared",
    )
    return StrategyAnalysisRequest(
        uncertainty=uncertainty,
        strategies=(
            WindStrategy(
                "current-launch",
                "Current Launch",
                context.launch,
                math.radians(settings.aim_gain_deg_per_mps),
            ),
        ),
        target=TargetPoint(forward_m, right_m),
        analysis=StrategyAnalysisConfig(
            model_name=context.model_name,
            target_radius_m=_target_radius(context.target),
        ),
    )


__all__ = [
    "FlightExplorerLaunchValues",
    "WindStrategyLaunchContext",
    "WindStrategySettings",
    "build_flight_explorer_launch",
    "build_strategy_request",
    "scalar_ensemble_csv",
    "target_hold_note",
]
