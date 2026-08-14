"""Ball-flight subpackage of :mod:`shared.python.swing_sim` (epic #4103, #4107).

Self-facaded port of UpstreamDrift's pure-Python flight stack
(``physics/flight_models.py`` and the launch-derivation seam of
``physics/swing_ball_flight_pipeline.py``): seven literature flight models
behind :class:`FlightModelRegistry`, scipy ``solve_ivp`` RK45 integration
with a terminal ground event, a public launch-conditions deriver, frame
adapters between the app frame (x target / y up / z right) and the flight
frame (x forward / y left / z up), and a graceful Rust fast path backed by
the canonical ``rust_core/tools-core/src/ball_flight.rs`` kernel.

Import from this facade only; module layout underneath is private.
"""

from __future__ import annotations

from ._rust_facade import is_rust_available, simulate_trajectory_rust
from .frames import from_flight_frame, to_flight_frame
from .launch import derive_launch_conditions
from .models import (
    BallFlightModel,
    ConstantCoefficientModel,
    ConstantCoefficientSpec,
    MacDonaldHanzelyModel,
    WaterlooPennerModel,
)
from .pipeline import FlightSimulatorProtocol, simulate
from .registry import FlightModelRegistry, FlightModelType, compare_models
from .types import (
    DEFAULT_BACKSPIN_AXIS,
    FlightResult,
    LaunchConditions,
    TrajectoryPoint,
    compute_flight_metrics,
)
from .wind import WIND_SCHEMA_VERSION, WindGust, WindScenario
from .wind_strategy import (
    WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION,
    DirectionalRisk,
    PerfectInformationCounterfactual,
    StrategyAnalysisConfig,
    StrategyAnalysisRequest,
    StrategyShotOutcome,
    StrategySummary,
    TargetPoint,
    WindStrategy,
    WindStrategyAnalysis,
    analyze_wind_strategies,
)
from .wind_uncertainty import (
    WIND_UNCERTAINTY_SCHEMA_VERSION,
    ScalarDistribution,
    WindEstimateError,
    WindTrial,
    WindUncertaintySpec,
    sample_wind_trials,
)

__all__ = [
    "DEFAULT_BACKSPIN_AXIS",
    "BallFlightModel",
    "ConstantCoefficientModel",
    "ConstantCoefficientSpec",
    "FlightModelRegistry",
    "FlightModelType",
    "FlightResult",
    "FlightSimulatorProtocol",
    "LaunchConditions",
    "MacDonaldHanzelyModel",
    "TrajectoryPoint",
    "WaterlooPennerModel",
    "WIND_SCHEMA_VERSION",
    "WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION",
    "WIND_UNCERTAINTY_SCHEMA_VERSION",
    "DirectionalRisk",
    "PerfectInformationCounterfactual",
    "ScalarDistribution",
    "StrategyAnalysisConfig",
    "StrategyAnalysisRequest",
    "StrategyShotOutcome",
    "StrategySummary",
    "TargetPoint",
    "WindGust",
    "WindScenario",
    "WindEstimateError",
    "WindStrategy",
    "WindStrategyAnalysis",
    "WindTrial",
    "WindUncertaintySpec",
    "analyze_wind_strategies",
    "compare_models",
    "compute_flight_metrics",
    "derive_launch_conditions",
    "from_flight_frame",
    "is_rust_available",
    "simulate",
    "simulate_trajectory_rust",
    "sample_wind_trials",
    "to_flight_frame",
]
