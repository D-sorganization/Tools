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
from .capability_contract import (
    CapabilityObjective,
    CapabilityParameter,
    ClubCapability,
    OptimizationAlternative,
    OptimizationRequest,
    OptimizationResult,
    PlayerCapabilityProfile,
    TargetDefinition,
)
from .capability_observation import (
    CAPABILITY_SAMPLE_OBSERVATION_SCHEMA_VERSION,
    CancellationCheck,
    CapabilityOptimizationCancelled,
    CapabilityOptimizationHooks,
    CapabilitySampleMetric,
    CapabilitySampleObservation,
    CapabilitySampleParameter,
    CapabilitySampleStatus,
    ObservationSink,
)
from .capability_optimizer import CapabilityEvaluator, optimize_capability
from .direction import (
    DEFINITIONS as LAUNCH_DIRECTION_DEFINITIONS,
)
from .direction import (
    LaunchDirection,
    LaunchDirectionConvention,
    launch_direction_from_mapping,
    launch_direction_sign_labels,
    launch_direction_to_flight_azimuth,
    migrate_launch_direction_mapping,
)
from .frames import from_flight_frame, to_flight_frame
from .ground_transfer import (
    FlightGroundTransferError,
    FlightGroundTransferSettings,
    build_ground_simulation_request,
    launch_relative_surface,
)
from .impact_solution_adapter import CenteredClubDeliveryAdapter
from .impact_solution_contract import (
    ClubProfileId,
    ForwardEvaluation,
    ForwardStatus,
    ImpactSolutionRequest,
    ImpactSolutionResult,
    ModelAvailability,
    ModelManifest,
)
from .impact_solution_solver import (
    ImpactForwardEvaluator,
    solve_impact_solution_families,
)
from .inverse_contract import (
    DecisionVariable,
    EvaluatedMetric,
    EvaluationStatus,
    FlightObjective,
    InverseFlightRequest,
    InverseFlightResult,
    ObjectiveMode,
    ObjectiveResidual,
    ParameterValue,
    SolutionCandidate,
    SolverEvaluation,
    SolverStatus,
)
from .inverse_solver import ForwardEvaluator, solve_inverse_flight
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
from .result_contract import (
    AvailabilityReason,
    FlightMetricCatalog,
    FlightMetricDefinition,
    FlightMetricId,
    ValueStatus,
    flight_metric_catalog,
)
from .result_metrics import (
    FlightMetricInputs,
    FlightMetricResult,
    FlightMetricValue,
    FlightRunManifest,
    GroundModelResult,
    MetricTrajectoryPoint,
    derive_flight_metric_result,
)
from .state import FlightStatePoint
from .surface_simulation import SurfaceFlightSimulationSettings
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
    "AvailabilityReason",
    "BallFlightModel",
    "CapabilityEvaluator",
    "CapabilityOptimizationCancelled",
    "CapabilityOptimizationHooks",
    "CapabilityObjective",
    "CapabilityParameter",
    "CapabilitySampleMetric",
    "CapabilitySampleObservation",
    "CapabilitySampleParameter",
    "CapabilitySampleStatus",
    "CAPABILITY_SAMPLE_OBSERVATION_SCHEMA_VERSION",
    "CancellationCheck",
    "ClubCapability",
    "ConstantCoefficientModel",
    "ConstantCoefficientSpec",
    "CenteredClubDeliveryAdapter",
    "ClubProfileId",
    "DecisionVariable",
    "EvaluatedMetric",
    "EvaluationStatus",
    "FlightObjective",
    "FlightModelRegistry",
    "FlightModelType",
    "FlightMetricCatalog",
    "FlightMetricDefinition",
    "FlightMetricId",
    "FlightMetricInputs",
    "FlightMetricResult",
    "FlightMetricValue",
    "FlightResult",
    "FlightGroundTransferError",
    "FlightGroundTransferSettings",
    "FlightStatePoint",
    "FlightRunManifest",
    "FlightSimulatorProtocol",
    "ForwardEvaluator",
    "ForwardEvaluation",
    "ForwardStatus",
    "GroundModelResult",
    "ImpactForwardEvaluator",
    "ImpactSolutionRequest",
    "ImpactSolutionResult",
    "InverseFlightRequest",
    "InverseFlightResult",
    "LaunchConditions",
    "LaunchDirection",
    "LaunchDirectionConvention",
    "LAUNCH_DIRECTION_DEFINITIONS",
    "MacDonaldHanzelyModel",
    "MetricTrajectoryPoint",
    "ModelAvailability",
    "ModelManifest",
    "ObjectiveMode",
    "ObjectiveResidual",
    "OptimizationAlternative",
    "OptimizationRequest",
    "OptimizationResult",
    "ObservationSink",
    "ParameterValue",
    "PlayerCapabilityProfile",
    "SolutionCandidate",
    "SolverEvaluation",
    "SolverStatus",
    "TargetDefinition",
    "TrajectoryPoint",
    "WaterlooPennerModel",
    "WIND_SCHEMA_VERSION",
    "WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION",
    "WIND_UNCERTAINTY_SCHEMA_VERSION",
    "DirectionalRisk",
    "PerfectInformationCounterfactual",
    "ScalarDistribution",
    "SurfaceFlightSimulationSettings",
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
    "build_ground_simulation_request",
    "launch_relative_surface",
    "compare_models",
    "compute_flight_metrics",
    "derive_launch_conditions",
    "derive_flight_metric_result",
    "flight_metric_catalog",
    "from_flight_frame",
    "is_rust_available",
    "launch_direction_from_mapping",
    "launch_direction_sign_labels",
    "launch_direction_to_flight_azimuth",
    "migrate_launch_direction_mapping",
    "optimize_capability",
    "simulate",
    "simulate_trajectory_rust",
    "solve_inverse_flight",
    "solve_impact_solution_families",
    "sample_wind_trials",
    "to_flight_frame",
    "ValueStatus",
]
