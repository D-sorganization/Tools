"""Contract test pinning the public API surface of swing_sim.flight.

Downstream consumers (impact stage #4106, UpstreamDrift, web backend)
import from the subpackage facade only; this test fails loudly when the
surface changes so removals are always deliberate.
"""

from __future__ import annotations

import dataclasses

import pytest

import shared.python.swing_sim.flight as flight

EXPECTED_PUBLIC_API = {
    "DEFAULT_BACKSPIN_AXIS",
    "AvailabilityReason",
    "BallFlightModel",
    "CAPABILITY_SAMPLE_OBSERVATION_SCHEMA_VERSION",
    "CancellationCheck",
    "CapabilityEvaluator",
    "CapabilityOptimizationCancelled",
    "CapabilityOptimizationHooks",
    "CapabilityObjective",
    "CapabilityParameter",
    "CapabilitySampleMetric",
    "CapabilitySampleObservation",
    "CapabilitySampleParameter",
    "CapabilitySampleStatus",
    "ClubCapability",
    "ConstantCoefficientModel",
    "ConstantCoefficientSpec",
    "CenteredClubDeliveryAdapter",
    "ClubProfileId",
    "DecisionVariable",
    "DirectionalRisk",
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
    "PerfectInformationCounterfactual",
    "SolutionCandidate",
    "SolverEvaluation",
    "SolverStatus",
    "ScalarDistribution",
    "SurfaceFlightSimulationSettings",
    "StrategyAnalysisConfig",
    "StrategyAnalysisRequest",
    "StrategyShotOutcome",
    "StrategySummary",
    "TargetPoint",
    "TargetDefinition",
    "TrajectoryPoint",
    "WIND_SCHEMA_VERSION",
    "WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION",
    "WIND_UNCERTAINTY_SCHEMA_VERSION",
    "WaterlooPennerModel",
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
}


@pytest.mark.contract
def test_public_api_surface_is_pinned() -> None:
    assert set(flight.__all__) == EXPECTED_PUBLIC_API


@pytest.mark.contract
def test_all_exports_resolve() -> None:
    for name in flight.__all__:
        assert getattr(flight, name) is not None, f"{name} did not resolve"


@pytest.mark.contract
def test_swing_sim_top_level_facade_unchanged() -> None:
    """The flight port must not widen the parent swing_sim facade (#4104)."""
    import shared.python.swing_sim as swing_sim

    assert "flight" not in swing_sim.__all__


FROZEN_VALUE_TYPES = (
    flight.LaunchConditions,
    flight.TrajectoryPoint,
    flight.FlightResult,
    flight.FlightGroundTransferSettings,
    flight.FlightStatePoint,
    flight.SurfaceFlightSimulationSettings,
    flight.ConstantCoefficientSpec,
    flight.LaunchDirection,
    flight.FlightMetricDefinition,
    flight.FlightMetricCatalog,
    flight.MetricTrajectoryPoint,
    flight.FlightMetricInputs,
    flight.FlightMetricValue,
    flight.FlightMetricResult,
    flight.FlightRunManifest,
    flight.GroundModelResult,
    flight.DecisionVariable,
    flight.FlightObjective,
    flight.InverseFlightRequest,
    flight.EvaluatedMetric,
    flight.SolverEvaluation,
    flight.ParameterValue,
    flight.ObjectiveResidual,
    flight.SolutionCandidate,
    flight.InverseFlightResult,
    flight.ImpactSolutionRequest,
    flight.ForwardEvaluation,
    flight.ModelManifest,
    flight.ImpactSolutionResult,
    flight.WindGust,
    flight.WindScenario,
    flight.ScalarDistribution,
    flight.WindEstimateError,
    flight.WindTrial,
    flight.WindUncertaintySpec,
    flight.DirectionalRisk,
    flight.PerfectInformationCounterfactual,
    flight.TargetPoint,
    flight.WindStrategy,
    flight.StrategyAnalysisConfig,
    flight.StrategyAnalysisRequest,
    flight.StrategyShotOutcome,
    flight.StrategySummary,
    flight.WindStrategyAnalysis,
    flight.CapabilityParameter,
    flight.ClubCapability,
    flight.PlayerCapabilityProfile,
    flight.TargetDefinition,
    flight.OptimizationRequest,
    flight.OptimizationAlternative,
    flight.OptimizationResult,
    flight.CapabilityOptimizationHooks,
    flight.CapabilitySampleMetric,
    flight.CapabilitySampleObservation,
    flight.CapabilitySampleParameter,
)


@pytest.mark.contract
def test_value_types_are_frozen_dataclasses() -> None:
    for cls in FROZEN_VALUE_TYPES:
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} not a dataclass"
        assert cls.__dataclass_params__.frozen, f"{cls.__name__} must be frozen"


@pytest.mark.contract
def test_model_type_values_are_pinned() -> None:
    assert {m.value for m in flight.FlightModelType} == {
        "waterloo_penner",
        "macdonald_hanzely",
        "nathan",
        "ballantyne",
        "jcole",
        "rospie_dl",
        "charry_l3",
    }
