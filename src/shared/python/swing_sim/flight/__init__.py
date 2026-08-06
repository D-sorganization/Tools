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
from .types import (
    DEFAULT_BACKSPIN_AXIS,
    FlightResult,
    LaunchConditions,
    TrajectoryPoint,
    compute_flight_metrics,
)

__all__ = [
    "DEFAULT_BACKSPIN_AXIS",
    "AvailabilityReason",
    "BallFlightModel",
    "ConstantCoefficientModel",
    "ConstantCoefficientSpec",
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
    "FlightRunManifest",
    "FlightSimulatorProtocol",
    "ForwardEvaluator",
    "GroundModelResult",
    "InverseFlightRequest",
    "InverseFlightResult",
    "LaunchConditions",
    "LaunchDirection",
    "LaunchDirectionConvention",
    "LAUNCH_DIRECTION_DEFINITIONS",
    "MacDonaldHanzelyModel",
    "MetricTrajectoryPoint",
    "ObjectiveMode",
    "ObjectiveResidual",
    "ParameterValue",
    "SolutionCandidate",
    "SolverEvaluation",
    "SolverStatus",
    "TrajectoryPoint",
    "WaterlooPennerModel",
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
    "simulate",
    "simulate_trajectory_rust",
    "solve_inverse_flight",
    "to_flight_frame",
    "ValueStatus",
]
