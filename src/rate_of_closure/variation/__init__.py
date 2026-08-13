"""Rate-owned adapters between complete simulations and variation studies."""

from __future__ import annotations

from .ensemble_chunks import (
    CollectingEnsembleSink,
    EnsembleChunkSink,
    EnsembleStreamHeader,
    SimulationResultChunk,
)
from .morris_rate_adapter import (
    RATE_MORRIS_OUTPUTS,
    RATE_MORRIS_VARIABLE_KEYS,
    RateMorrisEvaluator,
    evaluate_rate_morris_design,
)
from .request_builder import apply_global_simulation_values
from .simulation_adapter import (
    APP_FRAME_ID,
    CONTACT_OUTPUT_NAMES,
    EVALUATED_HIT,
    EVALUATED_NO_IMPACT,
    IMPACT_OUTPUT_NAMES,
    NUMERICAL_FAILURE,
    SHOT_OUTPUT_NAMES,
    TEE_HEIGHT_VARIABLE_KEY,
    SimulationEnsembleRequest,
    SimulationEnsembleResult,
    SimulationTrialOutcome,
    TrialEvaluationStatus,
    apply_ball_setup_sample,
    build_simulation_ensemble_request,
    run_simulation_ensemble,
    run_simulation_ensemble_chunks,
    spatial_point_ids,
)
from .trial_projection import (
    SimulationExecutor,
    TrialCapture,
    capture_simulation,
    project_simulation_outcome,
)

__all__ = [
    "APP_FRAME_ID",
    "CONTACT_OUTPUT_NAMES",
    "CollectingEnsembleSink",
    "EVALUATED_HIT",
    "EVALUATED_NO_IMPACT",
    "EnsembleChunkSink",
    "EnsembleStreamHeader",
    "IMPACT_OUTPUT_NAMES",
    "NUMERICAL_FAILURE",
    "RATE_MORRIS_OUTPUTS",
    "RATE_MORRIS_VARIABLE_KEYS",
    "RateMorrisEvaluator",
    "SHOT_OUTPUT_NAMES",
    "SimulationEnsembleRequest",
    "SimulationEnsembleResult",
    "SimulationTrialOutcome",
    "SimulationExecutor",
    "SimulationResultChunk",
    "TrialCapture",
    "TrialEvaluationStatus",
    "TEE_HEIGHT_VARIABLE_KEY",
    "apply_ball_setup_sample",
    "apply_global_simulation_values",
    "build_simulation_ensemble_request",
    "capture_simulation",
    "evaluate_rate_morris_design",
    "project_simulation_outcome",
    "run_simulation_ensemble",
    "run_simulation_ensemble_chunks",
    "spatial_point_ids",
]
