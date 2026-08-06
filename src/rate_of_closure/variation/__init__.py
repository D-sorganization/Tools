"""Rate-owned adapters between complete simulations and variation studies."""

from __future__ import annotations

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
    spatial_point_ids,
)

__all__ = [
    "APP_FRAME_ID",
    "CONTACT_OUTPUT_NAMES",
    "EVALUATED_HIT",
    "EVALUATED_NO_IMPACT",
    "IMPACT_OUTPUT_NAMES",
    "NUMERICAL_FAILURE",
    "SHOT_OUTPUT_NAMES",
    "SimulationEnsembleRequest",
    "SimulationEnsembleResult",
    "SimulationTrialOutcome",
    "TrialEvaluationStatus",
    "TEE_HEIGHT_VARIABLE_KEY",
    "apply_ball_setup_sample",
    "build_simulation_ensemble_request",
    "run_simulation_ensemble",
    "spatial_point_ids",
]
