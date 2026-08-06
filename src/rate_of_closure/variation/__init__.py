"""Rate-owned adapters between complete simulations and variation studies."""

from __future__ import annotations

from .chip_forgiveness import (
    BinomialEstimate,
    ChipStudyMetadata,
    ChipStudySummary,
    ChipTrialCohort,
    ChipTrialRecord,
    ConvergencePoint,
    MetricDistribution,
    summarize_chip_trials,
)
from .forgiveness_io import (
    chip_forgiveness_study_to_csv,
    chip_forgiveness_study_to_dict,
    chip_forgiveness_study_to_json,
)
from .forgiveness_projection import forgiveness_variation_dataset
from .forgiveness_ranking import ChipCandidateScore, pareto_frontier
from .forgiveness_runner import (
    CHIP_METRIC_NAMES,
    ChipForgivenessRequest,
    ChipForgivenessStudy,
    ChipLossModel,
    analyze_chip_forgiveness_ensemble,
    run_chip_forgiveness_study,
)
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
    "BinomialEstimate",
    "CHIP_METRIC_NAMES",
    "CONTACT_OUTPUT_NAMES",
    "ChipCandidateScore",
    "ChipForgivenessRequest",
    "ChipForgivenessStudy",
    "ChipLossModel",
    "ChipStudyMetadata",
    "ChipStudySummary",
    "ChipTrialCohort",
    "ChipTrialRecord",
    "ConvergencePoint",
    "EVALUATED_HIT",
    "EVALUATED_NO_IMPACT",
    "IMPACT_OUTPUT_NAMES",
    "MetricDistribution",
    "NUMERICAL_FAILURE",
    "SHOT_OUTPUT_NAMES",
    "SimulationEnsembleRequest",
    "SimulationEnsembleResult",
    "SimulationTrialOutcome",
    "TrialEvaluationStatus",
    "TEE_HEIGHT_VARIABLE_KEY",
    "apply_ball_setup_sample",
    "analyze_chip_forgiveness_ensemble",
    "build_simulation_ensemble_request",
    "chip_forgiveness_study_to_csv",
    "chip_forgiveness_study_to_dict",
    "chip_forgiveness_study_to_json",
    "forgiveness_variation_dataset",
    "pareto_frontier",
    "run_simulation_ensemble",
    "run_chip_forgiveness_study",
    "spatial_point_ids",
    "summarize_chip_trials",
]
