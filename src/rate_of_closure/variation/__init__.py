"""Rate-owned adapters between complete simulations and variation studies."""

from __future__ import annotations

from .chip_forgiveness import ChipStudySummary, ChipTrialCohort
from .complete_trial_record import (
    COMPLETE_TRIAL_SCHEMA,
    CompleteTrialRecord,
    CompleteTrialRecordSource,
)
from .durable_ensemble_chunks import (
    DurableEnsembleArchive,
    DurableEnsembleChunkSink,
)
from .durable_ensemble_evidence import (
    DURABLE_ENSEMBLE_ANALYSIS_METHOD,
    DURABLE_ENSEMBLE_EVIDENCE_SCHEMA,
    DURABLE_ENSEMBLE_LIMITATIONS,
    DurableAnalysisEvidence,
    DurableArchiveEvidence,
    DurableEnsembleEvidence,
    durable_ensemble_evidence,
    durable_ensemble_evidence_from_document,
    durable_ensemble_evidence_from_json,
    durable_ensemble_evidence_to_document,
    durable_ensemble_evidence_to_json,
)
from .ensemble_chunks import (
    CollectingEnsembleSink,
    EnsembleChunkSink,
    EnsembleResumeState,
    EnsembleStreamHeader,
    ResumableEnsembleChunkSink,
    SimulationResultChunk,
)
from .ensemble_source import (
    EnsembleWorkChunk,
    LazySimulationEnsembleSource,
    SimulationEnsembleSource,
)
from .forgiveness_projection import forgiveness_variation_dataset
from .forgiveness_runner import ChipForgivenessStudy
from .morris_rate_adapter import (
    RATE_MORRIS_OUTPUTS,
    RATE_MORRIS_VARIABLE_KEYS,
    RateMorrisEvaluator,
    evaluate_rate_morris_design,
)
from .paired_attribution_adapter import (
    RATE_PAIRED_ATTRIBUTION_ADAPTER_ID,
    build_rate_paired_attribution_input,
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
    build_ensemble_stream_header,
    build_simulation_ensemble_request,
    run_simulation_ensemble,
    run_simulation_ensemble_chunks,
    spatial_point_ids,
    spatial_source_layouts,
)
from .streaming_ensemble_analysis import (
    AnalyzingDurableEnsembleSink,
    DurableEnsembleLayout,
    DurableEnsembleSummary,
    StreamingOutputMoments,
    analyze_durable_ensemble,
)
from .trial_projection import (
    SimulationExecutor,
    TrialCapture,
    capture_simulation,
    project_simulation_outcome,
)

__all__ = [
    "APP_FRAME_ID",
    "AnalyzingDurableEnsembleSink",
    "CONTACT_OUTPUT_NAMES",
    "ChipForgivenessStudy",
    "ChipStudySummary",
    "ChipTrialCohort",
    "CollectingEnsembleSink",
    "COMPLETE_TRIAL_SCHEMA",
    "CompleteTrialRecord",
    "CompleteTrialRecordSource",
    "DurableEnsembleArchive",
    "DurableAnalysisEvidence",
    "DurableArchiveEvidence",
    "DurableEnsembleEvidence",
    "DurableEnsembleLayout",
    "DurableEnsembleChunkSink",
    "DurableEnsembleSummary",
    "DURABLE_ENSEMBLE_ANALYSIS_METHOD",
    "DURABLE_ENSEMBLE_EVIDENCE_SCHEMA",
    "DURABLE_ENSEMBLE_LIMITATIONS",
    "EVALUATED_HIT",
    "EVALUATED_NO_IMPACT",
    "EnsembleChunkSink",
    "EnsembleResumeState",
    "EnsembleStreamHeader",
    "EnsembleWorkChunk",
    "IMPACT_OUTPUT_NAMES",
    "NUMERICAL_FAILURE",
    "RATE_MORRIS_OUTPUTS",
    "RATE_MORRIS_VARIABLE_KEYS",
    "RATE_PAIRED_ATTRIBUTION_ADAPTER_ID",
    "ResumableEnsembleChunkSink",
    "RateMorrisEvaluator",
    "SHOT_OUTPUT_NAMES",
    "SimulationEnsembleRequest",
    "SimulationEnsembleSource",
    "SimulationEnsembleResult",
    "SimulationExecutor",
    "SimulationResultChunk",
    "SimulationTrialOutcome",
    "StreamingOutputMoments",
    "LazySimulationEnsembleSource",
    "TEE_HEIGHT_VARIABLE_KEY",
    "TrialCapture",
    "TrialEvaluationStatus",
    "apply_ball_setup_sample",
    "apply_global_simulation_values",
    "analyze_durable_ensemble",
    "build_simulation_ensemble_request",
    "build_ensemble_stream_header",
    "build_rate_paired_attribution_input",
    "capture_simulation",
    "durable_ensemble_evidence",
    "durable_ensemble_evidence_from_document",
    "durable_ensemble_evidence_from_json",
    "durable_ensemble_evidence_to_document",
    "durable_ensemble_evidence_to_json",
    "evaluate_rate_morris_design",
    "forgiveness_variation_dataset",
    "project_simulation_outcome",
    "run_simulation_ensemble",
    "run_simulation_ensemble_chunks",
    "spatial_point_ids",
    "spatial_source_layouts",
]
