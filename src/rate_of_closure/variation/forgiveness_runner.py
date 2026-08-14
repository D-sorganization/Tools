"""Execute complete Rate simulations as qualified chip-forgiveness studies."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from rate_of_closure.simulation import (
    SimulationConfig,
    SimulationRun,
    impact_kinematics_for_run,
    run_simulation,
    turf_interaction_for_run,
)
from rate_of_closure.variation.chip_forgiveness import (
    ChipStudyMetadata,
    ChipStudySummary,
    ChipTrialCohort,
    ChipTrialRecord,
    summarize_chip_trials,
)
from rate_of_closure.variation.forgiveness_loss import ChipLossModel
from rate_of_closure.variation.simulation_types import (
    NUMERICAL_FAILURE,
    SimulationEnsembleRequest,
    SimulationEnsembleResult,
    SimulationTrialOutcome,
)
from shared.python.contracts import ContractViolationError
from shared.python.golf_club import (
    ContactSequence,
    GroundPlane,
    TurfCalibrationStatus,
    TurfContactProfile,
    WedgeHeadParameters,
)
from shared.python.swing_sim.solver.solve import (
    CancelledError,
    ProgressCallback,
    ProgressReport,
)
from shared.python.swing_sim.variation.spec import SCHEMA_VERSION, VariationPlan

_APP_FRAME_ID = "app_frame:x_target,y_up,z_right"
_SOLVER_ID = "rate-of-closure/canonical"
_FAILURE_TYPES = (
    TypeError,
    ValueError,
    ContractViolationError,
    RuntimeError,
    FloatingPointError,
    OverflowError,
)

CHIP_METRIC_NAMES: tuple[str, ...] = (
    "carry_m",
    "lateral_m",
    "max_height_m",
    "landing_angle_deg",
    "leading_edge_clearance_at_ball_m",
    "minimum_pre_ball_clearance_m",
    "ground_after_ball_margin_s",
    "low_point_clearance_m",
    "delivered_bounce_deg",
    "path_projected_effective_bounce_deg",
    "reference_aoa_deg",
    "bounce_utilization_margin_deg",
    "peak_turf_penetration_m",
    "normal_turf_impulse_n_s",
    "shaft_rotation_rate_rad_s",
    "shaft_counterfactual_aoa_delta_deg",
    "shaft_shapley_aoa_deg",
    "shaft_vertical_velocity_share",
    "leading_edge_3d_rate_rad_s",
    "face_normal_3d_rate_rad_s",
    "leading_edge_relative_arc_heading_rate_rad_s",
)


@dataclass(frozen=True)
class ChipForgivenessRequest:
    """Complete immutable physics, population, objective, and reporting request."""

    candidate_id: str
    ensemble: SimulationEnsembleRequest
    wedge_parameters: WedgeHeadParameters
    ground: GroundPlane
    turf_profile: TurfContactProfile
    loss_model: ChipLossModel = field(default_factory=ChipLossModel)
    cvar_tail_fraction: float = 0.1
    bootstrap_samples: int = 2_000

    def __post_init__(self) -> None:
        """Validate owning contract types before any expensive execution."""
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a nonempty string")
        expected_types = (
            (self.ensemble, SimulationEnsembleRequest, "ensemble"),
            (self.wedge_parameters, WedgeHeadParameters, "wedge_parameters"),
            (self.ground, GroundPlane, "ground"),
            (self.turf_profile, TurfContactProfile, "turf_profile"),
            (self.loss_model, ChipLossModel, "loss_model"),
        )
        for value, expected, name in expected_types:
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if (
            self.loss_model.include_turf_penetration
            and self.turf_profile.calibration_status
            is not TurfCalibrationStatus.CALIBRATED
        ):
            raise ValueError(
                "turf penetration may enter the loss only for a calibrated profile"
            )


@dataclass(frozen=True)
class ChipForgivenessStudy:
    """Retained records, sampled population, and qualified decision summary."""

    records: tuple[ChipTrialRecord, ...]
    summary: ChipStudySummary
    plan: VariationPlan
    input_names: tuple[str, ...]
    sampled_inputs: np.ndarray = field(repr=False)
    request: ChipForgivenessRequest = field(repr=False)

    def __post_init__(self) -> None:
        """Freeze the sampled population used to produce the records."""
        samples = np.array(self.sampled_inputs, dtype=float, copy=True)
        if not isinstance(self.plan, VariationPlan):
            raise TypeError("plan must be a VariationPlan")
        if not isinstance(self.request, ChipForgivenessRequest):
            raise TypeError("request must be a ChipForgivenessRequest")
        expected = (len(self.records), len(self.input_names))
        if samples.shape != expected:
            raise ValueError("sampled_inputs must align with records and input_names")
        samples.setflags(write=False)
        object.__setattr__(self, "sampled_inputs", samples)


SimulationExecutor = Callable[[SimulationConfig], SimulationRun]


def _cohort(sequence: ContactSequence) -> ChipTrialCohort:
    return {
        ContactSequence.BALL_FIRST: ChipTrialCohort.BALL_FIRST,
        ContactSequence.BALL_ONLY: ChipTrialCohort.BALL_ONLY,
        ContactSequence.GROUND_FIRST: ChipTrialCohort.GROUND_FIRST,
        ContactSequence.SIMULTANEOUS: ChipTrialCohort.SIMULTANEOUS,
        ContactSequence.GROUND_ONLY_MISS: ChipTrialCohort.GROUND_ONLY_MISS,
        ContactSequence.NO_CONTACT_MISS: ChipTrialCohort.NO_CONTACT_MISS,
    }[sequence]


def _empty_metrics() -> dict[str, float | None]:
    return dict.fromkeys(CHIP_METRIC_NAMES)


def _trial_metrics(
    run: SimulationRun, request: ChipForgivenessRequest
) -> tuple[ChipTrialCohort, dict[str, float | None], str | None]:
    turf = turf_interaction_for_run(
        run, request.wedge_parameters, request.ground, request.turf_profile
    )
    ground = turf.ground_clearance.analysis
    impact = impact_kinematics_for_run(run).analysis
    metrics = _empty_metrics()
    if run.launch is not None:
        metrics.update(
            carry_m=run.launch["carry_m"],
            lateral_m=float(run.flight_positions[-1, 2]),
            max_height_m=run.launch["max_height_m"],
            landing_angle_deg=run.launch["landing_angle_deg"],
        )
    ground_point = np.asarray(request.ground.point_m)
    ground_normal = np.asarray(request.ground.normal_unit)
    low_point_clearance = float(
        (np.asarray(ground.low_point_world_m) - ground_point) @ ground_normal
    )
    reduced = turf.reduced_contact
    metrics.update(
        leading_edge_clearance_at_ball_m=ground.leading_edge_clearance_at_ball_m,
        minimum_pre_ball_clearance_m=ground.minimum_pre_ball_clearance_m,
        ground_after_ball_margin_s=ground.ground_after_ball_time_margin_s,
        low_point_clearance_m=low_point_clearance,
        delivered_bounce_deg=ground.delivered_bounce_deg_at_ball,
        path_projected_effective_bounce_deg=(
            ground.path_projected_effective_bounce_deg_at_ball
        ),
        reference_aoa_deg=ground.reference_aoa_deg_at_ball,
        bounce_utilization_margin_deg=ground.bounce_utilization_margin_deg,
        peak_turf_penetration_m=(
            None if reduced is None else reduced.peak_penetration_m
        ),
        normal_turf_impulse_n_s=(
            None if reduced is None else reduced.normal_impulse_n_s
        ),
        shaft_rotation_rate_rad_s=impact.shaft_rotation_rate_rad_s,
        shaft_counterfactual_aoa_delta_deg=(impact.shaft_counterfactual_aoa_delta_deg),
        shaft_shapley_aoa_deg=impact.shaft_shapley_aoa_deg,
        shaft_vertical_velocity_share=impact.shaft_vertical_velocity_share,
        leading_edge_3d_rate_rad_s=impact.leading_edge_3d_rate_rad_s,
        face_normal_3d_rate_rad_s=impact.face_normal_3d_rate_rad_s,
        leading_edge_relative_arc_heading_rate_rad_s=(
            impact.leading_edge_relative_arc_heading_rate_rad_s
        ),
    )
    turf_status = None if reduced is None else reduced.status.value
    return _cohort(ground.sequence), metrics, turf_status


def _failure_record(
    index: int, error: Exception, model: ChipLossModel
) -> ChipTrialRecord:
    metrics = _empty_metrics()
    loss, violated = model.evaluate(ChipTrialCohort.NUMERICAL_FAILURE, metrics)
    return ChipTrialRecord(
        index,
        ChipTrialCohort.NUMERICAL_FAILURE,
        loss,
        violated,
        metrics,
        f"{type(error).__name__}: {error}",
    )


def _metadata(request: ChipForgivenessRequest) -> ChipStudyMetadata:
    plan = request.ensemble.plan
    return ChipStudyMetadata(
        candidate_id=request.candidate_id,
        plan_schema=f"swing-sim.variation-plan/v{SCHEMA_VERSION}",
        coordinate_frame=_APP_FRAME_ID,
        seed=plan.seed,
        noise_model_id="+".join(
            spec.spec_id or spec.variable_key for spec in plan.noise
        ),
        objective_id=request.loss_model.objective_id,
        turf_profile_id=request.turf_profile.profile_id,
        turf_calibration_status=request.turf_profile.calibration_status.value,
        solver_id=_SOLVER_ID,
        sampling_design="iid-monte-carlo-joint",
        inference_method_id="wilson+mulberry32-iid-bootstrap-v1",
        limitations=(
            "Conditional on the retained plan and objective. Turf response is a "
            "reduced diagnostic and does not replay the swing under turf force."
        ),
    )


def _study(
    request: ChipForgivenessRequest, records: tuple[ChipTrialRecord, ...]
) -> ChipForgivenessStudy:
    """Build one immutable result from an already evaluated canonical record set."""
    summary = summarize_chip_trials(
        _metadata(request),
        records,
        cvar_tail_fraction=request.cvar_tail_fraction,
        bootstrap_samples=request.bootstrap_samples,
    )
    return ChipForgivenessStudy(
        records=records,
        summary=summary,
        plan=request.ensemble.plan,
        input_names=tuple(spec.variable_key for spec in request.ensemble.plan.noise),
        sampled_inputs=request.ensemble.sampled_inputs,
        request=request,
    )


def _outcome_failure_record(
    outcome: SimulationTrialOutcome, model: ChipLossModel
) -> ChipTrialRecord:
    """Retain an ensemble failure's original type and message without a rerun."""
    metrics = _empty_metrics()
    loss, violated = model.evaluate(ChipTrialCohort.NUMERICAL_FAILURE, metrics)
    failure_type = outcome.failure_type or "NumericalFailure"
    failure_message = outcome.failure_message or ""
    return ChipTrialRecord(
        outcome.trial_index,
        ChipTrialCohort.NUMERICAL_FAILURE,
        loss,
        violated,
        metrics,
        f"{failure_type}: {failure_message}",
    )


def analyze_chip_forgiveness_ensemble(
    request: ChipForgivenessRequest,
    result: SimulationEnsembleResult,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> ChipForgivenessStudy:
    """Analyze retained ensemble runs without executing the physics a second time."""
    if not isinstance(request, ChipForgivenessRequest):
        raise TypeError("request must be ChipForgivenessRequest")
    if not isinstance(result, SimulationEnsembleResult):
        raise TypeError("result must be SimulationEnsembleResult")
    if not result.runs:
        raise ValueError("result must retain runs for forgiveness analysis")
    if result.variation.plan != request.ensemble.plan:
        raise ValueError("result plan must match the forgiveness request")
    cancellation = cancel_event or threading.Event()
    started = time.monotonic()
    records: list[ChipTrialRecord] = []
    failures = 0
    for outcome, run in zip(result.outcomes, result.runs, strict=True):
        if cancellation.is_set():
            raise CancelledError
        if outcome.status is NUMERICAL_FAILURE:
            failures += 1
            records.append(_outcome_failure_record(outcome, request.loss_model))
        else:
            assert run is not None
            try:
                cohort, metrics, turf_status = _trial_metrics(run, request)
                loss, violated = request.loss_model.evaluate(
                    cohort, metrics, turf_contact_status=turf_status
                )
                records.append(
                    ChipTrialRecord(
                        outcome.trial_index,
                        cohort,
                        loss,
                        violated,
                        metrics,
                        turf_contact_status=turf_status,
                    )
                )
            except CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - retain post-process failures
                failures += 1
                records.append(
                    _failure_record(outcome.trial_index, error, request.loss_model)
                )
        if progress_cb is not None:
            progress_cb(
                ProgressReport(
                    iteration=len(records),
                    cost=float(failures),
                    best_cost=0.0,
                    improvement_pct=0.0,
                    elapsed_s=time.monotonic() - started,
                )
            )
    return _study(request, tuple(records))


def run_chip_forgiveness_study(
    request: ChipForgivenessRequest,
    executor: SimulationExecutor = run_simulation,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> ChipForgivenessStudy:
    """Execute every configured trial, retaining hits, misses, and failures."""
    if not isinstance(request, ChipForgivenessRequest):
        raise TypeError("request must be ChipForgivenessRequest")
    if not callable(executor):
        raise TypeError("executor must be callable")
    cancellation = cancel_event or threading.Event()
    started = time.monotonic()
    records: list[ChipTrialRecord] = []
    failures = 0
    for index, config in enumerate(request.ensemble.configs):
        if cancellation.is_set():
            raise CancelledError
        try:
            run = executor(config)
            if not isinstance(run, SimulationRun):
                raise TypeError("executor must return SimulationRun")
            cohort, metrics, turf_status = _trial_metrics(run, request)
            loss, violated = request.loss_model.evaluate(
                cohort, metrics, turf_contact_status=turf_status
            )
            record = ChipTrialRecord(
                index,
                cohort,
                loss,
                violated,
                metrics,
                turf_contact_status=turf_status,
            )
        except _FAILURE_TYPES as error:
            failures += 1
            record = _failure_record(index, error, request.loss_model)
        records.append(record)
        if progress_cb is not None:
            progress_cb(
                ProgressReport(
                    iteration=index + 1,
                    cost=float(failures),
                    best_cost=0.0,
                    improvement_pct=0.0,
                    elapsed_s=time.monotonic() - started,
                )
            )
    return _study(request, tuple(records))


__all__ = [
    "CHIP_METRIC_NAMES",
    "ChipForgivenessRequest",
    "ChipForgivenessStudy",
    "ChipLossModel",
    "analyze_chip_forgiveness_ensemble",
    "run_chip_forgiveness_study",
]
