"""Lossless, source-neutral records for one governed ensemble trial.

The record is evidence plumbing, not a new mechanics model.  It owns the
complete arrays returned by :class:`~rate_of_closure.simulation.SimulationRun`
and preserves explicit absence for misses and numerical failures.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from rate_of_closure.simulation import SimulationConfig, SimulationRun
from shared.python.contracts import require
from shared.python.swing_sim.run_config import DOUBLE_PENDULUM_JOINT_IDS
from shared.python.swing_sim.variation.execution_metadata import (
    VariationExecutionMetadata,
    execution_document_to_json_dict,
    make_execution_metadata,
)
from shared.python.swing_sim.variation.execution_provenance import (
    PYTHON_DEFAULT_PROVENANCE,
)

from ._complete_trial_fields import COMPLETE_TRIAL_UNITS, CommonFields, PhaseFields
from ._complete_trial_state import (
    DELIVERY_FIELDS,
    IMPACT_FIELDS,
    LAUNCH_FIELDS,
    POST_IMPACT_FIELDS,
    state_mapping,
)
from ._simulation_config_identity import simulation_configuration_stream_sha256
from .locus_execution_capabilities import load_locus_execution_contract
from .simulation_types import (
    EVALUATED_HIT,
    EVALUATED_NO_IMPACT,
    NUMERICAL_FAILURE,
    SimulationTrialOutcome,
    TrialEvaluationStatus,
)
from .trial_projection import TrialCapture

if TYPE_CHECKING:
    from .ensemble_chunks import EnsembleStreamHeader

COMPLETE_TRIAL_SCHEMA = "rate-complete-trial/v1"
INTERPOLATION_STATUSES = ("exact_sample", "nearest_sample_only", "unavailable")


def _owned_array(value: object, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    """Own one finite real array with an exact trailing shape."""
    raw = np.asarray(value)
    require(
        raw.ndim == 1 + len(shape_tail) and raw.shape[1:] == shape_tail,
        f"{name} has invalid shape",
        raw.shape,
    )
    require(
        np.issubdtype(raw.dtype, np.number)
        and not np.issubdtype(raw.dtype, np.bool_)
        and not np.issubdtype(raw.dtype, np.complexfloating),
        f"{name} must contain real numbers",
    )
    result = cast(np.ndarray, np.array(raw, dtype=float, copy=True))
    require(bool(np.all(np.isfinite(result))), f"{name} must be finite")
    result.setflags(write=False)
    return result


def _owned_vector(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    require(raw.ndim == 1, f"{name} must be one-dimensional", raw.shape)
    return _owned_array(raw[:, np.newaxis], (1,), name).reshape(raw.shape)


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class CompleteTrialRecordSource:
    """Inputs and configuration that identify one canonical trial row."""

    trial_index: int
    sampled_inputs: np.ndarray
    config: SimulationConfig

    def __post_init__(self) -> None:
        require(
            type(self.trial_index) is int and self.trial_index >= 0,
            "trial_index must be a non-negative integer",
        )
        require(
            isinstance(self.config, SimulationConfig), "config must be SimulationConfig"
        )
        inputs = _owned_vector(self.sampled_inputs, "sampled_inputs")
        object.__setattr__(self, "sampled_inputs", inputs)


@dataclass(frozen=True, slots=True)
class CompleteTrialRecord:
    """Complete immutable model output for one attempted ensemble trial."""

    trial_index: int
    status: TrialEvaluationStatus
    sampled_inputs: np.ndarray
    plan_sha256: str
    execution_sha256: str
    stream_configuration_sha256: str
    configuration_sha256: str
    sampled_input_sha256: str
    registry_sha256: str
    adapter_ids: tuple[str, ...]
    source_repository: str
    source_revision: str | None
    source_revision_status: str
    source_revision_reason: str | None
    source_kind: str
    coordinate_frame: str
    spatial_point_ids: tuple[str, ...]
    torque_joint_ids: tuple[str, ...]
    units: Mapping[str, str]
    candidate_time_s: float | None
    impact_time_s: float | None
    event_sample_index: int | None
    event_interpolation_status: str
    pre_impact_sample_count: int
    failure_type: str | None
    failure_message: str | None
    swing_times_s: np.ndarray
    swing_positions_m: np.ndarray
    swing_poses: np.ndarray
    swing_twists: np.ndarray
    swing_joint_positions_m: np.ndarray
    swing_applied_torques_nm: np.ndarray
    impact_outcome: Mapping[str, object] | None
    delivery_state: Mapping[str, object] | None
    post_impact_state: Mapping[str, object] | None
    launch_state: Mapping[str, object] | None
    flight_times_s: np.ndarray
    flight_positions_m: np.ndarray
    flight_velocities_mps: np.ndarray
    schema_version: str = COMPLETE_TRIAL_SCHEMA

    def __post_init__(self) -> None:
        """Validate dimensions, identities, and phase availability."""
        _validate_record_identity(self)
        _validate_record_arrays(self)
        _validate_record_states(self)
        _validate_record_phases(self)


def _validate_digest(value: str, name: str) -> None:
    require(
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef"),
        f"{name} must be a lowercase SHA-256 digest",
    )


def _validate_record_identity(record: CompleteTrialRecord) -> None:
    require(record.schema_version == COMPLETE_TRIAL_SCHEMA, "unsupported record schema")
    require(isinstance(record.status, TrialEvaluationStatus), "invalid trial status")
    for name in (
        "plan_sha256",
        "execution_sha256",
        "stream_configuration_sha256",
        "configuration_sha256",
        "sampled_input_sha256",
        "registry_sha256",
    ):
        _validate_digest(cast(str, getattr(record, name)), name)
    require(bool(record.source_kind), "source_kind must be non-empty")
    require(bool(record.coordinate_frame), "coordinate_frame must be non-empty")
    require(dict(record.units) == dict(COMPLETE_TRIAL_UNITS), "trial units are invalid")
    object.__setattr__(record, "units", COMPLETE_TRIAL_UNITS)
    require(
        record.event_interpolation_status in INTERPOLATION_STATUSES,
        "unsupported event interpolation status",
    )
    inputs = _owned_vector(record.sampled_inputs, "sampled_inputs")
    require(
        _sha256_array(inputs) == record.sampled_input_sha256,
        "sampled input digest mismatch",
    )
    object.__setattr__(record, "sampled_inputs", inputs)


def _validate_record_arrays(record: CompleteTrialRecord) -> None:
    points = 0 if record.source_kind == "manual" else len(record.spatial_point_ids)
    torques = len(record.torque_joint_ids)
    times = _owned_vector(record.swing_times_s, "swing_times_s")
    sample_count = len(times)
    positions = _owned_array(record.swing_positions_m, (3,), "swing_positions_m")
    poses = _owned_array(record.swing_poses, (4, 4), "swing_poses")
    twists = _owned_array(record.swing_twists, (6,), "swing_twists")
    joints = _owned_array(
        record.swing_joint_positions_m, (points, 3), "swing_joint_positions_m"
    )
    applied = _owned_array(
        record.swing_applied_torques_nm, (torques,), "swing_applied_torques_nm"
    )
    require(
        all(
            len(value) == sample_count
            for value in (positions, poses, twists, joints, applied)
        ),
        "swing arrays must share one sample count",
    )
    if sample_count:
        require(bool(np.all(np.diff(times) > 0.0)), "swing_times_s must increase")
    flight_times = _owned_vector(record.flight_times_s, "flight_times_s")
    flight_positions = _owned_array(
        record.flight_positions_m, (3,), "flight_positions_m"
    )
    flight_velocities = _owned_array(
        record.flight_velocities_mps, (3,), "flight_velocities_mps"
    )
    require(
        len(flight_times) == len(flight_positions) == len(flight_velocities),
        "flight arrays must share one sample count",
    )
    if len(flight_times):
        require(
            bool(np.all(np.diff(flight_times) > 0.0)), "flight_times_s must increase"
        )
    for name, value in (
        ("swing_times_s", times),
        ("swing_positions_m", positions),
        ("swing_poses", poses),
        ("swing_twists", twists),
        ("swing_joint_positions_m", joints),
        ("swing_applied_torques_nm", applied),
        ("flight_times_s", flight_times),
        ("flight_positions_m", flight_positions),
        ("flight_velocities_mps", flight_velocities),
    ):
        object.__setattr__(record, name, value)


def _validate_record_states(record: CompleteTrialRecord) -> None:
    for name, expected in (
        ("impact_outcome", IMPACT_FIELDS),
        ("delivery_state", DELIVERY_FIELDS),
        ("post_impact_state", POST_IMPACT_FIELDS),
        ("launch_state", LAUNCH_FIELDS),
    ):
        value = getattr(record, name)
        if value is not None:
            object.__setattr__(record, name, state_mapping(value, expected, name))


def _validate_record_phases(record: CompleteTrialRecord) -> None:
    if record.status is NUMERICAL_FAILURE:
        require(len(record.swing_times_s) == 0, "failure cannot contain swing physics")
        require(
            len(record.flight_times_s) == 0, "failure cannot contain flight physics"
        )
        require(record.failure_type is not None, "failure requires failure_type")
        require(record.failure_message is not None, "failure requires failure_message")
        require(
            all(
                value is None
                for value in (
                    record.candidate_time_s,
                    record.impact_time_s,
                    record.event_sample_index,
                    record.impact_outcome,
                    record.delivery_state,
                    record.post_impact_state,
                    record.launch_state,
                )
            ),
            "failure cannot contain fabricated event or downstream state",
        )
        require(
            record.pre_impact_sample_count == 0, "failure pre-impact count must be zero"
        )
        return
    require(
        record.failure_type is None and record.failure_message is None,
        "evaluated trial cannot carry failure metadata",
    )
    require(
        record.candidate_time_s is not None, "evaluated trial requires candidate time"
    )
    require(
        record.event_sample_index is not None, "evaluated trial requires event index"
    )
    require(
        0 < record.pre_impact_sample_count <= len(record.swing_times_s),
        "evaluated trial requires a bounded pre-impact prefix",
    )
    require(record.impact_outcome is not None, "evaluated trial requires contact state")
    is_hit = record.status is EVALUATED_HIT
    require(
        (record.impact_time_s is not None) == is_hit,
        "impact time must agree with status",
    )
    for name in ("delivery_state", "post_impact_state", "launch_state"):
        require(
            (getattr(record, name) is not None) == is_hit,
            f"{name} must agree with status",
        )
    require(
        (len(record.flight_times_s) > 0) == is_hit,
        "flight availability must agree with status",
    )
    require(
        record.status in {EVALUATED_HIT, EVALUATED_NO_IMPACT},
        "unsupported evaluated status",
    )


def build_complete_trial_record(
    source: CompleteTrialRecordSource,
    capture: TrialCapture,
    outcome: SimulationTrialOutcome,
    header: EnsembleStreamHeader,
) -> CompleteTrialRecord:
    """Bind one capture to its sampled row and immutable stream identity."""
    require(isinstance(source, CompleteTrialRecordSource), "invalid record source")
    require(isinstance(capture, TrialCapture), "capture must be TrialCapture")
    require(isinstance(outcome, SimulationTrialOutcome), "invalid trial outcome")
    require(source.trial_index == outcome.trial_index, "trial identity mismatch")
    metadata = make_execution_metadata(header.plan)
    execution = execution_document_to_json_dict(header.plan)
    adapters = _adapter_ids(header)
    common = _common_fields(source, outcome, header, metadata, execution, adapters)
    if capture.run is None:
        return CompleteTrialRecord(**common, **_failure_fields(source, outcome, header))
    return CompleteTrialRecord(**common, **_run_fields(capture.run))


def _adapter_ids(header: EnsembleStreamHeader) -> tuple[str, ...]:
    contract = load_locus_execution_contract()
    keys = {item.variable_key for item in header.plan.noise} | set(
        header.plan.base_variables
    )
    adapters = {
        contract.capabilities[key].adapter_id
        for key in keys
        if contract.capabilities[key].adapter_id is not None
    }
    return tuple(sorted(cast(set[str], adapters)))


def _common_fields(
    source: CompleteTrialRecordSource,
    outcome: SimulationTrialOutcome,
    header: EnsembleStreamHeader,
    metadata: VariationExecutionMetadata,
    execution: dict[str, object],
    adapters: tuple[str, ...],
) -> CommonFields:
    registry_sha256 = metadata.registry_sha256
    return {
        "trial_index": source.trial_index,
        "status": outcome.status,
        "sampled_inputs": source.sampled_inputs,
        "plan_sha256": metadata.plan_sha256,
        "execution_sha256": _sha256_json(execution),
        "stream_configuration_sha256": header.configuration_sha256,
        "configuration_sha256": simulation_configuration_stream_sha256(
            (source.config,), count=1
        ),
        "sampled_input_sha256": _sha256_array(source.sampled_inputs),
        "registry_sha256": registry_sha256,
        "adapter_ids": adapters,
        "source_repository": PYTHON_DEFAULT_PROVENANCE.source_repository,
        "source_revision": PYTHON_DEFAULT_PROVENANCE.source_revision,
        "source_revision_status": PYTHON_DEFAULT_PROVENANCE.source_revision_status,
        "source_revision_reason": PYTHON_DEFAULT_PROVENANCE.source_revision_reason,
        "source_kind": source.config.source_kind,
        "coordinate_frame": header.coordinate_frame,
        "spatial_point_ids": header.point_ids,
        "torque_joint_ids": _torque_ids(source.config.source_kind),
        "units": COMPLETE_TRIAL_UNITS,
        "failure_type": outcome.failure_type,
        "failure_message": outcome.failure_message,
    }


def _torque_ids(source_kind: str) -> tuple[str, ...]:
    return DOUBLE_PENDULUM_JOINT_IDS if source_kind == "double_pendulum" else ()


def _failure_fields(
    source: CompleteTrialRecordSource,
    outcome: SimulationTrialOutcome,
    header: EnsembleStreamHeader,
) -> PhaseFields:
    points = 0 if source.config.source_kind == "manual" else len(header.point_ids)
    torques = len(_torque_ids(source.config.source_kind))
    return {
        "candidate_time_s": None,
        "impact_time_s": None,
        "event_sample_index": None,
        "event_interpolation_status": "unavailable",
        "pre_impact_sample_count": 0,
        "swing_times_s": np.empty(0),
        "swing_positions_m": np.empty((0, 3)),
        "swing_poses": np.empty((0, 4, 4)),
        "swing_twists": np.empty((0, 6)),
        "swing_joint_positions_m": np.empty((0, points, 3)),
        "swing_applied_torques_nm": np.empty((0, torques)),
        "impact_outcome": None,
        "delivery_state": None,
        "post_impact_state": None,
        "launch_state": None,
        "flight_times_s": np.empty(0),
        "flight_positions_m": np.empty((0, 3)),
        "flight_velocities_mps": np.empty((0, 3)),
    }


def _run_fields(run: SimulationRun) -> PhaseFields:
    event_time = run.inspection_time_s
    event_index = int(np.argmin(np.abs(run.swing_times - event_time)))
    exact = math.isclose(
        float(run.swing_times[event_index]), event_time, rel_tol=0.0, abs_tol=1e-12
    )
    impact = state_mapping(
        run.impact_outcome.to_dict(), IMPACT_FIELDS, "impact_outcome"
    )
    delivery = (
        None
        if run.delivery is None
        else state_mapping(run.delivery, DELIVERY_FIELDS, "delivery_state")
    )
    post = (
        None
        if run.post_impact is None
        else state_mapping(run.post_impact, POST_IMPACT_FIELDS, "post_impact_state")
    )
    launch = (
        None
        if run.launch is None
        else state_mapping(run.launch, LAUNCH_FIELDS, "launch_state")
    )
    return {
        "candidate_time_s": float(run.impact_outcome.candidate_time_s),
        "impact_time_s": None
        if run.impact_time_s is None
        else float(run.impact_time_s),
        "event_sample_index": event_index,
        "event_interpolation_status": "exact_sample"
        if exact
        else "nearest_sample_only",
        "pre_impact_sample_count": int(
            np.searchsorted(run.swing_times, event_time, side="right")
        ),
        "swing_times_s": run.swing_times,
        "swing_positions_m": run.swing_positions,
        "swing_poses": run.swing_poses,
        "swing_twists": run.swing_twists,
        "swing_joint_positions_m": run.swing_joints,
        "swing_applied_torques_nm": run.swing_applied_torques_nm,
        "impact_outcome": impact,
        "delivery_state": delivery,
        "post_impact_state": post,
        "launch_state": launch,
        "flight_times_s": run.flight_times,
        "flight_positions_m": run.flight_positions,
        "flight_velocities_mps": run.flight_velocities,
    }


__all__ = [
    "COMPLETE_TRIAL_SCHEMA",
    "CompleteTrialRecord",
    "CompleteTrialRecordSource",
    "build_complete_trial_record",
]
