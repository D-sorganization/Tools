"""Typed construction bundles for complete trial records."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TypedDict

import numpy as np

from .simulation_types import TrialEvaluationStatus

COMPLETE_TRIAL_UNITS = MappingProxyType(
    {
        "candidate_time_s": "s",
        "impact_time_s": "s",
        "swing_times_s": "s",
        "swing_positions_m": "m",
        "swing_poses": "homogeneous_transform",
        "swing_twists": "linear_m_per_s_then_angular_rad_per_s",
        "swing_joint_positions_m": "m",
        "swing_applied_torques_nm": "N*m",
        "flight_times_s": "s",
        "flight_positions_m": "m",
        "flight_velocities_mps": "m/s",
    }
)


class CommonFields(TypedDict):
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
    failure_type: str | None
    failure_message: str | None


class PhaseFields(TypedDict):
    candidate_time_s: float | None
    impact_time_s: float | None
    event_sample_index: int | None
    event_interpolation_status: str
    pre_impact_sample_count: int
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


__all__ = ["COMPLETE_TRIAL_UNITS", "CommonFields", "PhaseFields"]
