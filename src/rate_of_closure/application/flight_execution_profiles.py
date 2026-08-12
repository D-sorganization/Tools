"""Versioned, fail-closed flight execution-profile qualification."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from rate_of_closure.application._regional_ground_execution_job_values import (
    FlightExecutionInput,
    canonical_text,
    digest,
)
from rate_of_closure.application._workspace_validation import exact_mapping, stable_id
from rate_of_closure.application.regional_ground_execution_job import (
    canonical_flight_result_sha256,
    canonical_flight_trajectory_sha256,
)
from shared.python.swing_sim.flight import (
    FlightGroundTransferSettings,
    FlightModelRegistry,
    FlightModelType,
    FlightResult,
    LaunchConditions,
    SurfaceFlightSimulationSettings,
    compute_flight_metrics,
    launch_relative_surface,
)

FLIGHT_EXECUTION_PROFILE_REGISTRY_SCHEMA_VERSION = (
    "rate-of-closure/flight-execution-profile-registry/v1"
)
_WATERLOO_MODEL_ID = "waterloo_penner"
_TOOLS_CORE_MODEL_VERSION = "tools-core/1.0.0"
_WATERLOO_RECOMPUTATION_CONTRACT = "waterloo-penner-adaptive-rk45-planar-contact/v1"
_SETTING_IDS = ("max_time_s", "sample_every", "step_s")
_SETTING_FIELDS = frozenset(_SETTING_IDS)
_MAX_TIME_S = 120.0
_MIN_STEP_S = 0.0001
_MAX_STEP_S = 0.1
_MAX_SAMPLE_EVERY = 10_000
_MAX_RETAINED_INTERVAL_S = 1.0


class FlightExecutionQualificationReason(StrEnum):
    """Stable outcome of exact profile lookup and evidence recomputation."""

    QUALIFIED = "qualified"
    PROFILE_NOT_REGISTERED = "profile_not_registered"
    SETTINGS_SCHEMA_INVALID = "settings_schema_invalid"
    RECOMPUTATION_FAILED = "recomputation_failed"
    TRAJECTORY_DIGEST_MISMATCH = "trajectory_digest_mismatch"
    RESULT_DIGEST_MISMATCH = "result_digest_mismatch"


@dataclass(frozen=True, slots=True)
class FlightExecutionProfile:
    """Public immutable descriptor for one exact executable profile."""

    model_id: str
    model_version: str
    setting_ids: tuple[str, ...]
    recomputation_contract: str
    schema_version: str = FLIGHT_EXECUTION_PROFILE_REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        stable_id(self.model_id, "profile model_id")
        canonical_text(self.model_version, "profile model_version")
        if self.setting_ids != tuple(sorted(set(self.setting_ids))):
            raise ValueError("profile setting_ids must be sorted and unique")
        for setting_id in self.setting_ids:
            stable_id(setting_id, "profile setting_id")
        canonical_text(self.recomputation_contract, "recomputation_contract")
        if self.schema_version != FLIGHT_EXECUTION_PROFILE_REGISTRY_SCHEMA_VERSION:
            raise ValueError("unsupported flight execution-profile schema")


@dataclass(frozen=True, slots=True)
class FlightExecutionQualification:
    """Typed qualification evidence without exposing a physical result."""

    reason: FlightExecutionQualificationReason
    model_id: str
    model_version: str
    recomputed_trajectory_sha256: str | None = None
    recomputed_result_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.reason) is not FlightExecutionQualificationReason:
            raise TypeError("reason must be an exact qualification reason")
        canonical_text(self.model_id, "qualification model_id")
        canonical_text(self.model_version, "qualification model_version")
        has_digests = self.recomputed_trajectory_sha256 is not None
        if has_digests != (self.recomputed_result_sha256 is not None):
            raise ValueError("recomputed flight digests must be present together")
        if has_digests:
            digest(self.recomputed_trajectory_sha256, "recomputed trajectory digest")
            digest(self.recomputed_result_sha256, "recomputed result digest")
        recomputed = self.reason in {
            FlightExecutionQualificationReason.QUALIFIED,
            FlightExecutionQualificationReason.TRAJECTORY_DIGEST_MISMATCH,
            FlightExecutionQualificationReason.RESULT_DIGEST_MISMATCH,
        }
        if recomputed != has_digests:
            raise ValueError("qualification reason and recomputation evidence disagree")

    @property
    def qualified(self) -> bool:
        """Return whether both declared digests match recomputed evidence."""
        return self.reason is FlightExecutionQualificationReason.QUALIFIED


class FlightExecutionProfileQualificationError(RuntimeError):
    """Fail-closed signal raised when qualified result release is forbidden."""

    def __init__(self, qualification: FlightExecutionQualification) -> None:
        if type(qualification) is not FlightExecutionQualification:
            raise TypeError("qualification must be exact")
        self.qualification = qualification
        super().__init__(
            f"flight execution evidence is not qualified: {qualification.reason.value}"
        )


@dataclass(frozen=True, slots=True)
class _WaterlooSettings:
    max_time_s: float
    step_s: float
    sample_every: int

    @property
    def retained_interval_s(self) -> float:
        return self.step_s * self.sample_every


_WATERLOO_PROFILE = FlightExecutionProfile(
    _WATERLOO_MODEL_ID,
    _TOOLS_CORE_MODEL_VERSION,
    _SETTING_IDS,
    _WATERLOO_RECOMPUTATION_CONTRACT,
)
_PROFILES = {(_WATERLOO_MODEL_ID, _TOOLS_CORE_MODEL_VERSION): _WATERLOO_PROFILE}


def registered_flight_execution_profiles() -> tuple[FlightExecutionProfile, ...]:
    """Return the stable registry descriptors in exact identity order."""
    return tuple(_PROFILES[key] for key in sorted(_PROFILES))


def _positive_bounded(value: object, name: str, maximum: float) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if not 0.0 < number <= maximum:
        raise ValueError(f"{name} lies outside its profile bound")
    return number


def _waterloo_settings(values: Mapping[str, float]) -> _WaterlooSettings:
    data = exact_mapping(values, _SETTING_FIELDS, "flight profile settings")
    max_time_s = _positive_bounded(data["max_time_s"], "max_time_s", _MAX_TIME_S)
    step_s = _positive_bounded(data["step_s"], "step_s", _MAX_STEP_S)
    if step_s < _MIN_STEP_S:
        raise ValueError("step_s lies outside its profile bound")
    sample_value = data["sample_every"]
    if (
        type(sample_value) not in (int, float)
        or not math.isfinite(float(sample_value))
        or not float(sample_value).is_integer()
    ):
        raise ValueError("sample_every must be a finite whole number")
    sample_every = int(sample_value)
    if not 1 <= sample_every <= _MAX_SAMPLE_EVERY:
        raise ValueError("sample_every lies outside its profile bound")
    settings = _WaterlooSettings(max_time_s, step_s, sample_every)
    if settings.retained_interval_s > _MAX_RETAINED_INTERVAL_S:
        raise ValueError("retained sample interval lies outside its profile bound")
    return settings


def _recompute_waterloo(
    launch: LaunchConditions,
    transfer: FlightGroundTransferSettings,
    settings: _WaterlooSettings,
) -> FlightResult:
    surface = launch_relative_surface(
        transfer.surface,
        launch.ball_radius,
        launch.ball_setup,
    )
    simulation = SurfaceFlightSimulationSettings(
        surface,
        settings.max_time_s,
        settings.step_s,
    )
    model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
    raw = model.simulate_to_surface(launch, simulation)
    retained = list(raw.trajectory[:: settings.sample_every])
    if raw.trajectory and (not retained or retained[-1] is not raw.trajectory[-1]):
        retained.append(raw.trajectory[-1])
    return compute_flight_metrics(retained, raw.model_name)


def _identity(flight: FlightExecutionInput) -> tuple[str, str]:
    return (flight.model_id, flight.model_version)


def _not_recomputed(
    flight: FlightExecutionInput,
    reason: FlightExecutionQualificationReason,
) -> FlightExecutionQualification:
    return FlightExecutionQualification(reason, flight.model_id, flight.model_version)


def _evaluate(
    launch: LaunchConditions,
    transfer: FlightGroundTransferSettings,
    flight: FlightExecutionInput,
) -> tuple[FlightExecutionQualification, FlightResult | None]:
    profile = _PROFILES.get(_identity(flight))
    if profile is None:
        return (
            _not_recomputed(
                flight,
                FlightExecutionQualificationReason.PROFILE_NOT_REGISTERED,
            ),
            None,
        )
    try:
        settings = _waterloo_settings(flight.settings)
    except (TypeError, ValueError):
        return (
            _not_recomputed(
                flight,
                FlightExecutionQualificationReason.SETTINGS_SCHEMA_INVALID,
            ),
            None,
        )
    try:
        result = _recompute_waterloo(launch, transfer, settings)
    except Exception:
        return (
            _not_recomputed(
                flight,
                FlightExecutionQualificationReason.RECOMPUTATION_FAILED,
            ),
            None,
        )
    return (_compare_digests(flight, result), result)


def _compare_digests(
    flight: FlightExecutionInput,
    result: FlightResult,
) -> FlightExecutionQualification:
    trajectory_digest = canonical_flight_trajectory_sha256(result)
    result_digest = canonical_flight_result_sha256(result)
    reason = FlightExecutionQualificationReason.QUALIFIED
    if trajectory_digest != flight.trajectory_sha256:
        reason = FlightExecutionQualificationReason.TRAJECTORY_DIGEST_MISMATCH
    elif result_digest != flight.result_sha256:
        reason = FlightExecutionQualificationReason.RESULT_DIGEST_MISMATCH
    return FlightExecutionQualification(
        reason,
        flight.model_id,
        flight.model_version,
        trajectory_digest,
        result_digest,
    )


def qualify_flight_execution_input(
    launch: LaunchConditions,
    transfer: FlightGroundTransferSettings,
    flight: FlightExecutionInput,
) -> FlightExecutionQualification:
    """Recompute and compare evidence without releasing a physical result."""
    _validate_boundary_inputs(launch, transfer, flight)
    qualification, _result = _evaluate(launch, transfer, flight)
    return qualification


def recompute_qualified_flight_result(
    launch: LaunchConditions,
    transfer: FlightGroundTransferSettings,
    flight: FlightExecutionInput,
) -> FlightResult:
    """Return recomputed flight only after both declared digests match."""
    _validate_boundary_inputs(launch, transfer, flight)
    qualification, result = _evaluate(launch, transfer, flight)
    if not qualification.qualified or result is None:
        raise FlightExecutionProfileQualificationError(qualification)
    return result


def _validate_boundary_inputs(
    launch: LaunchConditions,
    transfer: FlightGroundTransferSettings,
    flight: FlightExecutionInput,
) -> None:
    if type(launch) is not LaunchConditions:
        raise TypeError("launch must be an exact LaunchConditions")
    if type(transfer) is not FlightGroundTransferSettings:
        raise TypeError("transfer must be an exact FlightGroundTransferSettings")
    if type(flight) is not FlightExecutionInput:
        raise TypeError("flight must be an exact FlightExecutionInput")
    flight.__post_init__()


__all__ = [
    "FLIGHT_EXECUTION_PROFILE_REGISTRY_SCHEMA_VERSION",
    "FlightExecutionProfile",
    "FlightExecutionProfileQualificationError",
    "FlightExecutionQualification",
    "FlightExecutionQualificationReason",
    "qualify_flight_execution_input",
    "recompute_qualified_flight_result",
    "registered_flight_execution_profiles",
]
