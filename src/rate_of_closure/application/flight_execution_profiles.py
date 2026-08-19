"""Versioned, fail-closed flight execution-profile qualification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from rate_of_closure.application._flight_execution_profile_runtime import (
    WATERLOO_SETTING_IDS,
    WaterlooSettings,
    parse_waterloo_settings,
    recompute_waterloo,
)
from rate_of_closure.application._regional_ground_execution_job_values import (
    FlightExecutionInput,
    canonical_text,
    digest,
)
from rate_of_closure.application._workspace_validation import stable_id
from rate_of_closure.application.regional_ground_execution_job import (
    canonical_flight_evidence_sha256,
)
from shared.python.swing_sim.flight import (
    CancellationCheck,
    FlightCancellationCallbackError,
    FlightGroundTransferSettings,
    FlightResult,
    FlightSimulationCancelled,
    LaunchConditions,
)

FLIGHT_EXECUTION_PROFILE_REGISTRY_SCHEMA_VERSION = (
    "rate-of-closure/flight-execution-profile-registry/v1"
)
_WATERLOO_MODEL_ID = "waterloo_penner"
_TOOLS_CORE_MODEL_VERSION = "tools-core/1.0.0"
_WATERLOO_RECOMPUTATION_CONTRACT = "waterloo-penner-adaptive-rk45-planar-contact/v1"
_SETTING_IDS = WATERLOO_SETTING_IDS


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


def _evaluate(
    launch: LaunchConditions,
    transfer: FlightGroundTransferSettings,
    flight: FlightExecutionInput,
    cancellation_requested: CancellationCheck | None = None,
) -> tuple[FlightExecutionQualification, FlightResult | None]:
    qualification, result = _recompute_registered(
        launch,
        transfer,
        model_id=flight.model_id,
        model_version=flight.model_version,
        settings=flight.settings,
        cancellation_requested=cancellation_requested,
    )
    if not qualification.qualified or result is None:
        return qualification, result
    return (_compare_digests(flight, qualification), result)


def _unqualified(
    reason: FlightExecutionQualificationReason,
    model_id: str,
    model_version: str,
) -> tuple[FlightExecutionQualification, None]:
    return FlightExecutionQualification(reason, model_id, model_version), None


def _resolved_settings(
    model_id: str,
    model_version: str,
    settings: Mapping[str, float],
) -> tuple[FlightExecutionQualification | None, WaterlooSettings | None]:
    if (model_id, model_version) not in _PROFILES:
        return _unqualified(
            FlightExecutionQualificationReason.PROFILE_NOT_REGISTERED,
            model_id,
            model_version,
        )
    try:
        return None, parse_waterloo_settings(settings)
    except (TypeError, ValueError):
        return _unqualified(
            FlightExecutionQualificationReason.SETTINGS_SCHEMA_INVALID,
            model_id,
            model_version,
        )


def _qualified_recomputation(
    model_id: str,
    model_version: str,
    result: FlightResult,
    cancellation_requested: CancellationCheck | None,
) -> FlightExecutionQualification:
    trajectory_sha256, result_sha256 = canonical_flight_evidence_sha256(
        result,
        cancellation_requested=cancellation_requested,
    )
    return FlightExecutionQualification(
        FlightExecutionQualificationReason.QUALIFIED,
        model_id,
        model_version,
        trajectory_sha256,
        result_sha256,
    )


def _recompute_registered(
    launch: LaunchConditions,
    transfer: FlightGroundTransferSettings,
    *,
    model_id: str,
    model_version: str,
    settings: Mapping[str, float],
    cancellation_requested: CancellationCheck | None = None,
) -> tuple[FlightExecutionQualification, FlightResult | None]:
    """Resolve and recompute one registered profile without declared digests."""
    stable_id(model_id, "profile model_id")
    canonical_text(model_version, "profile model_version")
    unqualified, resolved_settings = _resolved_settings(
        model_id,
        model_version,
        settings,
    )
    if unqualified is not None or resolved_settings is None:
        assert unqualified is not None
        return unqualified, None
    try:
        result = recompute_waterloo(
            launch,
            transfer,
            resolved_settings,
            cancellation_requested,
        )
    except (FlightSimulationCancelled, FlightCancellationCallbackError):
        raise
    except Exception:
        return _unqualified(
            FlightExecutionQualificationReason.RECOMPUTATION_FAILED,
            model_id,
            model_version,
        )
    return (
        _qualified_recomputation(
            model_id,
            model_version,
            result,
            cancellation_requested,
        ),
        result,
    )


def _compare_digests(
    flight: FlightExecutionInput,
    qualification: FlightExecutionQualification,
) -> FlightExecutionQualification:
    trajectory_digest = qualification.recomputed_trajectory_sha256
    result_digest = qualification.recomputed_result_sha256
    if trajectory_digest is None or result_digest is None:
        raise ValueError("qualified recomputation must include both digests")
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


def build_qualified_flight_execution_input(
    launch: LaunchConditions,
    transfer: FlightGroundTransferSettings,
    *,
    model_id: str,
    model_version: str,
    settings: Mapping[str, float],
) -> FlightExecutionInput:
    """Recompute a registered profile and bind its exact result identities."""
    if type(launch) is not LaunchConditions:
        raise TypeError("launch must be an exact LaunchConditions")
    if type(transfer) is not FlightGroundTransferSettings:
        raise TypeError("transfer must be an exact FlightGroundTransferSettings")
    qualification, result = _recompute_registered(
        launch,
        transfer,
        model_id=model_id,
        model_version=model_version,
        settings=settings,
    )
    if not qualification.qualified or result is None:
        raise FlightExecutionProfileQualificationError(qualification)
    assert qualification.recomputed_trajectory_sha256 is not None
    assert qualification.recomputed_result_sha256 is not None
    built = FlightExecutionInput(
        model_id,
        model_version,
        settings,
        qualification.recomputed_trajectory_sha256,
        qualification.recomputed_result_sha256,
    )
    evidence = qualify_flight_execution_input(launch, transfer, built)
    if not evidence.qualified:
        raise FlightExecutionProfileQualificationError(evidence)
    return built


def recompute_qualified_flight_result(
    launch: LaunchConditions,
    transfer: FlightGroundTransferSettings,
    flight: FlightExecutionInput,
    *,
    cancellation_requested: CancellationCheck | None = None,
) -> FlightResult:
    """Return one digest-matched flight or raise typed cancellation."""
    _validate_boundary_inputs(launch, transfer, flight)
    if cancellation_requested is not None and not callable(cancellation_requested):
        raise TypeError("cancellation_requested must be callable or None")
    qualification, result = _evaluate(
        launch,
        transfer,
        flight,
        cancellation_requested,
    )
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
    "build_qualified_flight_execution_input",
    "qualify_flight_execution_input",
    "recompute_qualified_flight_result",
    "registered_flight_execution_profiles",
]
