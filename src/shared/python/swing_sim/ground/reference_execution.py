"""One-shot bounded execution over the authoritative bounce and surface phases."""

from __future__ import annotations

from .bounce_simulation import simulate_repeated_bounce
from .bounce_types import BounceTerminationReason, RepeatedBounceResult
from .contract_records import GroundSimulationRequest, GroundSimulationResult
from .ground_result_composer import GroundCompositionError, compose_ground_result
from .reference_execution_types import (
    GroundReferenceCancelled,
    GroundReferenceExecution,
    GroundReferenceExecutionError,
    GroundReferencePhase,
)
from .request_identity import ground_request_fingerprint
from .skid_roll_result_types import SkidRollResult
from .skid_roll_simulation import simulate_skid_roll
from .surface_motion_types import SkidRollTerminationReason

_COMPOSABLE_SUFFIX_REASONS = frozenset(
    {
        SkidRollTerminationReason.REST,
        SkidRollTerminationReason.LEFT_SURFACE,
        SkidRollTerminationReason.TIME_LIMIT,
        SkidRollTerminationReason.EVENT_LIMIT,
    }
)


def _selected_execution(
    execution: GroundReferenceExecution | None,
) -> GroundReferenceExecution:
    selected = GroundReferenceExecution() if execution is None else execution
    if type(selected) is not GroundReferenceExecution:
        raise ValueError("execution must be an exact GroundReferenceExecution")
    try:
        selected.validate()
    except AttributeError as error:
        raise ValueError(
            "execution must be an exact GroundReferenceExecution"
        ) from error
    return selected


def _raise_terminal(
    phase: GroundReferencePhase,
    native_reason: str,
    request_fingerprint_sha256: str,
) -> None:
    error_type = (
        GroundReferenceCancelled
        if native_reason == "cancelled"
        else GroundReferenceExecutionError
    )
    raise error_type(phase, native_reason, request_fingerprint_sha256)


def _validate_prefix(
    prefix: RepeatedBounceResult,
    request_fingerprint_sha256: str,
) -> None:
    reason = prefix.termination.reason
    if reason is not BounceTerminationReason.SETTLED_TO_SKID:
        _raise_terminal(
            GroundReferencePhase.BOUNCE,
            reason.value,
            request_fingerprint_sha256,
        )


def _validate_suffix(
    suffix: SkidRollResult,
    request_fingerprint_sha256: str,
) -> None:
    reason = suffix.termination.reason
    if reason not in _COMPOSABLE_SUFFIX_REASONS:
        _raise_terminal(
            GroundReferencePhase.SKID_ROLL,
            reason.value,
            request_fingerprint_sha256,
        )


def run_ground_reference(
    request: GroundSimulationRequest,
    execution: GroundReferenceExecution | None = None,
) -> GroundSimulationResult:
    """Run the qualified planar v1 pipeline or fail with typed evidence.

    The function delegates physics to the existing bounce and skid/roll
    solvers. It returns only outcomes the public ``GroundSimulationResult``
    contract can represent without relabeling or fabricating terminal states.
    """
    if type(request) is not GroundSimulationRequest:
        raise ValueError("request must be an exact GroundSimulationRequest")
    selected = _selected_execution(execution)
    if selected.resolver is not None:
        selected.resolver.validate_request(request)
    fingerprint = ground_request_fingerprint(request)
    prefix = simulate_repeated_bounce(
        request,
        selected.bounce_settings,
        is_cancelled=selected.is_cancelled,
    )
    _validate_prefix(prefix, fingerprint)
    suffix = simulate_skid_roll(request, prefix, selected.skid_roll_execution())
    _validate_suffix(suffix, fingerprint)
    try:
        return compose_ground_result(request, prefix, suffix)
    except GroundCompositionError as error:
        raise GroundReferenceExecutionError(
            GroundReferencePhase.COMPOSITION,
            "composition_error",
            fingerprint,
        ) from error


__all__ = ["run_ground_reference"]
