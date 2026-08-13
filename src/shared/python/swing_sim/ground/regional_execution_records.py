"""Immutable evidence records for regional ground execution v1."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

from ._vector_math import dot
from .contract_records import GroundSimulationResult
from .contract_types import (
    UNIT_SYSTEM_SI,
    GroundEventType,
    GroundProvenance,
    GroundResultStatus,
    _text,
    _WireRecord,
)
from .regional_plan_records import (
    REGIONAL_PLAN_LIMITATIONS,
    GroundRegionalMaterialPlanRequest,
)
from .regional_surface_types import SurfaceRegionTransition
from .regional_transition_binding import validate_transition_against_plan

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

REGIONAL_GROUND_EXECUTION_SCHEMA_VERSION = "ground-regional-execution-result/v1"
REGIONAL_GROUND_EXECUTION_LIMITATIONS = REGIONAL_PLAN_LIMITATIONS
MAX_REGIONAL_GROUND_EXECUTION_WIRE_BYTES = 8_388_608
REGIONAL_GROUND_EXECUTOR_ID = "tools-ground-regional-executor"
REGIONAL_GROUND_EXECUTOR_VERSION = "1.0.0"


class RegionalGroundExecutionStatus(StrEnum):
    """Envelope outcome without widening the frozen ground-result status."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RegionalGroundExecutionFailureReason(StrEnum):
    """Internal outcomes that cannot be serialized as ground-result v1."""

    CANCELLED = "cancelled"
    STEP_LIMIT = "step_limit"
    SURFACE_TRANSITION_LIMIT = "surface_transition_limit"
    UNSUPPORTED_SURFACE = "unsupported_surface"
    NUMERICAL_FAILURE = "numerical_failure"
    COMPOSITION_FAILURE = "composition_failure"


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    normalized = str(_text(value, name))
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return normalized


def execution_input_sha256(ground_digest: str, plan_digest: str) -> str:
    """Hash the two canonical input identities without mixing in physics."""
    ground = _digest(ground_digest, "ground_request_sha256")
    plan = _digest(plan_digest, "regional_plan_sha256")
    payload = canonical_numeric_json(
        {"ground_request_sha256": ground, "regional_plan_sha256": plan}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fixed_limitations(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("limitations must be an array")
    limitations = tuple(value)
    if limitations != REGIONAL_GROUND_EXECUTION_LIMITATIONS:
        raise ValueError("limitations must declare the complete v1 qualification")
    return limitations


@dataclass(frozen=True)
class RegionalGroundExecutionResult(_WireRecord):
    """Strict execution envelope preserving plan and transition provenance."""

    request_id: str
    surface_id: str
    plan_id: str
    ground_request_sha256: str
    regional_plan_sha256: str
    regional_plan: GroundRegionalMaterialPlanRequest
    status: RegionalGroundExecutionStatus
    failure_reason: RegionalGroundExecutionFailureReason | None
    ground_result: GroundSimulationResult | None
    plan_provenance: GroundProvenance
    executor_provenance: GroundProvenance
    model_id: str
    model_version: str
    transitions: tuple[SurfaceRegionTransition, ...]
    limitations: tuple[str, ...] = REGIONAL_GROUND_EXECUTION_LIMITATIONS
    unit_system: str = UNIT_SYSTEM_SI
    schema_version: str = REGIONAL_GROUND_EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "surface_id",
            "plan_id",
            "model_id",
            "model_version",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        ground_digest = _digest(self.ground_request_sha256, "ground_request_sha256")
        plan_digest = _digest(self.regional_plan_sha256, "regional_plan_sha256")
        object.__setattr__(self, "ground_request_sha256", ground_digest)
        object.__setattr__(self, "regional_plan_sha256", plan_digest)
        object.__setattr__(self, "status", RegionalGroundExecutionStatus(self.status))
        if self.failure_reason is not None:
            object.__setattr__(
                self,
                "failure_reason",
                RegionalGroundExecutionFailureReason(self.failure_reason),
            )
        self._validate_nested_types()
        transitions = tuple(self.transitions)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "limitations", _fixed_limitations(self.limitations))
        if self.unit_system != UNIT_SYSTEM_SI:
            raise ValueError(f"unsupported unit_system: {self.unit_system}")
        if self.schema_version != REGIONAL_GROUND_EXECUTION_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        self._validate_status()
        self._validate_transition_ledger()

    @property
    def execution_input_sha256(self) -> str:
        """Return the canonical joint input identity used by the executor."""
        return execution_input_sha256(
            self.ground_request_sha256,
            self.regional_plan_sha256,
        )

    def _validate_nested_types(self) -> None:
        if type(self.plan_provenance) is not GroundProvenance:
            raise ValueError("plan_provenance must be an exact GroundProvenance")
        if type(self.executor_provenance) is not GroundProvenance:
            raise ValueError("executor_provenance must be an exact GroundProvenance")
        if type(self.regional_plan) is not GroundRegionalMaterialPlanRequest:
            raise ValueError("regional_plan must be an exact plan request")
        if (
            self.ground_result is not None
            and type(self.ground_result) is not GroundSimulationResult
        ):
            raise ValueError("ground_result must be an exact GroundSimulationResult")
        if not isinstance(self.transitions, (list, tuple)) or any(
            type(item) is not SurfaceRegionTransition for item in self.transitions
        ):
            raise ValueError("transitions must contain exact transition records")
        if self.executor_provenance.input_sha256 != self.execution_input_sha256:
            raise ValueError(
                "executor provenance must match canonical execution inputs"
            )
        if self.executor_provenance.producer != REGIONAL_GROUND_EXECUTOR_ID:
            raise ValueError("executor producer must match the v1 authority")
        if (
            self.executor_provenance.producer_version
            != REGIONAL_GROUND_EXECUTOR_VERSION
        ):
            raise ValueError("executor version must match the v1 authority")
        self._validate_plan_identity()

    def _validate_plan_identity(self) -> None:
        plan = self.regional_plan
        plan_digest = hashlib.sha256(plan.to_json().encode("utf-8")).hexdigest()
        if plan_digest != self.regional_plan_sha256:
            raise ValueError("regional_plan_sha256 must match the embedded plan")
        if self.plan_id != plan.request_id:
            raise ValueError("plan_id must match the embedded regional plan")
        if self.surface_id != plan.base_surface.surface_id:
            raise ValueError("surface_id must match the regional plan base surface")
        if self.plan_provenance != plan.provenance:
            raise ValueError("plan provenance must match the embedded regional plan")

    def _validate_status(self) -> None:
        ground = self.ground_result
        if ground is None:
            self._validate_null_result_status()
            return
        if self.failure_reason is not None:
            raise ValueError("successful execution cannot declare failure_reason")
        expected = {
            GroundResultStatus.COMPLETE: RegionalGroundExecutionStatus.COMPLETE,
            GroundResultStatus.PARTIAL: RegionalGroundExecutionStatus.PARTIAL,
        }.get(ground.status)
        if expected is None or self.status is not expected:
            raise ValueError("execution status must match the embedded ground result")
        if (ground.request_id, ground.surface_id) != (self.request_id, self.surface_id):
            raise ValueError(
                "embedded ground result identities must match the envelope"
            )
        if (ground.model_id, ground.model_version) != (
            self.model_id,
            self.model_version,
        ):
            raise ValueError("embedded model identity must match the envelope")

    def _validate_null_result_status(self) -> None:
        if self.failure_reason is None:
            raise ValueError("null ground_result requires failure_reason")
        if self.status is RegionalGroundExecutionStatus.CANCELLED:
            if (
                self.failure_reason
                is not RegionalGroundExecutionFailureReason.CANCELLED
            ):
                raise ValueError("cancelled status requires cancelled failure_reason")
            return
        if self.status is not RegionalGroundExecutionStatus.FAILED:
            raise ValueError("null ground_result requires failed or cancelled status")
        if self.failure_reason is RegionalGroundExecutionFailureReason.CANCELLED:
            raise ValueError("cancelled failure_reason requires cancelled status")

    def _validate_transition_ledger(self) -> None:
        transitions = self.transitions
        if any(item.from_surface_id == item.to_surface_id for item in transitions):
            raise ValueError("transition surface identities must differ")
        if any(
            right.event_sequence <= left.event_sequence or right.time_s < left.time_s
            for left, right in zip(transitions, transitions[1:], strict=False)
        ):
            raise ValueError("transition ledger must be strictly ordered")
        if self.ground_result is None:
            if transitions:
                raise ValueError("null ground_result cannot declare transitions")
            return
        events = tuple(
            event
            for event in self.ground_result.events
            if event.event_type is GroundEventType.SURFACE_TRANSITION
        )
        if len(events) != len(transitions):
            raise ValueError("transition ledger must match ground result events")
        for event, transition in zip(events, transitions, strict=True):
            if (
                event.sequence != transition.event_sequence
                or event.time_s != transition.time_s
                or event.position_m != transition.position_m
            ):
                raise ValueError("transition ledger must match ground result events")
            validate_transition_against_plan(
                self.regional_plan,
                transition,
                dot(event.velocity_before_m_s, self.regional_plan.axis_unit),
            )

    @classmethod
    def from_dict(cls, payload: object) -> RegionalGroundExecutionResult:
        """Parse an exact regional execution mapping."""
        from .regional_execution_wire import regional_execution_result_from_dict

        return regional_execution_result_from_dict(payload)


__all__ = [
    "MAX_REGIONAL_GROUND_EXECUTION_WIRE_BYTES",
    "REGIONAL_GROUND_EXECUTION_LIMITATIONS",
    "REGIONAL_GROUND_EXECUTION_SCHEMA_VERSION",
    "REGIONAL_GROUND_EXECUTOR_ID",
    "REGIONAL_GROUND_EXECUTOR_VERSION",
    "RegionalGroundExecutionFailureReason",
    "RegionalGroundExecutionResult",
    "RegionalGroundExecutionStatus",
    "execution_input_sha256",
]
