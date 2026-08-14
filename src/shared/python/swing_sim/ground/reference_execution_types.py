"""Typed controls and failures for the canonical ground reference executor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

from .bounce_types import BounceModelSettings
from .request_identity import validate_request_fingerprint
from .skid_roll_simulation import SkidRollExecution
from .surface_motion_types import SkidRollSettings
from .surface_resolver import SurfaceResolver

GROUND_REFERENCE_EXECUTION_SCHEMA_VERSION = "ground-reference-execution/v1"
CancellationCheck = Callable[[], bool]


class GroundReferencePhase(StrEnum):
    """Pipeline phase that produced a non-representable terminal state."""

    BOUNCE = "bounce"
    SKID_ROLL = "skid_roll"
    COMPOSITION = "composition"


class GroundReferenceExecutionError(RuntimeError):
    """Typed fail-closed evidence for an outcome outside the public v1 result."""

    def __init__(
        self,
        phase: GroundReferencePhase,
        native_reason: str,
        request_fingerprint_sha256: str,
    ) -> None:
        self.phase = GroundReferencePhase(phase)
        if not isinstance(native_reason, str) or not native_reason.strip():
            raise ValueError("native_reason must be nonempty text")
        self.native_reason = native_reason.strip()
        self.request_fingerprint_sha256 = validate_request_fingerprint(
            request_fingerprint_sha256
        )
        super().__init__(
            f"ground reference {self.phase.value} failed: {self.native_reason}"
        )


class GroundReferenceCancelled(GroundReferenceExecutionError):
    """Distinct operational signal for caller-requested cancellation."""


@dataclass(frozen=True)
class GroundReferenceExecution:
    """Immutable phase controls for one bounded canonical ground run."""

    bounce_settings: BounceModelSettings = field(default_factory=BounceModelSettings)
    skid_roll_settings: SkidRollSettings = field(default_factory=SkidRollSettings)
    resolver: SurfaceResolver | None = None
    is_cancelled: CancellationCheck | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Recheck exact nested controls at a public execution boundary."""
        if type(self.bounce_settings) is not BounceModelSettings:
            raise ValueError("bounce_settings must be exact BounceModelSettings")
        if type(self.skid_roll_settings) is not SkidRollSettings:
            raise ValueError("skid_roll_settings must be exact SkidRollSettings")
        if self.resolver is not None and type(self.resolver) is not SurfaceResolver:
            raise ValueError("resolver must be exact SurfaceResolver")
        if self.is_cancelled is not None and not callable(self.is_cancelled):
            raise ValueError("is_cancelled must be callable")

    def skid_roll_execution(self) -> SkidRollExecution:
        """Build the existing suffix execution without duplicating its contract."""
        return SkidRollExecution(
            settings=self.skid_roll_settings,
            resolver=self.resolver,
            is_cancelled=self.is_cancelled,
        )


__all__ = [
    "GROUND_REFERENCE_EXECUTION_SCHEMA_VERSION",
    "GroundReferenceCancelled",
    "GroundReferenceExecution",
    "GroundReferenceExecutionError",
    "GroundReferencePhase",
]
