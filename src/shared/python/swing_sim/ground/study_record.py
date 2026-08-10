"""Top-level versioned record for one qualified ground study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from shared.python.swing_sim.solver.spatial_targets import SpatialTarget

from .contract_types import (
    GroundFrame,
    GroundResultStatus,
    GroundSurfaceProfile,
    GroundTerminationReason,
    _positive,
    _text,
)
from .profile_binding import BoundGroundSurface
from .result_types import GroundWarning
from .study_derivation import derive_solver_eligibility, derive_study_status
from .study_geometry import (
    endpoint_is_airborne,
    intrinsic_target_miss,
    surface_contact_point,
    validate_nonpenetrating_endpoint,
    validate_target,
)
from .study_target import canonical_ground_target
from .study_types import (
    GROUND_STUDY_SCHEMA_VERSION,
    GroundEndpointKind,
    GroundSolverEligibility,
    GroundStudyMetrics,
    GroundStudyProfile,
    GroundStudyStatus,
    GroundTargetEvaluation,
    GroundTargetUnavailableReason,
)
from .unavailable_types import GroundUnavailableField
from .validation import validate_status_reason


@dataclass(frozen=True)
class GroundStudyProjection:
    """Versioned projection safe for target, variation, and solver adapters."""

    request_id: str
    request_sha256: str
    result_sha256: str
    surface: GroundSurfaceProfile
    ball_radius_m: float
    model_id: str
    model_version: str
    result_status: GroundResultStatus
    status: GroundStudyStatus
    termination_reason: GroundTerminationReason
    metrics: GroundStudyMetrics | None
    target: SpatialTarget | None
    first_contact_target: GroundTargetEvaluation | None
    final_target: GroundTargetEvaluation | None
    final_target_unavailable_reason: GroundTargetUnavailableReason | None
    solver_eligibility: GroundSolverEligibility
    profile: GroundStudyProfile | None
    warnings: tuple[GroundWarning, ...]
    unavailable_fields: tuple[GroundUnavailableField, ...]
    schema_version: str = GROUND_STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("request_id", "model_id", "model_version"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("request_sha256", "result_sha256"):
            digest = _text(getattr(self, name), name).lower()
            if len(digest) != 64 or any(
                item not in "0123456789abcdef" for item in digest
            ):
                raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
            object.__setattr__(self, name, digest)
        if type(self.surface) is not GroundSurfaceProfile:
            raise TypeError("surface must use the exact GroundSurfaceProfile type")
        object.__setattr__(
            self, "ball_radius_m", _positive(self.ball_radius_m, "ball_radius_m")
        )
        if self.target is not None:
            if type(self.target) is not SpatialTarget:
                raise TypeError("target must use the exact SpatialTarget type")
            object.__setattr__(self, "target", canonical_ground_target(self.target))
        object.__setattr__(
            self, "result_status", GroundResultStatus(self.result_status)
        )
        object.__setattr__(self, "status", GroundStudyStatus(self.status))
        object.__setattr__(
            self, "termination_reason", GroundTerminationReason(self.termination_reason)
        )
        if self.final_target_unavailable_reason is not None:
            object.__setattr__(
                self,
                "final_target_unavailable_reason",
                GroundTargetUnavailableReason(self.final_target_unavailable_reason),
            )
        self._validate_nested_types()
        if self.schema_version != GROUND_STUDY_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        self._validate_coherence()

    @property
    def surface_id(self) -> str:
        """Return the exact bound surface identifier."""
        return str(self.surface.surface_id)

    @property
    def frame(self) -> GroundFrame:
        """Return the exact bound surface frame."""
        return self.surface.frame

    def _validate_nested_types(self) -> None:
        optional_exact = (
            (self.metrics, GroundStudyMetrics, "metrics"),
            (self.target, SpatialTarget, "target"),
            (self.first_contact_target, GroundTargetEvaluation, "first_contact_target"),
            (self.final_target, GroundTargetEvaluation, "final_target"),
            (self.profile, GroundStudyProfile, "profile"),
        )
        for value, expected, name in optional_exact:
            if value is not None and type(value) is not expected:
                raise TypeError(f"{name} must use the exact {expected.__name__} type")
        if type(self.solver_eligibility) is not GroundSolverEligibility:
            raise TypeError("solver_eligibility must use the exact contract type")
        if type(self.warnings) is not tuple or not all(
            type(item) is GroundWarning for item in self.warnings
        ):
            raise TypeError("warnings must use exact GroundWarning records")
        if type(self.unavailable_fields) is not tuple or not all(
            type(item) is GroundUnavailableField for item in self.unavailable_fields
        ):
            raise TypeError("unavailable_fields must use exact contract records")

    def _validate_coherence(self) -> None:
        validate_status_reason(self.result_status, self.termination_reason)
        numeric = self.result_status in {
            GroundResultStatus.COMPLETE,
            GroundResultStatus.PARTIAL,
        }
        if numeric != (self.metrics is not None):
            raise ValueError("numeric result status must match metrics availability")
        if self.status is not derive_study_status(
            self.result_status,
            self.termination_reason,
        ):
            raise ValueError("study status does not match result termination")
        self._validate_endpoint_geometry(numeric)
        self._validate_target_presence(numeric)
        self._validate_target_identity()
        self._validate_profile_binding()
        self._validate_unavailable_fields()
        self._validate_solver_eligibility()

    def _validate_endpoint_geometry(self, numeric: bool) -> None:
        if not numeric:
            return
        assert self.metrics is not None, "numeric result requires metrics"
        surface_contact_point(
            self.metrics.first_contact_position_m,
            self.surface,
            self.ball_radius_m,
        )
        if self.result_status is GroundResultStatus.COMPLETE:
            surface_contact_point(
                self.metrics.final_observed_position_m,
                self.surface,
                self.ball_radius_m,
            )
        else:
            validate_nonpenetrating_endpoint(
                self.metrics.final_observed_position_m,
                self.surface,
                self.ball_radius_m,
            )

    def _validate_target_presence(self, numeric: bool) -> None:
        if self.target is None:
            if (
                self.first_contact_target is not None
                or self.final_target is not None
                or self.final_target_unavailable_reason is not None
            ):
                raise ValueError("target evaluations require a target")
            return
        if numeric:
            self._validate_numeric_target_presence()
        elif (
            self.first_contact_target is not None
            or self.final_target is not None
            or self.final_target_unavailable_reason is not None
        ):
            raise ValueError("nonnumeric results cannot fabricate target evaluations")
        if self.first_contact_target is not None and (
            self.first_contact_target.endpoint_kind
            is not GroundEndpointKind.FIRST_CONTACT
        ):
            raise ValueError("first target evaluation must use first_contact")
        if self.final_target is not None:
            expected_kind = (
                GroundEndpointKind.FINAL_REST
                if self.termination_reason is GroundTerminationReason.REST
                else GroundEndpointKind.FINAL_OBSERVED
            )
            if self.final_target.endpoint_kind is not expected_kind:
                raise ValueError(
                    "final target endpoint kind does not match termination"
                )
        if self.final_target_unavailable_reason is not None:
            if self.result_status is not GroundResultStatus.PARTIAL:
                raise ValueError(
                    "only partial results may have airborne target endpoint"
                )
            assert self.metrics is not None, "numeric target requires metrics"
            if not endpoint_is_airborne(
                self.metrics.final_observed_position_m,
                self.surface,
                self.ball_radius_m,
            ):
                raise ValueError("only an airborne endpoint can be target-unavailable")

    def _validate_numeric_target_presence(self) -> None:
        if self.first_contact_target is None:
            raise ValueError("numeric targeted results require first-contact target")
        has_final = self.final_target is not None
        has_reason = self.final_target_unavailable_reason is not None
        if has_final == has_reason:
            raise ValueError(
                "numeric target requires final evaluation or unavailable reason"
            )

    def _validate_target_identity(self) -> None:
        if self.target is None:
            return
        validate_target(self.target, self.surface)
        expected_label = self.target.label
        expected_center = self.target.point.app_coordinates_m
        for evaluation in (self.first_contact_target, self.final_target):
            if evaluation is None:
                continue
            self._validate_target_evaluation(
                evaluation, expected_label, expected_center
            )
        if self.metrics is not None and self.first_contact_target is not None:
            if (
                self.first_contact_target.ball_center_m
                != self.metrics.first_contact_position_m
            ):
                raise ValueError("first target ball center does not match metrics")
        if self.metrics is not None and self.final_target is not None:
            if (
                self.final_target.ball_center_m
                != self.metrics.final_observed_position_m
            ):
                raise ValueError("final target ball center does not match metrics")

    def _validate_target_evaluation(
        self,
        evaluation: GroundTargetEvaluation,
        expected_label: str,
        expected_center: tuple[float, float, float],
    ) -> None:
        if evaluation.target_label != expected_label:
            raise ValueError("target evaluation label does not match target")
        if any(
            not math.isclose(left, right, abs_tol=1e-9)
            for left, right in zip(
                evaluation.target_center_m,
                expected_center,
                strict=True,
            )
        ):
            raise ValueError("target evaluation center does not match target")
        expected_contact = surface_contact_point(
            evaluation.ball_center_m,
            self.surface,
            self.ball_radius_m,
        )
        if any(
            not math.isclose(left, right, abs_tol=1e-9)
            for left, right in zip(
                evaluation.contact_point_m,
                expected_contact,
                strict=True,
            )
        ):
            raise ValueError("target contact point does not match sphere-plane contact")
        assert self.target is not None, "target evaluation requires target"
        expected_miss = intrinsic_target_miss(
            self.target,
            expected_contact,
            self.surface,
        )
        if evaluation.miss != expected_miss:
            raise ValueError("target miss does not match target geometry")

    def _validate_profile_binding(self) -> None:
        if self.profile is None:
            return
        material = self.profile.material_profile
        BoundGroundSurface(
            self.surface,
            material,
            material.profile_id,
            material.revision,
            material.canonical_sha256(),
            material.qualification,
            material.applicability,
            self.profile.operating_condition,
            self.profile.warnings,
        )

    def _validate_unavailable_fields(self) -> None:
        has_unavailable = bool(self.unavailable_fields)
        expected = self.result_status is GroundResultStatus.UNAVAILABLE
        if has_unavailable != expected:
            raise ValueError("unavailable fields must match unavailable result status")

    def _validate_solver_eligibility(self) -> None:
        expected = derive_solver_eligibility(
            self.result_status,
            self.termination_reason,
            self.profile,
        )
        if self.solver_eligibility != expected:
            raise ValueError("solver eligibility does not match study evidence")

    def to_dict(self) -> dict[str, Any]:
        """Return a strict deterministic JSON-compatible mapping."""
        from .study_wire import study_to_dict

        return study_to_dict(self)

    def to_json(self) -> str:
        """Return canonical numeric JSON."""
        from .study_wire import study_to_json

        return study_to_json(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GroundStudyProjection:
        """Parse one strict projection mapping."""
        from .study_wire import study_from_dict

        return study_from_dict(payload)

    @classmethod
    def from_json(cls, text: str) -> GroundStudyProjection:
        """Parse one strict projection JSON document."""
        from .study_wire import study_from_json

        return study_from_json(text)


__all__ = ["GroundStudyProjection"]
