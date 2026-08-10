"""Immutable qualified ground-study projection contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from shared.python.swing_sim.solver.spatial_targets import TargetMiss

from .contract_types import (
    Vector3,
    _nonnegative,
    _text,
    _vector,
)
from .profile_binding import ProfileOperatingCondition
from .profile_types import (
    GroundMaterialProfile,
    GroundModelUseStatus,
    GroundQualificationStatus,
)
from .result_types import GroundSummary

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

GROUND_STUDY_SCHEMA_VERSION = "ground-study-projection/v1"
_DISTANCE_TOLERANCE_M = 1e-9


class GroundStudyStatus(StrEnum):
    """Scientific availability of one ground study projection."""

    COMPLETE = "complete"
    CENSORED = "censored"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class GroundEndpointKind(StrEnum):
    """Physical endpoint evaluated against a spatial target."""

    FIRST_CONTACT = "first_contact"
    FINAL_OBSERVED = "final_observed"
    FINAL_REST = "final_rest"


class GroundTargetUnavailableReason(StrEnum):
    """Why a numeric final observation has no defensible surface miss."""

    ENDPOINT_AIRBORNE = "endpoint_airborne"


class GroundSolverEligibilityReason(StrEnum):
    """Canonical reasons a projected result may enter an objective."""

    ELIGIBLE = "eligible"
    RESULT_NOT_COMPLETE = "result_not_complete"
    NOT_REST_TERMINATED = "not_rest_terminated"
    MISSING_PROFILE_BINDING = "missing_profile_binding"
    PROFILE_UNQUALIFIED = "profile_unqualified"
    PROFILE_ILLUSTRATIVE = "profile_illustrative"


@dataclass(frozen=True)
class GroundStudyProfile:
    """Exact profile and operating-condition evidence used by one study."""

    material_profile: GroundMaterialProfile
    operating_condition: ProfileOperatingCondition
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.material_profile) is not GroundMaterialProfile:
            raise TypeError("material_profile must use the exact contract type")
        if type(self.operating_condition) is not ProfileOperatingCondition:
            raise TypeError("operating_condition must use the exact contract type")
        if type(self.warnings) is not tuple or not all(
            isinstance(item, str) for item in self.warnings
        ):
            raise TypeError("profile warnings must be a tuple of strings")
        normalized = tuple(_text(item, "profile warning") for item in self.warnings)
        if normalized != tuple(dict.fromkeys(normalized)):
            raise ValueError("profile warnings must be unique and stable")
        object.__setattr__(self, "warnings", normalized)

    @property
    def profile_id(self) -> str:
        """Return the material profile identifier."""
        return str(self.material_profile.profile_id)

    @property
    def profile_revision(self) -> str:
        """Return the material profile revision."""
        return str(self.material_profile.revision)

    @property
    def profile_sha256(self) -> str:
        """Return the canonical material profile digest."""
        return str(self.material_profile.canonical_sha256())

    @property
    def qualification_status(self) -> GroundQualificationStatus:
        """Return qualification derived from the embedded evidence gates."""
        return self.material_profile.qualification.status

    @property
    def model_use_status(self) -> GroundModelUseStatus:
        """Return scientific use status derived from embedded calibration."""
        return self.material_profile.model_use_status


@dataclass(frozen=True)
class GroundStudyMetrics:
    """Qualified distances plus the exact observed endpoint geometry."""

    summary: GroundSummary
    first_contact_position_m: Vector3
    final_observed_position_m: Vector3
    ground_elapsed_s: float

    def __post_init__(self) -> None:
        if type(self.summary) is not GroundSummary:
            raise TypeError("summary must use the exact GroundSummary type")
        object.__setattr__(
            self,
            "first_contact_position_m",
            _vector(self.first_contact_position_m, "first_contact_position_m"),
        )
        object.__setattr__(
            self,
            "final_observed_position_m",
            _vector(self.final_observed_position_m, "final_observed_position_m"),
        )
        object.__setattr__(
            self,
            "ground_elapsed_s",
            _nonnegative(self.ground_elapsed_s, "ground_elapsed_s"),
        )
        self._validate_summary_geometry()

    def _validate_summary_geometry(self) -> None:
        first = self.first_contact_position_m
        final = self.final_observed_position_m
        expected = {
            "carry_distance_m": math.hypot(first[0], first[2]),
            "total_distance_m": math.hypot(final[0], final[2]),
            "final_downrange_m": final[0],
            "final_offline_m": final[2],
            "surface_path_distance_m": (
                self.summary.skid_distance_m + self.summary.roll_distance_m
            ),
        }
        for name, value in expected.items():
            if not math.isclose(
                getattr(self.summary, name), value, abs_tol=_DISTANCE_TOLERANCE_M
            ):
                raise ValueError(f"summary {name} does not match study geometry")


@dataclass(frozen=True)
class GroundTargetEvaluation:
    """One ball/surface contact point evaluated against one spatial target."""

    target_label: str
    endpoint_kind: GroundEndpointKind
    ball_center_m: Vector3
    contact_point_m: Vector3
    target_center_m: Vector3
    center_residual_m: Vector3
    center_distance_m: float
    miss: TargetMiss

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_label", _text(self.target_label, "target_label")
        )
        object.__setattr__(
            self, "endpoint_kind", GroundEndpointKind(self.endpoint_kind)
        )
        for name in (
            "ball_center_m",
            "contact_point_m",
            "target_center_m",
            "center_residual_m",
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        expected = tuple(
            self.contact_point_m[index] - self.target_center_m[index]
            for index in range(3)
        )
        if any(
            not math.isclose(left, right, abs_tol=_DISTANCE_TOLERANCE_M)
            for left, right in zip(self.center_residual_m, expected, strict=True)
        ):
            raise ValueError("center_residual_m must equal contact point minus target")
        distance = _nonnegative(self.center_distance_m, "center_distance_m")
        norm = math.hypot(*self.center_residual_m)
        if not math.isclose(distance, norm, abs_tol=_DISTANCE_TOLERANCE_M):
            raise ValueError("center_distance_m must equal center residual norm")
        object.__setattr__(self, "center_distance_m", distance)
        if type(self.miss) is not TargetMiss:
            raise TypeError("target miss must use the exact TargetMiss type")
        reconstructed = tuple(
            self.miss.closest_point_m[index] + self.miss.vector_m[index]
            for index in range(3)
        )
        if any(
            not math.isclose(left, right, abs_tol=_DISTANCE_TOLERANCE_M)
            for left, right in zip(
                reconstructed,
                self.contact_point_m,
                strict=True,
            )
        ):
            raise ValueError("target miss must reconstruct the contact point")

    @property
    def accepted(self) -> bool:
        """Return true only for a defensibly evaluated accepted endpoint."""
        return bool(self.miss.accepted)


@dataclass(frozen=True)
class GroundSolverEligibility:
    """Fail-closed solver/objective admission decision."""

    eligible: bool
    reasons: tuple[GroundSolverEligibilityReason, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.eligible, bool):
            raise TypeError("eligible must be a boolean")
        if type(self.reasons) is not tuple:
            raise TypeError("eligibility reasons must be a tuple")
        reasons = tuple(GroundSolverEligibilityReason(item) for item in self.reasons)
        canonical = tuple(
            item for item in GroundSolverEligibilityReason if item in reasons
        )
        if reasons != canonical or len(reasons) != len(set(reasons)):
            raise ValueError("eligibility reasons must be unique and canonical")
        if self.eligible != (reasons == (GroundSolverEligibilityReason.ELIGIBLE,)):
            raise ValueError("eligible must match the canonical eligibility reason")
        if not reasons:
            raise ValueError("eligibility reasons must not be empty")
        object.__setattr__(self, "reasons", reasons)


__all__ = [
    "GROUND_STUDY_SCHEMA_VERSION",
    "GroundEndpointKind",
    "GroundSolverEligibility",
    "GroundSolverEligibilityReason",
    "GroundStudyMetrics",
    "GroundStudyProfile",
    "GroundStudyStatus",
    "GroundTargetEvaluation",
    "GroundTargetUnavailableReason",
]
