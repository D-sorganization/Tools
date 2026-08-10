"""Canonical status and objective-eligibility derivation for ground studies."""

from .contract_types import (
    CalibrationKind,
    GroundCalibration,
    GroundResultStatus,
    GroundTerminationReason,
)
from .profile_types import GroundModelUseStatus, GroundQualificationStatus
from .study_types import (
    GroundSolverEligibility,
    GroundSolverEligibilityReason,
    GroundStudyProfile,
    GroundStudyStatus,
)


def derive_study_status(
    result_status: GroundResultStatus,
    termination_reason: GroundTerminationReason,
) -> GroundStudyStatus:
    """Derive scientific availability from canonical result state."""
    if result_status is GroundResultStatus.FAILED:
        return GroundStudyStatus.FAILED
    if result_status is GroundResultStatus.UNAVAILABLE:
        return GroundStudyStatus.UNAVAILABLE
    if (
        result_status is GroundResultStatus.COMPLETE
        and termination_reason is GroundTerminationReason.REST
    ):
        return GroundStudyStatus.COMPLETE
    return GroundStudyStatus.CENSORED


def derive_solver_eligibility(
    result_status: GroundResultStatus,
    termination_reason: GroundTerminationReason,
    profile: GroundStudyProfile | None,
    calibration: GroundCalibration,
) -> GroundSolverEligibility:
    """Derive fail-closed objective admission from self-validating evidence."""
    reasons: list[GroundSolverEligibilityReason] = []
    if result_status is not GroundResultStatus.COMPLETE:
        reasons.append(GroundSolverEligibilityReason.RESULT_NOT_COMPLETE)
    if termination_reason is not GroundTerminationReason.REST:
        reasons.append(GroundSolverEligibilityReason.NOT_REST_TERMINATED)
    if profile is None:
        reasons.append(GroundSolverEligibilityReason.MISSING_PROFILE_BINDING)
    else:
        if profile.qualification_status is GroundQualificationStatus.UNQUALIFIED:
            reasons.append(GroundSolverEligibilityReason.PROFILE_UNQUALIFIED)
        if profile.model_use_status is GroundModelUseStatus.ILLUSTRATIVE:
            reasons.append(GroundSolverEligibilityReason.PROFILE_ILLUSTRATIVE)
    if calibration.kind not in {CalibrationKind.MEASURED, CalibrationKind.LITERATURE}:
        reasons.append(GroundSolverEligibilityReason.MODEL_CALIBRATION_NOT_VALIDATED)
    if calibration.confidence <= 0.0:
        reasons.append(GroundSolverEligibilityReason.MODEL_CALIBRATION_ZERO_CONFIDENCE)
    canonical = tuple(item for item in GroundSolverEligibilityReason if item in reasons)
    if not canonical:
        canonical = (GroundSolverEligibilityReason.ELIGIBLE,)
    return GroundSolverEligibility(not reasons, canonical)


__all__ = ["derive_solver_eligibility", "derive_study_status"]
