"""Stable identifiers and schema versions for ground material profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

GROUND_MATERIAL_PROFILE_SCHEMA_VERSION = "ground-material-profile/v1"
GROUND_PROFILE_LIBRARY_SCHEMA_VERSION = "ground-profile-library/v1"


class GroundParameterId(StrEnum):
    """Canonical solver-facing material parameter identifiers."""

    NORMAL_RESTITUTION = "normal_restitution"
    STATIC_FRICTION = "static_friction"
    KINETIC_FRICTION = "kinetic_friction"
    ROLLING_RESISTANCE = "rolling_resistance"
    FIRMNESS_PA = "firmness_pa"
    HARDNESS_FRACTION = "hardness_fraction"
    GRASS_HEIGHT_M = "grass_height_m"
    COMPRESSIBILITY_FRACTION = "compressibility_fraction"
    COMPRESSION_DAMPING_FRACTION = "compression_damping_fraction"
    TURF_DENSITY_KG_M3 = "turf_density_kg_m3"
    MOISTURE_FRACTION = "moisture_fraction"


CANONICAL_GROUND_PARAMETER_IDS = tuple(GroundParameterId)


class GroundEvidenceKind(StrEnum):
    """Source classes permitted by the v1 evidence record."""

    MEASURED_DATASET = "measured_dataset"
    PEER_REVIEWED_LITERATURE = "peer_reviewed_literature"
    MANUFACTURER_SPECIFICATION = "manufacturer_specification"
    ENGINEERING_ESTIMATE = "engineering_estimate"


class GroundQualificationGateId(StrEnum):
    """Stable ordered qualification gates for one profile."""

    EVIDENCE_TRACEABLE = "evidence_traceable"
    VALIDITY_BOUNDS_TRACEABLE = "validity_bounds_traceable"
    RIGHTS_REUSABLE = "rights_reusable"
    UNCERTAINTY_DECLARED = "uncertainty_declared"
    CALIBRATION_TRACEABLE = "calibration_traceable"
    APPLICABILITY_BOUNDED = "applicability_bounded"
    PROVENANCE_REPRODUCIBLE = "provenance_reproducible"


class GroundQualificationStatus(StrEnum):
    """Derived profile qualification status."""

    QUALIFIED = "qualified"
    UNQUALIFIED = "unqualified"


class GroundModelUseStatus(StrEnum):
    """Fail-closed model-use classification derived from qualification gates."""

    ILLUSTRATIVE = "illustrative"
    CALIBRATED = "calibrated"


__all__ = [
    "CANONICAL_GROUND_PARAMETER_IDS",
    "GROUND_MATERIAL_PROFILE_SCHEMA_VERSION",
    "GROUND_PROFILE_LIBRARY_SCHEMA_VERSION",
    "GroundEvidenceKind",
    "GroundModelUseStatus",
    "GroundParameterId",
    "GroundQualificationGateId",
    "GroundQualificationStatus",
]
