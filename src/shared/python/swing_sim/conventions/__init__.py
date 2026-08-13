"""Public façade for launch-monitor convention contracts."""

from .catalog import convention_registry
from .registry import (
    AvailabilityRule,
    ComparabilityReason,
    ComparisonCompatibility,
    ConventionId,
    ConventionRegistry,
    EventTime,
    ParameterDefinition,
    ParameterId,
    QuantityStatus,
    ReferencePoint,
    SignRule,
    compare_definitions,
)
from .transforms import Matrix3, Vector3, shift_point_velocity, transform_vector

__all__ = [
    "AvailabilityRule",
    "ComparabilityReason",
    "ComparisonCompatibility",
    "ConventionId",
    "ConventionRegistry",
    "EventTime",
    "Matrix3",
    "ParameterDefinition",
    "ParameterId",
    "QuantityStatus",
    "ReferencePoint",
    "SignRule",
    "Vector3",
    "compare_definitions",
    "convention_registry",
    "shift_point_velocity",
    "transform_vector",
]
