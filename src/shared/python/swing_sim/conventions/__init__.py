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
    ParameterGroup,
    ParameterId,
    QuantityStatus,
    ReferencePoint,
    SignRule,
    compare_definitions,
    parameter_group,
    parameter_group_label,
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
    "ParameterGroup",
    "ParameterId",
    "QuantityStatus",
    "ReferencePoint",
    "SignRule",
    "Vector3",
    "compare_definitions",
    "convention_registry",
    "parameter_group",
    "parameter_group_label",
    "shift_point_velocity",
    "transform_vector",
]
