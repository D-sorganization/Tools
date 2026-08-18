"""Registry-backed launch-direction conversion and legacy migration.

Public conventions are owned by :mod:`shared.python.swing_sim.conventions`.
Every catalogued launch direction is target-frame, degree-valued, and
positive right.  The internal flight frame is x forward / y left / z up,
so conversion to its azimuth is an explicit sign flip rather than another
vendor convention.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Final, TypeAlias

from shared.python.swing_sim.conventions import (
    ConventionId,
    ParameterDefinition,
    ParameterId,
    SignRule,
    convention_registry,
)

CANONICAL_DIRECTION_KEY: Final = "launch_direction_deg"
CONVENTION_KEY: Final = "launch_direction_convention"
SCHEMA_VERSION_KEY: Final = "launch_direction_schema_version"
SCHEMA_VERSION: Final = 1
LEGACY_DIRECTION_KEYS: Final = ("launch_azimuth_deg", "azimuth_deg")

# Backward-compatible public type name, now an alias to the canonical registry ID.
LaunchDirectionConvention: TypeAlias = ConventionId

_LEGACY_CONVENTION_ALIASES: Final = {
    "launch_monitor_comparable": ConventionId.TRACKMAN_COMPARABLE.value,
}
_SUPPORTED_CONVENTIONS: Final = (
    ConventionId.APP_NATIVE,
    ConventionId.TRACKMAN_COMPARABLE,
)

_registry = convention_registry()
DEFINITIONS: Final[Mapping[ConventionId, ParameterDefinition]] = MappingProxyType(
    {
        convention: _registry.definition(convention, ParameterId.LAUNCH_DIRECTION)
        for convention in _SUPPORTED_CONVENTIONS
    }
)


def _validated_degrees(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("launch direction must be a real number, not bool")
    degrees = float(value)
    if not math.isfinite(degrees):
        raise ValueError("launch direction must be finite")
    if not -180.0 <= degrees <= 180.0:
        raise ValueError("launch direction must be within [-180, 180] degrees")
    return degrees


def _validated_convention(value: object) -> ConventionId:
    if not isinstance(value, (str, ConventionId)):
        raise ValueError(f"unknown launch-direction convention: {value!r}")
    canonical = _LEGACY_CONVENTION_ALIASES.get(str(value), str(value))
    try:
        convention = ConventionId(canonical)
        definition = DEFINITIONS[convention]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"unsupported launch-direction convention: {value!r}") from exc
    if definition.sign_rule is not SignRule.POSITIVE_RIGHT:
        raise ValueError(
            f"unsupported launch-direction sign rule: {definition.sign_rule.value}"
        )
    return convention


@dataclass(frozen=True)
class LaunchDirection:
    """A direction value tied to one source-backed registry convention."""

    degrees: float
    convention: ConventionId

    def __post_init__(self) -> None:
        object.__setattr__(self, "degrees", _validated_degrees(self.degrees))
        object.__setattr__(self, "convention", _validated_convention(self.convention))

    @property
    def definition(self) -> ParameterDefinition:
        """Return the canonical provenance and geometry definition."""
        return DEFINITIONS[self.convention]

    def to(self, target: ConventionId) -> LaunchDirection:
        """Convert through registry sign rules without changing physical state."""
        return LaunchDirection(self.degrees, _validated_convention(target))


def launch_direction_to_flight_azimuth(direction: LaunchDirection) -> float:
    """Return the internal left-positive flight-frame azimuth in degrees."""
    _validated_convention(direction.convention)
    return -direction.degrees


def launch_direction_sign_labels(convention: ConventionId) -> tuple[str, str]:
    """Return positive/negative labels derived from the registry sign rule."""
    _validated_convention(convention)
    return "right of the target line", "left of the target line"


def migrate_launch_direction_mapping(values: Mapping[str, object]) -> dict[str, object]:
    """Add canonical fields while preserving all imported fields and aliases."""
    migrated = dict(values)
    present = [
        (key, _validated_degrees(values[key]))
        for key in (CANONICAL_DIRECTION_KEY, *LEGACY_DIRECTION_KEYS)
        if key in values
    ]
    if not present:
        raise KeyError("no launch-direction field found")
    first_key, first_value = present[0]
    for key, value in present[1:]:
        if not math.isclose(first_value, value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"conflicting launch-direction values in {first_key!r} and {key!r}"
            )
    convention = _validated_convention(
        values.get(CONVENTION_KEY, ConventionId.APP_NATIVE.value)
    )
    migrated[CANONICAL_DIRECTION_KEY] = first_value
    migrated[CONVENTION_KEY] = convention.value
    migrated[SCHEMA_VERSION_KEY] = SCHEMA_VERSION
    return migrated


def launch_direction_from_mapping(values: Mapping[str, object]) -> LaunchDirection:
    """Parse canonical or legacy imported fields as a typed direction."""
    migrated = migrate_launch_direction_mapping(values)
    return LaunchDirection(
        _validated_degrees(migrated[CANONICAL_DIRECTION_KEY]),
        _validated_convention(migrated[CONVENTION_KEY]),
    )


__all__ = [
    "CANONICAL_DIRECTION_KEY",
    "CONVENTION_KEY",
    "DEFINITIONS",
    "LEGACY_DIRECTION_KEYS",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_KEY",
    "LaunchDirection",
    "LaunchDirectionConvention",
    "launch_direction_from_mapping",
    "launch_direction_sign_labels",
    "launch_direction_to_flight_azimuth",
    "migrate_launch_direction_mapping",
]
