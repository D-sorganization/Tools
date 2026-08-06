"""Canonical launch-direction conventions and lossless legacy migration.

Launch Direction is the horizontal angle at launch relative to the target
line.  The application and launch-monitor-comparable conventions use
positive right / negative left.  The internal flight frame uses +y left,
so its azimuth has the opposite sign.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import Final

TRACKMAN_LAUNCH_DIRECTION_SOURCE: Final = (
    "https://www.trackman.com/blog/golf/what-is-launch-direction"
)
CANONICAL_DIRECTION_KEY: Final = "launch_direction_deg"
CONVENTION_KEY: Final = "launch_direction_convention"
SCHEMA_VERSION_KEY: Final = "launch_direction_schema_version"
SCHEMA_VERSION: Final = 1
LEGACY_DIRECTION_KEYS: Final = ("launch_azimuth_deg", "azimuth_deg")


class LaunchDirectionConvention(StrEnum):
    """Supported numeric sign conventions for horizontal launch direction."""

    APP_NATIVE = "app_native"
    LAUNCH_MONITOR_COMPARABLE = "launch_monitor_comparable"
    FLIGHT_FRAME = "flight_frame"


@dataclass(frozen=True)
class LaunchDirectionDefinition:
    """Human-readable convention metadata carried beside numeric values."""

    positive_direction: str
    negative_direction: str
    reference: str
    source_url: str | None
    retrieved_on: str | None
    definition_version: str
    comparability_status: str


DEFINITIONS: Final[dict[LaunchDirectionConvention, LaunchDirectionDefinition]] = {
    LaunchDirectionConvention.APP_NATIVE: LaunchDirectionDefinition(
        positive_direction="right of the target line",
        negative_direction="left of the target line",
        reference="horizontal angle from the target line",
        source_url=None,
        retrieved_on=None,
        definition_version="roc-launch-direction-v1",
        comparability_status="canonical",
    ),
    LaunchDirectionConvention.LAUNCH_MONITOR_COMPARABLE: LaunchDirectionDefinition(
        positive_direction="right of the target line",
        negative_direction="left of the target line",
        reference=(
            "horizontal ball-CG motion relative to the target line after separation"
        ),
        source_url=TRACKMAN_LAUNCH_DIRECTION_SOURCE,
        retrieved_on="2026-08-06",
        definition_version="trackman-public-definition-2026-08-06",
        comparability_status="definition-and-sign-comparable",
    ),
    LaunchDirectionConvention.FLIGHT_FRAME: LaunchDirectionDefinition(
        positive_direction="left of the target line (+y flight)",
        negative_direction="right of the target line (-y flight)",
        reference="horizontal angle from +x in the internal flight frame",
        source_url=None,
        retrieved_on=None,
        definition_version="swing-sim-flight-frame-v1",
        comparability_status="internal-only",
    ),
}


def _validated_degrees(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("launch direction must be a real number, not bool")
    degrees = float(value)
    if not math.isfinite(degrees):
        raise ValueError("launch direction must be finite")
    if not -180.0 <= degrees <= 180.0:
        raise ValueError("launch direction must be within [-180, 180] degrees")
    return degrees


def _right_positive(degrees: float, convention: LaunchDirectionConvention) -> float:
    return -degrees if convention is LaunchDirectionConvention.FLIGHT_FRAME else degrees


@dataclass(frozen=True)
class LaunchDirection:
    """A direction value whose sign convention cannot be implicit."""

    degrees: float
    convention: LaunchDirectionConvention

    def __post_init__(self) -> None:
        object.__setattr__(self, "degrees", _validated_degrees(self.degrees))
        if not isinstance(self.convention, LaunchDirectionConvention):
            raise TypeError("convention must be a LaunchDirectionConvention")

    def to(self, target: LaunchDirectionConvention) -> LaunchDirection:
        """Convert to *target* without changing the represented direction."""
        if not isinstance(target, LaunchDirectionConvention):
            raise TypeError("target must be a LaunchDirectionConvention")
        canonical = _right_positive(self.degrees, self.convention)
        converted = (
            -canonical
            if target is LaunchDirectionConvention.FLIGHT_FRAME
            else canonical
        )
        return LaunchDirection(converted, target)


def launch_direction_to_flight_azimuth(direction: LaunchDirection) -> float:
    """Return the internal left-positive flight-frame azimuth in degrees."""
    return direction.to(LaunchDirectionConvention.FLIGHT_FRAME).degrees


def migrate_launch_direction_mapping(values: Mapping[str, object]) -> dict[str, object]:
    """Add canonical direction fields while preserving every imported field.

    Legacy fields are interpreted as the historical app-native convention.
    Duplicate aliases are accepted only when they agree, preventing silent
    corruption during mixed-version imports.
    """
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
    raw_convention = values.get(
        CONVENTION_KEY, LaunchDirectionConvention.APP_NATIVE.value
    )
    if not isinstance(raw_convention, str):
        raise ValueError(f"unknown launch-direction convention: {raw_convention!r}")
    try:
        convention = LaunchDirectionConvention(raw_convention)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"unknown launch-direction convention: {raw_convention!r}"
        ) from exc
    migrated[CANONICAL_DIRECTION_KEY] = first_value
    migrated[CONVENTION_KEY] = convention.value
    migrated[SCHEMA_VERSION_KEY] = SCHEMA_VERSION
    return migrated


def launch_direction_from_mapping(values: Mapping[str, object]) -> LaunchDirection:
    """Parse canonical or legacy imported fields as a typed direction."""
    migrated = migrate_launch_direction_mapping(values)
    degrees = _validated_degrees(migrated[CANONICAL_DIRECTION_KEY])
    return LaunchDirection(
        degrees,
        LaunchDirectionConvention(str(migrated[CONVENTION_KEY])),
    )


__all__ = [
    "CANONICAL_DIRECTION_KEY",
    "CONVENTION_KEY",
    "DEFINITIONS",
    "LEGACY_DIRECTION_KEYS",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_KEY",
    "TRACKMAN_LAUNCH_DIRECTION_SOURCE",
    "LaunchDirection",
    "LaunchDirectionConvention",
    "LaunchDirectionDefinition",
    "launch_direction_from_mapping",
    "launch_direction_to_flight_azimuth",
    "migrate_launch_direction_mapping",
]
