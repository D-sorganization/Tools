"""Immutable launch-monitor convention and comparability contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = "launch-monitor-conventions/v1"


class ConventionId(StrEnum):
    """Supported calculation/presentation convention families."""

    APP_NATIVE = "app_native"
    TRACKMAN_COMPARABLE = "trackman_comparable"
    FORESIGHT_COMPARABLE = "foresight_comparable"


class ParameterId(StrEnum):
    """Foundation quantities shared by every convention."""

    CLUB_SPEED = "club_speed"
    CLUB_PATH = "club_path"
    ATTACK_ANGLE = "attack_angle"
    FACE_ANGLE = "face_angle"
    DYNAMIC_LOFT = "dynamic_loft"
    FACE_TO_PATH = "face_to_path"
    SPIN_LOFT = "spin_loft"
    LAUNCH_DIRECTION = "launch_direction"


class ReferencePoint(StrEnum):
    """Physical datum at which a quantity is defined."""

    TRACKED_HEAD_REFERENCE = "tracked_head_reference"
    GEOMETRIC_CENTER = "geometric_center"
    FACE_CENTER = "face_center"
    IMPACT_LOCATION = "impact_location"
    MIXED_CLUB_DELIVERY = "mixed_club_delivery"
    BALL_CENTER = "ball_center"


class EventTime(StrEnum):
    """Event-time policy associated with a reported value."""

    INSPECTION_EVENT = "inspection_event"
    JUST_BEFORE_FIRST_CONTACT = "just_before_first_contact"
    IMPACT = "impact"
    MAXIMUM_COMPRESSION = "maximum_compression"
    JUST_AFTER_SEPARATION = "just_after_separation"


class SignRule(StrEnum):
    """Typed direction rule instead of an undocumented signed scalar."""

    NONNEGATIVE = "nonnegative"
    POSITIVE_RIGHT = "positive_right"
    POSITIVE_UP = "positive_up"


class QuantityStatus(StrEnum):
    """Relationship between app output and a physical/device quantity."""

    DERIVED = "derived"
    MODELED = "modeled"
    MEASURED_COMPARABLE = "measured_comparable"


class AvailabilityRule(StrEnum):
    """Minimum state needed to report a definition."""

    ALWAYS = "always"
    NONZERO_CLUB_TRAVEL = "nonzero_club_travel"
    FACE_GEOMETRY = "face_geometry"
    COLLISION_COMPLETE = "collision_complete"


class ComparabilityReason(StrEnum):
    """Typed reason two values must not be silently subtracted."""

    PARAMETER = "parameter"
    REFERENCE_POINT = "reference_point"
    EVENT_TIME = "event_time"
    FRAME = "frame"
    GEOMETRY = "geometry"
    UNIT = "unit"
    AVAILABILITY = "availability"


@dataclass(frozen=True)
class ParameterDefinition:
    """Complete provenance and geometry contract for one parameter."""

    convention_id: ConventionId
    parameter_id: ParameterId
    label: str
    source_url: str
    retrieved_on: str
    reference_point: ReferencePoint
    event_time: EventTime
    frame_id: str
    geometry_contract: str
    sign_rule: SignRule
    unit: str
    quantity_status: QuantityStatus
    availability: AvailabilityRule

    def __post_init__(self) -> None:
        for name in ("label", "frame_id", "geometry_contract", "unit"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        parsed = urlparse(self.source_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("source_url must be an absolute HTTPS URL")
        try:
            date.fromisoformat(self.retrieved_on)
        except ValueError as error:
            raise ValueError("retrieved_on must be an ISO date") from error

    @property
    def key(self) -> str:
        """Return the stable convention-qualified identifier."""
        return f"{self.convention_id.value}.{self.parameter_id.value}"

    def to_dict(self) -> dict[str, str]:
        """Return a strict JSON-ready record with enum values flattened."""
        return {
            name: value.value if isinstance(value, StrEnum) else value
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ParameterDefinition:
        """Validate and construct one definition from a wire record."""
        required = {field.name for field in cls.__dataclass_fields__.values()}
        if set(payload) != required:
            raise ValueError("parameter definition fields do not match v1 schema")
        return cls(
            convention_id=ConventionId(payload["convention_id"]),
            parameter_id=ParameterId(payload["parameter_id"]),
            label=str(payload["label"]),
            source_url=str(payload["source_url"]),
            retrieved_on=str(payload["retrieved_on"]),
            reference_point=ReferencePoint(payload["reference_point"]),
            event_time=EventTime(payload["event_time"]),
            frame_id=str(payload["frame_id"]),
            geometry_contract=str(payload["geometry_contract"]),
            sign_rule=SignRule(payload["sign_rule"]),
            unit=str(payload["unit"]),
            quantity_status=QuantityStatus(payload["quantity_status"]),
            availability=AvailabilityRule(payload["availability"]),
        )


@dataclass(frozen=True)
class ComparisonCompatibility:
    """Compatibility decision with exhaustive, stable reason codes."""

    comparable: bool
    reasons: tuple[ComparabilityReason, ...]


@dataclass(frozen=True)
class ConventionRegistry:
    """Versioned deterministic collection of convention definitions."""

    definitions: tuple[ParameterDefinition, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        ordered = tuple(sorted(self.definitions, key=lambda item: item.key))
        if len({item.key for item in ordered}) != len(ordered):
            raise ValueError("parameter definition keys must be unique")
        object.__setattr__(self, "definitions", ordered)

    def definition(
        self, convention: ConventionId, parameter: ParameterId
    ) -> ParameterDefinition:
        """Return one exact definition or fail instead of guessing."""
        key = f"{convention.value}.{parameter.value}"
        for definition in self.definitions:
            if definition.key == key:
                return definition
        raise KeyError(key)

    def for_convention(
        self, convention: ConventionId
    ) -> tuple[ParameterDefinition, ...]:
        """Return the deterministically ordered convention catalog."""
        return tuple(
            item for item in self.definitions if item.convention_id is convention
        )

    def to_json(self) -> str:
        """Serialize deterministically for hashing, caching, and evidence."""
        payload = {
            "definitions": [item.to_dict() for item in self.definitions],
            "schema_version": self.schema_version,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ConventionRegistry:
        """Load v1 or migrate the explicit v0 `vendor` field rename."""
        version = payload.get("schema_version")
        if version not in {SCHEMA_VERSION, "launch-monitor-conventions/v0"}:
            raise ValueError(f"unsupported schema_version: {version}")
        raw_definitions = payload.get("definitions")
        if not isinstance(raw_definitions, list):
            raise ValueError("definitions must be a list")
        definitions = []
        for raw in raw_definitions:
            if not isinstance(raw, dict):
                raise ValueError("each definition must be an object")
            migrated = dict(raw)
            if "vendor" in migrated:
                migrated["convention_id"] = migrated.pop("vendor")
            definitions.append(ParameterDefinition.from_dict(migrated))
        return cls(tuple(definitions))


def compare_definitions(
    first: ParameterDefinition, second: ParameterDefinition
) -> ComparisonCompatibility:
    """Decide whether two values may be directly compared without transforms."""
    checks = (
        (ComparabilityReason.PARAMETER, first.parameter_id, second.parameter_id),
        (
            ComparabilityReason.REFERENCE_POINT,
            first.reference_point,
            second.reference_point,
        ),
        (ComparabilityReason.EVENT_TIME, first.event_time, second.event_time),
        (ComparabilityReason.FRAME, first.frame_id, second.frame_id),
        (
            ComparabilityReason.GEOMETRY,
            first.geometry_contract,
            second.geometry_contract,
        ),
        (ComparabilityReason.UNIT, first.unit, second.unit),
        (ComparabilityReason.AVAILABILITY, first.availability, second.availability),
    )
    reasons = tuple(reason for reason, left, right in checks if left != right)
    return ComparisonCompatibility(not reasons, reasons)
