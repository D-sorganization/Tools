"""Canonical, versioned ball-flight result metric contracts.

The catalog is deliberately independent of an integrator and UI.  It defines
what a result means; :mod:`result_metrics` derives values from trajectory data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

SCHEMA_VERSION = "ball-flight-metrics/v1"
FRAME_ID = "target_frame:x_downrange,y_up,z_right"
APP_SOURCE = "https://github.com/D-sorganization/Tools/blob/main/docs/specs/BALL_FLIGHT_RESULT_CONTRACT.md"
TRACKMAN_SOURCE = "https://www.trackman.com/blog/golf/40-trackman-parameters"
FORESIGHT_SOURCE = "https://help.foresightsports.com/hc/en-us/articles/47144162581523-Ball-Launch-Data-Measurements-Ball-Flight-Results"


class FlightMetricId(StrEnum):
    """Stable IDs for launch, airborne-flight, target, and ground values."""

    INITIAL_VELOCITY = "initial_velocity"
    BALL_SPEED = "ball_speed"
    VERTICAL_LAUNCH_ANGLE = "vertical_launch_angle"
    LAUNCH_DIRECTION = "launch_direction"
    SPIN_VECTOR = "spin_vector"
    TOTAL_SPIN = "total_spin"
    SPIN_AXIS_TILT = "spin_axis_tilt"
    LANDING_POSITION = "landing_position"
    LANDING_VELOCITY = "landing_velocity"
    CARRY_DISTANCE = "carry_distance"
    CARRY_OFFLINE = "carry_offline"
    APEX_HEIGHT = "apex_height"
    FLIGHT_TIME = "flight_time"
    LANDING_ANGLE = "landing_angle"
    CURVE = "curve"
    TERMINAL_SPEED = "terminal_speed"
    TERMINAL_DIRECTION = "terminal_direction"
    TARGET_RESIDUAL = "target_residual"
    TARGET_DOWNRANGE_RESIDUAL = "target_downrange_residual"
    TARGET_LATERAL_RESIDUAL = "target_lateral_residual"
    TOTAL_DISTANCE = "total_distance"
    ROLL_DISTANCE = "roll_distance"
    BOUNCE_COUNT = "bounce_count"
    FINAL_OFFLINE = "final_offline"


class ValueStatus(StrEnum):
    """How a value entered a result; never infer measurement from modeling."""

    INPUT = "input"
    DIRECTLY_SIMULATED = "directly_simulated"
    DERIVED = "derived"
    MODEL_DEPENDENT = "model_dependent"
    ESTIMATED = "estimated"
    OPTIMIZED = "optimized"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"


class AvailabilityReason(StrEnum):
    """Typed reasons a requested value has no defensible numeric value."""

    INSUFFICIENT_TRAJECTORY = "insufficient_trajectory"
    NO_GROUND_CROSSING = "no_ground_crossing"
    ZERO_HORIZONTAL_SPEED = "zero_horizontal_speed"
    ZERO_SPIN = "zero_spin"
    TARGET_NOT_CONFIGURED = "target_not_configured"
    GROUND_MODEL_REQUIRED = "ground_model_required"


class ComparabilityStatus(StrEnum):
    """Relationship to a public convention definition."""

    NATIVE = "native"
    DEFINITION_ALIGNED = "definition_aligned"
    NOT_COMPARABLE = "not_comparable"


class SignRule(StrEnum):
    """Display sign rule in the declared target frame."""

    NONNEGATIVE = "nonnegative"
    POSITIVE_RIGHT = "positive_right"
    POSITIVE_UP = "positive_up"
    POSITIVE_DOWN = "positive_down"
    VECTOR_COMPONENTS = "vector_components"
    SIGNED = "signed"


@dataclass(frozen=True)
class ComparisonCoverage:
    """One explicit cell in the app/vendor coverage matrix."""

    convention_id: str
    status: ComparabilityStatus
    reason_code: str
    source_url: str

    def __post_init__(self) -> None:
        if not self.convention_id or not self.reason_code:
            raise ValueError("coverage convention_id and reason_code must be nonempty")
        parsed = urlparse(self.source_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("coverage source_url must be absolute HTTPS")

    def to_dict(self) -> dict[str, str]:
        """Return a strict JSON-ready coverage record."""
        return {
            "convention_id": self.convention_id,
            "reason_code": self.reason_code,
            "source_url": self.source_url,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ComparisonCoverage:
        """Load a strict coverage record."""
        if set(payload) != {"convention_id", "reason_code", "source_url", "status"}:
            raise ValueError("coverage fields do not match v1 schema")
        return cls(
            str(payload["convention_id"]),
            ComparabilityStatus(payload["status"]),
            str(payload["reason_code"]),
            str(payload["source_url"]),
        )


@dataclass(frozen=True)
class FlightMetricDefinition:
    """Complete display, geometry, provenance, and availability definition."""

    metric_id: FlightMetricId
    label: str
    definition: str
    unit: str
    default_status: ValueStatus
    frame_id: str
    sign_rule: SignRule
    reference_event: str
    geometry_contract: str
    provenance: str
    availability_rule: str
    solver_objective: bool
    coverage: tuple[ComparisonCoverage, ...]

    def __post_init__(self) -> None:
        text_fields = (
            self.label,
            self.definition,
            self.unit,
            self.frame_id,
            self.reference_event,
            self.geometry_contract,
            self.provenance,
            self.availability_rule,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("metric text fields must be nonempty")
        if {item.convention_id for item in self.coverage} != {
            "app_native",
            "trackman_comparable",
            "foresight_comparable",
        }:
            raise ValueError(
                "coverage must contain each supported convention exactly once"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-ready definition."""
        payload = asdict(self)
        payload["metric_id"] = self.metric_id.value
        payload["default_status"] = self.default_status.value
        payload["sign_rule"] = self.sign_rule.value
        payload["coverage"] = [item.to_dict() for item in self.coverage]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FlightMetricDefinition:
        """Load one definition and reject missing or additional fields."""
        required = {field.name for field in cls.__dataclass_fields__.values()}
        if set(payload) != required:
            raise ValueError("metric definition fields do not match v1 schema")
        coverage = payload["coverage"]
        if not isinstance(coverage, list):
            raise ValueError("coverage must be a list")
        if not isinstance(payload["solver_objective"], bool):
            raise ValueError("solver_objective must be a boolean")
        if any(not isinstance(item, dict) for item in coverage):
            raise ValueError("coverage entries must be objects")
        return cls(
            metric_id=FlightMetricId(payload["metric_id"]),
            label=str(payload["label"]),
            definition=str(payload["definition"]),
            unit=str(payload["unit"]),
            default_status=ValueStatus(payload["default_status"]),
            frame_id=str(payload["frame_id"]),
            sign_rule=SignRule(payload["sign_rule"]),
            reference_event=str(payload["reference_event"]),
            geometry_contract=str(payload["geometry_contract"]),
            provenance=str(payload["provenance"]),
            availability_rule=str(payload["availability_rule"]),
            solver_objective=payload["solver_objective"],
            coverage=tuple(ComparisonCoverage.from_dict(item) for item in coverage),
        )


@dataclass(frozen=True)
class FlightMetricCatalog:
    """Immutable, deterministic collection of metric definitions."""

    definitions: tuple[FlightMetricDefinition, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        ordered = tuple(sorted(self.definitions, key=lambda item: item.metric_id.value))
        if len({item.metric_id for item in ordered}) != len(ordered):
            raise ValueError("metric IDs must be unique")
        object.__setattr__(self, "definitions", ordered)

    def definition(self, metric_id: FlightMetricId) -> FlightMetricDefinition:
        """Return one exact definition, failing closed for unknown IDs."""
        for definition in self.definitions:
            if definition.metric_id is metric_id:
                return definition
        raise KeyError(metric_id.value)

    def to_json(self) -> str:
        """Serialize with stable key and definition order."""
        payload = {
            "definitions": [item.to_dict() for item in self.definitions],
            "schema_version": self.schema_version,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> FlightMetricCatalog:
        """Load only the strict v1 catalog schema."""
        if set(payload) != {"definitions", "schema_version"}:
            raise ValueError("catalog fields do not match v1 schema")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {payload['schema_version']}")
        definitions = payload["definitions"]
        if not isinstance(definitions, list):
            raise ValueError("definitions must be a list")
        if any(not isinstance(item, dict) for item in definitions):
            raise ValueError("metric definitions must be objects")
        return cls(
            tuple(FlightMetricDefinition.from_dict(item) for item in definitions)
        )


_PUBLICLY_ALIGNED = {
    FlightMetricId.BALL_SPEED,
    FlightMetricId.VERTICAL_LAUNCH_ANGLE,
    FlightMetricId.LAUNCH_DIRECTION,
    FlightMetricId.TOTAL_SPIN,
    FlightMetricId.SPIN_AXIS_TILT,
    FlightMetricId.CARRY_DISTANCE,
    FlightMetricId.CARRY_OFFLINE,
    FlightMetricId.APEX_HEIGHT,
    FlightMetricId.FLIGHT_TIME,
    FlightMetricId.LANDING_ANGLE,
    FlightMetricId.CURVE,
    FlightMetricId.TOTAL_DISTANCE,
    FlightMetricId.ROLL_DISTANCE,
}


def _coverage(metric_id: FlightMetricId) -> tuple[ComparisonCoverage, ...]:
    native = ComparisonCoverage(
        "app_native", ComparabilityStatus.NATIVE, "canonical_app_definition", APP_SOURCE
    )
    status = (
        ComparabilityStatus.DEFINITION_ALIGNED
        if metric_id in _PUBLICLY_ALIGNED
        else ComparabilityStatus.NOT_COMPARABLE
    )
    reason = (
        "modeled_not_measured"
        if status is ComparabilityStatus.DEFINITION_ALIGNED
        else "public_definition_not_established"
    )
    return (
        native,
        ComparisonCoverage("trackman_comparable", status, reason, TRACKMAN_SOURCE),
        ComparisonCoverage("foresight_comparable", status, reason, FORESIGHT_SOURCE),
    )


@lru_cache(maxsize=1)
def flight_metric_catalog() -> FlightMetricCatalog:
    """Return the canonical source-backed metric catalog."""
    from .result_catalog_data import IDENTITIES

    definitions = tuple(
        FlightMetricDefinition(
            metric_id=metric_id,
            label=identity.label,
            definition=identity.definition,
            unit=identity.unit,
            default_status=identity.status,
            frame_id=FRAME_ID,
            sign_rule=identity.sign,
            reference_event=identity.event,
            geometry_contract=identity.geometry,
            provenance=APP_SOURCE,
            availability_rule=identity.availability,
            solver_objective=identity.solver,
            coverage=_coverage(metric_id),
        )
        for metric_id, identity in IDENTITIES.items()
    )
    return FlightMetricCatalog(definitions)


__all__ = [
    "AvailabilityReason",
    "ComparabilityStatus",
    "FlightMetricCatalog",
    "FlightMetricDefinition",
    "FlightMetricId",
    "SCHEMA_VERSION",
    "SignRule",
    "ValueStatus",
    "flight_metric_catalog",
]
