"""Trajectory-derived values for the canonical ball-flight metric catalog."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import TypeAlias

from .result_contract import (
    AvailabilityReason,
    FlightMetricId,
    ValueStatus,
)

Vector3: TypeAlias = tuple[float, float, float]
MetricNumber: TypeAlias = float | Vector3
_EXPORT_DECIMAL_PLACES = 11


def _vector(value: Vector3, name: str) -> Vector3:
    if len(value) != 3 or not all(math.isfinite(component) for component in value):
        raise ValueError(f"{name} must contain three finite components")
    return tuple(float(component) for component in value)  # type: ignore[return-value]


def _wire_number(value: float) -> float | int:
    rounded = round(value, _EXPORT_DECIMAL_PLACES)
    return int(rounded) if rounded.is_integer() else rounded


@dataclass(frozen=True)
class MetricTrajectoryPoint:
    """One target-frame trajectory sample in SI units."""

    time_s: float
    position_m: Vector3
    velocity_m_s: Vector3

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("time_s must be finite and nonnegative")
        object.__setattr__(self, "position_m", _vector(self.position_m, "position_m"))
        object.__setattr__(
            self, "velocity_m_s", _vector(self.velocity_m_s, "velocity_m_s")
        )


@dataclass(frozen=True)
class GroundModelResult:
    """Outputs supplied by an identified, qualified bounce-and-roll model."""

    model_id: str
    total_distance_m: float
    roll_distance_m: float
    bounce_count: int
    final_offline_m: float

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("ground model_id must be nonempty")
        values = (
            self.total_distance_m,
            self.roll_distance_m,
            self.bounce_count,
            self.final_offline_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ground outputs must be finite")
        if isinstance(self.bounce_count, bool) or not isinstance(
            self.bounce_count, int
        ):
            raise ValueError("bounce_count must be an integer")
        if min(self.total_distance_m, self.roll_distance_m, self.bounce_count) < 0.0:
            raise ValueError("ground distances and bounce_count must be nonnegative")


ManifestFields: TypeAlias = tuple[tuple[str, str], ...]


def _manifest_fields(value: ManifestFields, name: str) -> ManifestFields:
    normalized = tuple(
        sorted((str(key).strip(), str(item).strip()) for key, item in value)
    )
    if any(not key or not item for key, item in normalized):
        raise ValueError(f"{name} keys and values must be nonempty")
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError(f"{name} keys must be unique")
    return normalized


@dataclass(frozen=True)
class FlightRunManifest:
    """Reproducibility metadata that accompanies every metric result."""

    model_id: str
    model_version: str
    integration_status: str
    termination_reason: str
    environment: ManifestFields
    wind: ManifestFields
    uncertainty_status: str
    frame_id: str = "target_frame:x_downrange,y_up,z_right"

    def __post_init__(self) -> None:
        text = (
            self.model_id,
            self.model_version,
            self.integration_status,
            self.termination_reason,
            self.uncertainty_status,
            self.frame_id,
        )
        if any(not value.strip() for value in text):
            raise ValueError("manifest text fields must be nonempty")
        object.__setattr__(
            self, "environment", _manifest_fields(self.environment, "environment")
        )
        object.__setattr__(self, "wind", _manifest_fields(self.wind, "wind"))


@dataclass(frozen=True)
class FlightMetricInputs:
    """Validated raw state required to derive canonical flight values."""

    trajectory: tuple[MetricTrajectoryPoint, ...]
    spin_vector_rpm: Vector3
    target_position_m: Vector3 | None = None
    ground_result: GroundModelResult | None = None

    def __post_init__(self) -> None:
        points = tuple(self.trajectory)
        if any(
            next_point.time_s <= point.time_s
            for point, next_point in zip(points, points[1:], strict=False)
        ):
            raise ValueError("trajectory times must be strictly increasing")
        object.__setattr__(self, "trajectory", points)
        object.__setattr__(
            self, "spin_vector_rpm", _vector(self.spin_vector_rpm, "spin_vector_rpm")
        )
        if self.target_position_m is not None:
            object.__setattr__(
                self,
                "target_position_m",
                _vector(self.target_position_m, "target_position_m"),
            )

    def with_ground_result(self, result: GroundModelResult) -> FlightMetricInputs:
        """Return a copy carrying explicitly qualified ground-model outputs."""
        return replace(self, ground_result=result)


@dataclass(frozen=True)
class FlightMetricValue:
    """One available or typed-unavailable metric value."""

    metric_id: FlightMetricId
    status: ValueStatus
    numeric: MetricNumber | None
    reason: AvailabilityReason | None
    provenance: str

    def __post_init__(self) -> None:
        if not self.provenance.strip():
            raise ValueError("value provenance must be nonempty")
        if self.status is ValueStatus.UNAVAILABLE:
            if self.numeric is not None or self.reason is None:
                raise ValueError(
                    "unavailable values require a reason and no numeric value"
                )
        elif self.numeric is None or self.reason is not None:
            raise ValueError("available values require numeric data and no reason")
        if isinstance(self.numeric, tuple):
            _vector(self.numeric, "numeric")
        elif self.numeric is not None and not math.isfinite(self.numeric):
            raise ValueError("numeric metric values must be finite")

    def to_dict(self) -> dict[str, object]:
        """Return deterministic wire keys and flattened enum values."""
        if isinstance(self.numeric, tuple):
            numeric: object = [_wire_number(value) for value in self.numeric]
        elif self.numeric is not None:
            numeric = _wire_number(self.numeric)
        else:
            numeric = None
        return {
            "metric_id": self.metric_id.value,
            "numeric": numeric,
            "provenance": self.provenance,
            "reason": self.reason.value if self.reason else None,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class FlightMetricResult:
    """Complete run manifest plus one value for every catalog definition."""

    manifest: FlightRunManifest
    values: tuple[FlightMetricValue, ...]
    schema_version: str = "ball-flight-result/v1"

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.values, key=lambda item: item.metric_id.value))
        if {item.metric_id for item in ordered} != set(FlightMetricId):
            raise ValueError("result must contain each canonical metric exactly once")
        object.__setattr__(self, "values", ordered)

    def value(self, metric_id: FlightMetricId) -> FlightMetricValue:
        """Return one metric value by stable ID."""
        for value in self.values:
            if value.metric_id is metric_id:
                return value
        raise KeyError(metric_id.value)

    def scalar(self, metric_id: FlightMetricId) -> float:
        """Return an available scalar or fail instead of coercing unavailability."""
        numeric = self.value(metric_id).numeric
        if numeric is None or isinstance(numeric, tuple):
            raise ValueError(f"{metric_id.value} is not an available scalar")
        return numeric

    def vector(self, metric_id: FlightMetricId) -> Vector3:
        """Return an available vector or fail instead of coercing unavailability."""
        numeric = self.value(metric_id).numeric
        if not isinstance(numeric, tuple):
            raise ValueError(f"{metric_id.value} is not an available vector")
        return numeric

    def to_json(self) -> str:
        """Serialize deterministically for API/export parity."""
        manifest = asdict(self.manifest)
        manifest["environment"] = dict(self.manifest.environment)
        manifest["wind"] = dict(self.manifest.wind)
        payload = {
            "manifest": manifest,
            "schema_version": self.schema_version,
            "values": [value.to_dict() for value in self.values],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def derive_flight_metric_result(
    inputs: FlightMetricInputs, manifest: FlightRunManifest
) -> FlightMetricResult:
    """Derive every canonical value without fabricating missing model output."""
    from .result_derivation import derive_flight_metric_result as derive

    return derive(inputs, manifest)


__all__ = [
    "FlightMetricInputs",
    "FlightMetricResult",
    "FlightMetricValue",
    "FlightRunManifest",
    "GroundModelResult",
    "MetricTrajectoryPoint",
    "derive_flight_metric_result",
]
