"""Strict request and evaluator contracts for impact solution families."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .inverse_contract import EvaluatedMetric, InverseFlightRequest, SolverEvaluation
from .result_contract import FlightMetricId

REQUEST_SCHEMA_VERSION = "impact-solution-request/v1"
RESULT_SCHEMA_VERSION = "impact-solution-result/v1"
TARGET_FRAME_ID = "target_frame:x_downrange,y_up,z_right"
DELIVERY_FRAME_ID = "app_frame:x_target,y_up,z_right"
IMPACT_REFERENCE_POINT = "ball_center_at_first_contact"
CONVENTION_ID = "app_native"
IMPACT_MODEL_ID = "rigid_body_centered"

_DELIVERY_UNITS = {
    "attack_angle_deg": "deg",
    "club_path_deg": "deg",
    "clubhead_speed_mps": "m/s",
    "dynamic_loft_deg": "deg",
    "face_angle_deg": "deg",
}
_DELIVERY_LIMITS = {
    "attack_angle_deg": (-89.0, 89.0),
    "club_path_deg": (-89.0, 89.0),
    "clubhead_speed_mps": (1e-9, 100.0),
    "dynamic_loft_deg": (-89.0, 89.0),
    "face_angle_deg": (-89.0, 89.0),
}


class ClubProfileId(StrEnum):
    """Supported centered clubhead parameter sets."""

    CENTERED_DRIVER = "centered_driver"
    CENTERED_IRON = "centered_iron"


class ModelAvailability(StrEnum):
    """Whether one requested forward model can be evaluated locally."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ForwardStatus(StrEnum):
    """Typed outcome of a delivery-to-flight forward evaluation."""

    COMPLETE = "complete"
    NO_IMPACT = "no_impact"
    MODEL_UNAVAILABLE = "model_unavailable"
    FAILED = "failed"
    NONCONVERGED = "nonconverged"


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _text(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


def _exact(payload: dict[str, Any], fields: set[str], name: str) -> None:
    if set(payload) != fields:
        raise ValueError(f"{name} fields do not match v1 schema")


def _wire_number(value: float) -> float | int:
    rounded = round(value, 11)
    return int(rounded) if rounded.is_integer() else rounded


@dataclass(frozen=True)
class ModelManifest:
    """Declared model identities, availability and implementation provenance."""

    impact_model_id: str
    impact_status: ModelAvailability
    flight_model_id: str
    flight_status: ModelAvailability
    provenance: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "impact_status", ModelAvailability(self.impact_status))
        object.__setattr__(self, "flight_status", ModelAvailability(self.flight_status))
        for field_name in ("impact_model_id", "flight_model_id"):
            object.__setattr__(
                self, field_name, _text(getattr(self, field_name), field_name)
            )
        normalized = tuple(sorted(self.provenance))
        if not normalized or any(
            not key.strip() or not value.strip() for key, value in normalized
        ):
            raise ValueError("model provenance must contain nonempty keys and values")
        if len({key for key, _ in normalized}) != len(normalized):
            raise ValueError("model provenance keys must be unique")
        object.__setattr__(self, "provenance", normalized)

    def to_dict(self) -> dict[str, object]:
        """Return the strict wire record."""
        return {
            "flight_model_id": self.flight_model_id,
            "flight_status": self.flight_status.value,
            "impact_model_id": self.impact_model_id,
            "impact_status": self.impact_status.value,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelManifest:
        """Parse one strict model manifest."""
        _exact(
            payload,
            {
                "flight_model_id",
                "flight_status",
                "impact_model_id",
                "impact_status",
                "provenance",
            },
            "model manifest",
        )
        if not isinstance(payload["provenance"], dict):
            raise ValueError("model manifest provenance must be an object")
        return cls(
            str(payload["impact_model_id"]),
            ModelAvailability(payload["impact_status"]),
            str(payload["flight_model_id"]),
            ModelAvailability(payload["flight_status"]),
            tuple(
                (str(key), str(value)) for key, value in payload["provenance"].items()
            ),
        )


@dataclass(frozen=True)
class ImpactSolutionRequest:
    """A fully declared desired-flight to delivery-family request."""

    inverse_request: InverseFlightRequest
    club_profile_id: ClubProfileId
    flight_model_id: str
    family_count: int
    family_radius: float
    sensitivity_fraction: float
    impact_event_time_s: float
    target_frame_id: str = TARGET_FRAME_ID
    delivery_frame_id: str = DELIVERY_FRAME_ID
    impact_reference_point: str = IMPACT_REFERENCE_POINT
    convention_id: str = CONVENTION_ID
    impact_model_id: str = IMPACT_MODEL_ID
    schema_version: str = REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REQUEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        expected = {
            "target_frame_id": TARGET_FRAME_ID,
            "delivery_frame_id": DELIVERY_FRAME_ID,
            "impact_reference_point": IMPACT_REFERENCE_POINT,
            "convention_id": CONVENTION_ID,
            "impact_model_id": IMPACT_MODEL_ID,
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise ValueError(f"{name} must be {required}")
        object.__setattr__(self, "club_profile_id", ClubProfileId(self.club_profile_id))
        object.__setattr__(
            self, "flight_model_id", _text(self.flight_model_id, "flight_model_id")
        )
        for variable in self.inverse_request.variables:
            expected_unit = _DELIVERY_UNITS.get(variable.parameter_id)
            if expected_unit is None:
                raise ValueError(
                    f"unsupported delivery variable: {variable.parameter_id}"
                )
            if variable.unit != expected_unit:
                parameter_id = variable.parameter_id
                raise ValueError(
                    f"{parameter_id} canonical unit is {expected_unit}, "
                    f"not {variable.unit}"
                )
            supported_lower, supported_upper = _DELIVERY_LIMITS[variable.parameter_id]
            if (
                variable.lower_bound < supported_lower
                or variable.upper_bound > supported_upper
            ):
                raise ValueError(
                    f"{variable.parameter_id} bounds exceed supported range "
                    f"[{supported_lower}, {supported_upper}]"
                )
        if (
            isinstance(self.family_count, bool)
            or not isinstance(self.family_count, int)
            or self.family_count < 1
        ):
            raise ValueError("family_count must be a positive integer")
        if self.family_count > self.inverse_request.candidate_count:
            raise ValueError("family_count must not exceed candidate_count")
        radius = _finite(self.family_radius, "family_radius")
        sensitivity = _finite(self.sensitivity_fraction, "sensitivity_fraction")
        event_time = _finite(self.impact_event_time_s, "impact_event_time_s")
        if not 0.0 < radius <= 1.0:
            raise ValueError("family_radius must be in (0, 1]")
        if not 0.0 < sensitivity <= 0.25:
            raise ValueError("sensitivity_fraction must be in (0, 0.25]")
        if event_time < 0.0:
            raise ValueError("impact_event_time_s must be nonnegative")
        object.__setattr__(self, "family_radius", radius)
        object.__setattr__(self, "sensitivity_fraction", sensitivity)
        object.__setattr__(self, "impact_event_time_s", event_time)

    def to_dict(self) -> dict[str, object]:
        """Return the strict v1 wire representation."""
        return {
            "club_profile_id": self.club_profile_id.value,
            "convention_id": self.convention_id,
            "delivery_frame_id": self.delivery_frame_id,
            "family_count": self.family_count,
            "family_radius": _wire_number(self.family_radius),
            "flight_model_id": self.flight_model_id,
            "impact_event_time_s": _wire_number(self.impact_event_time_s),
            "impact_model_id": self.impact_model_id,
            "impact_reference_point": self.impact_reference_point,
            "inverse_request": self.inverse_request.to_dict(),
            "schema_version": self.schema_version,
            "sensitivity_fraction": _wire_number(self.sensitivity_fraction),
            "target_frame_id": self.target_frame_id,
        }

    def to_json(self) -> str:
        """Serialize deterministically with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ImpactSolutionRequest:
        """Parse and validate an exact v1 request."""
        fields = {
            "club_profile_id",
            "convention_id",
            "delivery_frame_id",
            "family_count",
            "family_radius",
            "flight_model_id",
            "impact_event_time_s",
            "impact_model_id",
            "impact_reference_point",
            "inverse_request",
            "schema_version",
            "sensitivity_fraction",
            "target_frame_id",
        }
        _exact(payload, fields, "impact solution request")
        if not isinstance(payload["inverse_request"], dict):
            raise ValueError("inverse_request must be an object")
        return cls(
            InverseFlightRequest.from_dict(payload["inverse_request"]),
            ClubProfileId(payload["club_profile_id"]),
            str(payload["flight_model_id"]),
            payload["family_count"],
            float(payload["family_radius"]),
            float(payload["sensitivity_fraction"]),
            float(payload["impact_event_time_s"]),
            str(payload["target_frame_id"]),
            str(payload["delivery_frame_id"]),
            str(payload["impact_reference_point"]),
            str(payload["convention_id"]),
            str(payload["impact_model_id"]),
            str(payload["schema_version"]),
        )


@dataclass(frozen=True)
class ForwardEvaluation:
    """Full cached delivery-to-flight evaluation used by family solving."""

    status: ForwardStatus
    launch_metrics: tuple[EvaluatedMetric, ...]
    flight_metrics: tuple[EvaluatedMetric, ...]
    model_manifest: ModelManifest
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ForwardStatus(self.status))
        launch = tuple(self.launch_metrics)
        flight = tuple(self.flight_metrics)
        ids = [item.metric_id for item in launch + flight]
        if len(ids) != len(set(ids)):
            raise ValueError("forward metric IDs must be unique")
        if self.status is ForwardStatus.COMPLETE:
            if not launch or not flight or self.reason is not None:
                raise ValueError(
                    "complete forward evaluations require metrics and no reason"
                )
        elif launch or flight or self.reason is None or not self.reason.strip():
            raise ValueError("incomplete forward evaluations require only a reason")
        object.__setattr__(self, "launch_metrics", launch)
        object.__setattr__(self, "flight_metrics", flight)

    def launch_metric(self, metric_id: FlightMetricId) -> EvaluatedMetric:
        """Return one launch metric by stable identifier."""
        return next(item for item in self.launch_metrics if item.metric_id is metric_id)

    def flight_metric(self, metric_id: FlightMetricId) -> EvaluatedMetric:
        """Return one flight metric by stable identifier."""
        return next(item for item in self.flight_metrics if item.metric_id is metric_id)

    def to_solver_evaluation(self) -> SolverEvaluation:
        """Adapt this rich evaluation to the existing inverse-solver seam."""
        from .inverse_contract import EvaluationStatus

        if self.status is ForwardStatus.COMPLETE:
            return SolverEvaluation(
                EvaluationStatus.COMPLETE, self.launch_metrics + self.flight_metrics
            )
        status = (
            EvaluationStatus.NO_IMPACT
            if self.status is ForwardStatus.NO_IMPACT
            else EvaluationStatus.FAILED
        )
        return SolverEvaluation(status, (), self.reason)


from .impact_solution_result import ImpactSolutionResult  # noqa: E402

__all__ = [
    "ClubProfileId",
    "ForwardEvaluation",
    "ForwardStatus",
    "ImpactSolutionRequest",
    "ImpactSolutionResult",
    "ModelAvailability",
    "ModelManifest",
]
