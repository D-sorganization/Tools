"""Strict execution-input wire and result pairing for repeated bounce."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, cast

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

from .bounce_types import (
    GROUND_IMPACT_MODEL_ID,
    GROUND_IMPACT_MODEL_VERSION,
    BounceModelSettings,
    RepeatedBounceResult,
)
from .contract_records import GroundSimulationRequest
from .contract_types import UNIT_SYSTEM_SI, GroundFrame, _positive
from .contract_wire import record_to_dict
from .request_identity import ground_request_fingerprint
from .strict_json import strict_json_object

REPEATED_BOUNCE_REQUEST_SCHEMA_VERSION = "ground-repeated-bounce-request/v1"
MAX_REPEATED_BOUNCE_REQUEST_WIRE_BYTES = 1_048_576

_REQUEST_FIELDS = {
    "capture_speed_m_s",
    "execution_input_sha256",
    "frame",
    "ground_request",
    "ground_request_sha256",
    "model_id",
    "model_version",
    "request_id",
    "schema_version",
    "surface_id",
    "unit_system",
}


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Return a string-keyed mapping or reject the wire value."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return value


def _digest(value: object, name: str) -> str:
    """Return one canonical lowercase SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def repeated_bounce_execution_input_sha256(
    ground_request_sha256: str,
    capture_speed_m_s: float,
) -> str:
    """Return the canonical identity of every configurable v1 bounce input."""
    ground_digest = _digest(ground_request_sha256, "ground_request_sha256")
    capture_speed = _positive(capture_speed_m_s, "capture_speed_m_s")
    payload = canonical_numeric_json(
        {
            "capture_speed_m_s": capture_speed,
            "ground_request_sha256": ground_digest,
            "model_id": GROUND_IMPACT_MODEL_ID,
            "model_version": GROUND_IMPACT_MODEL_VERSION,
            "schema_version": REPEATED_BOUNCE_REQUEST_SCHEMA_VERSION,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RepeatedBounceRequest:
    """One validated request for the fixed Python repeated-bounce authority."""

    ground_request: GroundSimulationRequest
    capture_speed_m_s: float = 0.05

    def __post_init__(self) -> None:
        """Normalize the sole configurable setting and exact nested type."""
        if type(self.ground_request) is not GroundSimulationRequest:
            raise ValueError("ground_request must be an exact GroundSimulationRequest")
        object.__setattr__(
            self,
            "capture_speed_m_s",
            _positive(self.capture_speed_m_s, "capture_speed_m_s"),
        )

    @property
    def schema_version(self) -> str:
        """Return the exact request-wire schema identity."""
        return REPEATED_BOUNCE_REQUEST_SCHEMA_VERSION

    @property
    def unit_system(self) -> str:
        """Return the fixed SI unit system."""
        unit_system: str = UNIT_SYSTEM_SI
        return unit_system

    @property
    def frame(self) -> GroundFrame:
        """Return the request's frozen target frame."""
        return self.ground_request.surface.frame

    @property
    def request_id(self) -> str:
        """Return the embedded physical request identity."""
        request_id: str = self.ground_request.request_id
        return request_id

    @property
    def surface_id(self) -> str:
        """Return the embedded surface identity."""
        surface_id: str = self.ground_request.surface.surface_id
        return surface_id

    @property
    def model_id(self) -> str:
        """Return the fixed repeated-bounce model identity."""
        model_id: str = GROUND_IMPACT_MODEL_ID
        return model_id

    @property
    def model_version(self) -> str:
        """Return the fixed repeated-bounce model version."""
        model_version: str = GROUND_IMPACT_MODEL_VERSION
        return model_version

    @property
    def ground_request_sha256(self) -> str:
        """Return the canonical embedded flight-to-ground request digest."""
        fingerprint: str = ground_request_fingerprint(self.ground_request)
        return fingerprint

    @property
    def execution_input_sha256(self) -> str:
        """Return the joint physical-request, model, and capture-setting digest."""
        return repeated_bounce_execution_input_sha256(
            self.ground_request_sha256,
            self.capture_speed_m_s,
        )

    @property
    def settings(self) -> BounceModelSettings:
        """Build the fixed-version settings record for later Python invocation."""
        return BounceModelSettings(capture_speed_m_s=self.capture_speed_m_s)

    def to_dict(self) -> dict[str, Any]:
        """Return the exact validated v1 request mapping."""
        return repeated_bounce_request_to_dict(self)

    def to_json(self) -> str:
        """Return deterministic canonical request JSON."""
        return repeated_bounce_request_to_json(self)

    @classmethod
    def from_dict(cls, payload: object) -> RepeatedBounceRequest:
        """Parse one exact request mapping without executing physics."""
        return repeated_bounce_request_from_dict(payload)


def _require_identity(data: dict[str, Any], request: RepeatedBounceRequest) -> None:
    """Require every redundant envelope identity to match its authority."""
    expected = {
        "execution_input_sha256": request.execution_input_sha256,
        "frame": request.frame.value,
        "ground_request_sha256": request.ground_request_sha256,
        "model_id": request.model_id,
        "model_version": request.model_version,
        "request_id": request.request_id,
        "surface_id": request.surface_id,
    }
    for field, value in expected.items():
        if data[field] != value:
            raise ValueError(f"{field} must match the embedded request authority")


def repeated_bounce_request_from_dict(payload: object) -> RepeatedBounceRequest:
    """Parse a strict v1 request envelope without invoking the solver."""
    data = _mapping(payload, "repeated bounce request")
    if set(data) != _REQUEST_FIELDS:
        raise ValueError("repeated bounce request fields do not match v1 schema")
    if data["schema_version"] != REPEATED_BOUNCE_REQUEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {data['schema_version']}")
    if data["unit_system"] != UNIT_SYSTEM_SI:
        raise ValueError(f"unsupported unit_system: {data['unit_system']}")
    _digest(data["ground_request_sha256"], "ground_request_sha256")
    _digest(data["execution_input_sha256"], "execution_input_sha256")
    ground = cast(
        GroundSimulationRequest,
        GroundSimulationRequest.from_dict(
            _mapping(data["ground_request"], "ground_request")
        ),
    )
    request = RepeatedBounceRequest(ground, data["capture_speed_m_s"])
    _require_identity(data, request)
    return request


def repeated_bounce_request_to_dict(
    request: RepeatedBounceRequest,
) -> dict[str, Any]:
    """Return an exact JSON-compatible request mapping."""
    if type(request) is not RepeatedBounceRequest:
        raise TypeError("repeated bounce request must be an exact request record")
    payload: dict[str, Any] = {
        "capture_speed_m_s": request.capture_speed_m_s,
        "execution_input_sha256": request.execution_input_sha256,
        "frame": request.frame.value,
        "ground_request": record_to_dict(request.ground_request),
        "ground_request_sha256": request.ground_request_sha256,
        "model_id": request.model_id,
        "model_version": request.model_version,
        "request_id": request.request_id,
        "schema_version": request.schema_version,
        "surface_id": request.surface_id,
        "unit_system": request.unit_system,
    }
    repeated_bounce_request_from_dict(payload)
    return payload


def repeated_bounce_request_to_json(request: RepeatedBounceRequest) -> str:
    """Serialize one validated request with canonical numeric JSON."""
    text = str(canonical_numeric_json(repeated_bounce_request_to_dict(request)))
    if len(text.encode("utf-8")) > MAX_REPEATED_BOUNCE_REQUEST_WIRE_BYTES:
        raise ValueError("repeated bounce request exceeds maximum wire size")
    return text


def repeated_bounce_request_from_json(text: str) -> RepeatedBounceRequest:
    """Parse bounded UTF-8 JSON with duplicate-key rejection at every depth."""
    if not isinstance(text, str):
        raise TypeError("repeated bounce request JSON must be text")
    if len(text.encode("utf-8")) > MAX_REPEATED_BOUNCE_REQUEST_WIRE_BYTES:
        raise ValueError("repeated bounce request exceeds maximum wire size")
    return repeated_bounce_request_from_dict(strict_json_object(text))


@dataclass(frozen=True)
class RepeatedBounceRequestResultPair:
    """Identity-safe pairing of one execution request and bounce result."""

    request: RepeatedBounceRequest
    result: RepeatedBounceResult

    def __post_init__(self) -> None:
        """Reject any request/result authority mismatch."""
        if type(self.request) is not RepeatedBounceRequest:
            raise ValueError("pairing requires an exact request record")
        if type(self.result) is not RepeatedBounceResult:
            raise ValueError("pairing requires an exact result record")
        if self.result.request_id != self.request.request_id:
            raise ValueError("result request identity must match the request")
        if self.result.surface_id != self.request.surface_id:
            raise ValueError("result surface identity must match the request")
        if self.result.frame is not self.request.frame:
            raise ValueError("result frame identity must match the request")
        if (self.result.model_id, self.result.model_version) != (
            self.request.model_id,
            self.request.model_version,
        ):
            raise ValueError("result model identity must match the request")
        if self.result.request_fingerprint_sha256 != self.request.ground_request_sha256:
            raise ValueError("result request fingerprint must match the request")

    @property
    def execution_input_sha256(self) -> str:
        """Return the complete request-side execution identity."""
        return self.request.execution_input_sha256


__all__ = [
    "MAX_REPEATED_BOUNCE_REQUEST_WIRE_BYTES",
    "REPEATED_BOUNCE_REQUEST_SCHEMA_VERSION",
    "RepeatedBounceRequest",
    "RepeatedBounceRequestResultPair",
    "repeated_bounce_execution_input_sha256",
    "repeated_bounce_request_from_dict",
    "repeated_bounce_request_from_json",
    "repeated_bounce_request_to_dict",
    "repeated_bounce_request_to_json",
]
