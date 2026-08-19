"""Player capability evidence contracts and covariance validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

PROFILE_SCHEMA_VERSION = "player-capability-profile/v1"
_MATRIX_TOLERANCE = 1e-10


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


@dataclass(frozen=True)
class CapabilityParameter:
    """One delivery parameter's safety, evidence, center, bias, and noise."""

    parameter_id: str
    unit: str
    lower_bound: float
    upper_bound: float
    evidence_lower_bound: float
    evidence_upper_bound: float
    baseline: float
    bias: float
    standard_deviation: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameter_id", _text(self.parameter_id, "parameter_id")
        )
        object.__setattr__(self, "unit", _text(self.unit, "parameter unit"))
        for name in (
            "lower_bound",
            "upper_bound",
            "evidence_lower_bound",
            "evidence_upper_bound",
            "baseline",
            "bias",
            "standard_deviation",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.lower_bound > self.upper_bound:
            raise ValueError("safe parameter bounds must be ordered")
        if (
            not self.lower_bound
            <= self.evidence_lower_bound
            <= self.evidence_upper_bound
            <= self.upper_bound
        ):
            raise ValueError("evidence bounds must lie within safe parameter bounds")
        if not self.lower_bound <= self.baseline <= self.upper_bound:
            raise ValueError("baseline must lie within safe parameter bounds")
        if self.standard_deviation < 0:
            raise ValueError("standard_deviation must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        """Return the strict v1 wire representation."""
        return {
            "baseline": self.baseline,
            "bias": self.bias,
            "evidence_lower_bound": self.evidence_lower_bound,
            "evidence_upper_bound": self.evidence_upper_bound,
            "lower_bound": self.lower_bound,
            "parameter_id": self.parameter_id,
            "standard_deviation": self.standard_deviation,
            "unit": self.unit,
            "upper_bound": self.upper_bound,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CapabilityParameter:
        """Parse a strict v1 parameter."""
        _exact(
            payload,
            {
                "baseline",
                "bias",
                "evidence_lower_bound",
                "evidence_upper_bound",
                "lower_bound",
                "parameter_id",
                "standard_deviation",
                "unit",
                "upper_bound",
            },
            "capability parameter",
        )
        return cls(
            str(payload["parameter_id"]),
            str(payload["unit"]),
            float(payload["lower_bound"]),
            float(payload["upper_bound"]),
            float(payload["evidence_lower_bound"]),
            float(payload["evidence_upper_bound"]),
            float(payload["baseline"]),
            float(payload["bias"]),
            float(payload["standard_deviation"]),
        )


@dataclass(frozen=True)
class ClubCapability:
    """Per-club delivery envelope with a covariance or correlation model."""

    club_id: str
    parameters: tuple[CapabilityParameter, ...]
    matrix_kind: str
    matrix: tuple[tuple[float, ...], ...]
    provenance: str
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "club_id", _text(self.club_id, "club_id"))
        object.__setattr__(
            self, "provenance", _text(self.provenance, "club provenance")
        )
        parameters = tuple(self.parameters)
        if not parameters or len({item.parameter_id for item in parameters}) != len(
            parameters
        ):
            raise ValueError("club parameter IDs must be nonempty and unique")
        if self.matrix_kind not in {"correlation", "covariance"}:
            raise ValueError("matrix_kind must be correlation or covariance")
        matrix = np.asarray(self.matrix, dtype=float)
        size = len(parameters)
        if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
            raise ValueError("capability matrix shape and values must match parameters")
        if not np.allclose(matrix, matrix.T, atol=_MATRIX_TOLERANCE, rtol=0):
            raise ValueError("capability matrix must be symmetric")
        if float(np.min(np.linalg.eigvalsh(matrix))) < -_MATRIX_TOLERANCE:
            raise ValueError("capability matrix must be positive semidefinite")
        diagonal = np.diag(matrix)
        if self.matrix_kind == "correlation" and not np.allclose(
            diagonal, 1, atol=_MATRIX_TOLERANCE, rtol=0
        ):
            raise ValueError("correlation matrix must have a unit diagonal")
        if self.matrix_kind == "covariance" and np.any(diagonal < 0):
            raise ValueError("covariance diagonal must be nonnegative")
        confidence = _finite(self.confidence, "club confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("club confidence must be within [0, 1]")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(
            self,
            "matrix",
            tuple(tuple(float(value) for value in row) for row in matrix),
        )
        object.__setattr__(self, "confidence", confidence)

    def covariance_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Return dimensional covariance in declared parameter order."""
        matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix_kind == "correlation":
            scales = np.asarray([item.standard_deviation for item in self.parameters])
            matrix = scales[:, None] * matrix * scales[None, :]
        return tuple(tuple(float(value) for value in row) for row in matrix)

    def to_dict(self) -> dict[str, object]:
        """Return the strict v1 wire representation."""
        return {
            "club_id": self.club_id,
            "confidence": self.confidence,
            "matrix": [list(row) for row in self.matrix],
            "matrix_kind": self.matrix_kind,
            "parameters": [item.to_dict() for item in self.parameters],
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClubCapability:
        """Parse a strict v1 club capability."""
        _exact(
            payload,
            {
                "club_id",
                "confidence",
                "matrix",
                "matrix_kind",
                "parameters",
                "provenance",
            },
            "club capability",
        )
        return cls(
            str(payload["club_id"]),
            tuple(
                CapabilityParameter.from_dict(item) for item in payload["parameters"]
            ),
            str(payload["matrix_kind"]),
            tuple(tuple(float(value) for value in row) for row in payload["matrix"]),
            str(payload["provenance"]),
            float(payload["confidence"]),
        )


@dataclass(frozen=True)
class PlayerCapabilityProfile:
    """Immutable multi-club evidence profile."""

    profile_id: str
    clubs: tuple[ClubCapability, ...]
    provenance: str
    confidence: float
    schema_version: str = PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _text(self.profile_id, "profile_id"))
        object.__setattr__(
            self, "provenance", _text(self.provenance, "profile provenance")
        )
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        clubs = tuple(self.clubs)
        if not clubs or len({item.club_id for item in clubs}) != len(clubs):
            raise ValueError("profile club IDs must be nonempty and unique")
        confidence = _finite(self.confidence, "profile confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("profile confidence must be within [0, 1]")
        object.__setattr__(self, "clubs", clubs)
        object.__setattr__(self, "confidence", confidence)

    def club(self, club_id: str) -> ClubCapability:
        """Return one club or fail closed for an unknown identifier."""
        try:
            return next(item for item in self.clubs if item.club_id == club_id)
        except StopIteration as exc:
            raise ValueError(f"unknown club_id: {club_id}") from exc

    def to_dict(self) -> dict[str, object]:
        """Return the strict v1 wire representation."""
        return {
            "clubs": [item.to_dict() for item in self.clubs],
            "confidence": self.confidence,
            "profile_id": self.profile_id,
            "provenance": self.provenance,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlayerCapabilityProfile:
        """Parse a strict v1 profile."""
        _exact(
            payload,
            {"clubs", "confidence", "profile_id", "provenance", "schema_version"},
            "player capability profile",
        )
        return cls(
            str(payload["profile_id"]),
            tuple(ClubCapability.from_dict(item) for item in payload["clubs"]),
            str(payload["provenance"]),
            float(payload["confidence"]),
            str(payload["schema_version"]),
        )


__all__ = [
    "CapabilityParameter",
    "ClubCapability",
    "PlayerCapabilityProfile",
    "PROFILE_SCHEMA_VERSION",
]
