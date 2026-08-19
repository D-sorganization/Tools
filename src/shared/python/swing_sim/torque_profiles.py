"""Canonical, UI-neutral prescribed joint-torque profile domain.

The serialized contract uses SI torque units and physical-time polynomial
coefficients ordered ``[c0, c1, ...]`` for ``c0 + c1*t + ...``. Stable model
and joint IDs keep desktop, web, and headless clients independent of labels.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from shared.python.contracts import require

from ._torque_profile_validation import (
    finite_float as _finite_float,
)
from ._torque_profile_validation import (
    finite_tuple as _finite_tuple,
)
from ._torque_profile_validation import (
    sha256_or_none as _sha256_or_none,
)
from ._torque_profile_validation import (
    source_metadata as _source_metadata,
)
from ._torque_profile_validation import (
    stable_id as _stable_id,
)
from ._torque_profile_validation import (
    strict_mapping as _strict_mapping,
)
from ._torque_profile_validation import (
    time_domain as _time_domain,
)
from ._torque_profile_validation import (
    unique_json_object as _unique_json_object,
)
from ._torque_profile_validation import (
    utc_timestamp_pair as _utc_timestamp_pair,
)

TORQUE_PROFILE_SCHEMA_VERSION = 1
TORQUE_UNIT = "N*m"
COEFFICIENT_ORDER = "ascending_c0_first"

_PROFILE_FIELDS = frozenset(
    {
        "schema_version",
        "profile_id",
        "model_id",
        "name",
        "description",
        "source",
        "source_metadata",
        "created_at_utc",
        "modified_at_utc",
        "torque_unit",
        "coefficient_order",
        "time_domain_s",
        "assignments",
    }
)
_ASSIGNMENT_FIELDS = frozenset({"joint_id", "coefficients", "fit_metadata"})
_FIT_FIELDS = frozenset(
    {
        "degree",
        "rmse_nm",
        "max_abs_error_nm",
        "r_squared",
        "condition_number",
        "original_sample_sha256",
    }
)


class TorqueProfileSource(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Provenance category for a prescribed torque profile."""

    DIRECT = "direct"
    DRAWN = "drawn"
    IMPORTED = "imported"
    OPTIMIZED = "optimized"
    FITTED_RUN = "fitted_run"


def _evaluate_validated_polynomial(
    coefficients: tuple[float, ...], time_s: float
) -> float:
    """Horner evaluation for already-validated immutable coefficients."""
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * time_s + coefficient
    require(math.isfinite(result), "evaluated torque must be finite", result)
    return result


def evaluate_ascending_polynomial(
    coefficients: Sequence[float], time_s: float
) -> float:
    """Validate and evaluate explicit ``c0``-first coefficients."""
    normalized = _finite_tuple(coefficients, "coefficients")
    time_value = _finite_float(time_s, "time_s")
    return _evaluate_validated_polynomial(normalized, time_value)


@dataclass(frozen=True)
class FitMetadata:
    """Quality and provenance information for a polynomial fit."""

    degree: int
    rmse_nm: float
    max_abs_error_nm: float
    r_squared: float
    condition_number: float
    original_sample_sha256: str | None = None

    def __post_init__(self) -> None:
        require(type(self.degree) is int and self.degree >= 0, "degree must be >= 0")
        for name in ("rmse_nm", "max_abs_error_nm", "r_squared", "condition_number"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        require(self.rmse_nm >= 0.0, "rmse_nm must be >= 0", self.rmse_nm)
        require(self.r_squared <= 1.0, "r_squared must be <= 1", self.r_squared)
        require(
            self.max_abs_error_nm >= 0.0,
            "max_abs_error_nm must be >= 0",
            self.max_abs_error_nm,
        )
        require(
            self.condition_number > 0.0,
            "condition_number must be > 0",
            self.condition_number,
        )
        object.__setattr__(
            self,
            "original_sample_sha256",
            _sha256_or_none(self.original_sample_sha256),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return the exact JSON-compatible fit metadata representation."""
        return {
            "degree": self.degree,
            "rmse_nm": self.rmse_nm,
            "max_abs_error_nm": self.max_abs_error_nm,
            "r_squared": self.r_squared,
            "condition_number": self.condition_number,
            "original_sample_sha256": self.original_sample_sha256,
        }

    @classmethod
    def from_json_dict(cls, data: object) -> FitMetadata:
        """Build metadata from an exact JSON object."""
        mapping = _strict_mapping(data, _FIT_FIELDS, "fit_metadata")
        degree = mapping["degree"]
        require(type(degree) is int, "degree must be an integer", degree)
        return cls(
            degree=degree,
            rmse_nm=_finite_float(mapping["rmse_nm"], "rmse_nm"),
            max_abs_error_nm=_finite_float(
                mapping["max_abs_error_nm"], "max_abs_error_nm"
            ),
            r_squared=_finite_float(mapping["r_squared"], "r_squared"),
            condition_number=_finite_float(
                mapping["condition_number"], "condition_number"
            ),
            original_sample_sha256=mapping["original_sample_sha256"],
        )


@dataclass(frozen=True)
class TorquePolynomial:
    """One SI joint-torque polynomial in physical-time ``c0``-first order."""

    coefficients: tuple[float, ...]
    fit_metadata: FitMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "coefficients", _finite_tuple(self.coefficients, "coefficients")
        )
        require(
            self.fit_metadata is None or isinstance(self.fit_metadata, FitMetadata),
            "fit_metadata must be FitMetadata when supplied",
            self.fit_metadata,
        )
        if self.fit_metadata is not None:
            require(
                self.fit_metadata.degree == len(self.coefficients) - 1,
                "fit degree must match polynomial coefficients",
            )

    def evaluate(self, time_s: float) -> float:
        """Evaluate torque in N*m without revalidating immutable coefficients."""
        time_value = _finite_float(time_s, "time_s")
        return _evaluate_validated_polynomial(self.coefficients, time_value)


@dataclass(frozen=True)
class JointTorqueAssignment:
    """Assign one polynomial to one stable model joint identifier."""

    joint_id: str
    polynomial: TorquePolynomial

    def __post_init__(self) -> None:
        _stable_id(self.joint_id, "joint_id")
        require(
            isinstance(self.polynomial, TorquePolynomial),
            "polynomial must be TorquePolynomial",
            self.polynomial,
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return the exact JSON-compatible assignment representation."""
        metadata = self.polynomial.fit_metadata
        return {
            "joint_id": self.joint_id,
            "coefficients": list(self.polynomial.coefficients),
            "fit_metadata": None if metadata is None else metadata.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, data: object) -> JointTorqueAssignment:
        """Build an assignment from an exact JSON object."""
        mapping = _strict_mapping(data, _ASSIGNMENT_FIELDS, "assignment")
        require(
            isinstance(mapping["coefficients"], list),
            "coefficients must be a JSON array",
        )
        raw_metadata = mapping["fit_metadata"]
        metadata = (
            None if raw_metadata is None else FitMetadata.from_json_dict(raw_metadata)
        )
        polynomial = TorquePolynomial(
            _finite_tuple(mapping["coefficients"], "coefficients"), metadata
        )
        return cls(
            joint_id=_stable_id(mapping["joint_id"], "joint_id"), polynomial=polynomial
        )


@dataclass(frozen=True)
class PrescribedTorqueProfile:
    """Versioned, portable collection of per-joint torque polynomials."""

    profile_id: str
    model_id: str
    name: str
    description: str
    source: TorqueProfileSource
    source_metadata: Mapping[str, str]
    created_at_utc: str
    modified_at_utc: str
    time_domain_s: tuple[float, float]
    assignments: tuple[JointTorqueAssignment, ...]

    def __post_init__(self) -> None:
        _stable_id(self.profile_id, "profile_id")
        _stable_id(self.model_id, "model_id")
        require(
            isinstance(self.name, str) and bool(self.name.strip()), "name is required"
        )
        require(
            isinstance(self.description, str) and bool(self.description.strip()),
            "description is required",
        )
        require(isinstance(self.source, TorqueProfileSource), "invalid profile source")
        metadata = _source_metadata(self.source_metadata)
        timestamps = _utc_timestamp_pair(self.created_at_utc, self.modified_at_utc)
        domain = _time_domain(self.time_domain_s)
        assignments = tuple(self.assignments)
        require(len(assignments) > 0, "assignments must not be empty")
        require(
            all(isinstance(item, JointTorqueAssignment) for item in assignments),
            "assignments must contain JointTorqueAssignment values",
        )
        joint_ids = tuple(item.joint_id for item in assignments)
        require(len(set(joint_ids)) == len(joint_ids), "joint IDs must be unique")
        object.__setattr__(self, "source_metadata", metadata)
        object.__setattr__(self, "created_at_utc", timestamps[0])
        object.__setattr__(self, "modified_at_utc", timestamps[1])
        object.__setattr__(self, "time_domain_s", domain)
        object.__setattr__(self, "assignments", assignments)

    def evaluate(self, time_s: float) -> dict[str, float]:
        """Evaluate all assigned joint torques inside the profile time domain."""
        time_value = _finite_float(time_s, "time_s")
        start_s, end_s = self.time_domain_s
        require(start_s <= time_value <= end_s, "time_s is outside profile domain")
        return {
            item.joint_id: _evaluate_validated_polynomial(
                item.polynomial.coefficients, time_value
            )
            for item in self.assignments
        }

    def to_json_dict(self) -> dict[str, Any]:
        """Return the exact versioned JSON-compatible profile representation."""
        return {
            "schema_version": TORQUE_PROFILE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "source": self.source.value,
            "source_metadata": dict(self.source_metadata),
            "created_at_utc": self.created_at_utc,
            "modified_at_utc": self.modified_at_utc,
            "torque_unit": TORQUE_UNIT,
            "coefficient_order": COEFFICIENT_ORDER,
            "time_domain_s": list(self.time_domain_s),
            "assignments": [item.to_json_dict() for item in self.assignments],
        }

    @classmethod
    def from_json_dict(cls, data: object) -> PrescribedTorqueProfile:
        """Build a profile with strict schema version and field validation."""
        mapping = _strict_mapping(data, _PROFILE_FIELDS, "profile")
        require(
            type(mapping["schema_version"]) is int
            and mapping["schema_version"] == TORQUE_PROFILE_SCHEMA_VERSION,
            "unsupported schema_version",
        )
        require(mapping["torque_unit"] == TORQUE_UNIT, "unsupported torque_unit")
        require(
            mapping["coefficient_order"] == COEFFICIENT_ORDER,
            "unsupported coefficient_order",
        )
        try:
            source = TorqueProfileSource(mapping["source"])
        except (TypeError, ValueError):
            require(False, "invalid profile source", mapping["source"])
            raise AssertionError("unreachable") from None
        require(
            isinstance(mapping["source_metadata"], dict),
            "source_metadata must be a JSON object",
        )
        require(
            isinstance(mapping["time_domain_s"], list),
            "time_domain_s must be a JSON array",
        )
        raw_assignments = mapping["assignments"]
        require(isinstance(raw_assignments, list), "assignments must be a JSON array")
        return cls(
            profile_id=_stable_id(mapping["profile_id"], "profile_id"),
            model_id=_stable_id(mapping["model_id"], "model_id"),
            name=mapping["name"],
            description=mapping["description"],
            source=source,
            source_metadata=_source_metadata(mapping["source_metadata"]),
            created_at_utc=mapping["created_at_utc"],
            modified_at_utc=mapping["modified_at_utc"],
            time_domain_s=_time_domain(mapping["time_domain_s"]),
            assignments=tuple(
                JointTorqueAssignment.from_json_dict(item) for item in raw_assignments
            ),
        )

    def dumps(self) -> str:
        """Serialize the profile as deterministic strict JSON."""
        return json.dumps(
            self.to_json_dict(), indent=2, sort_keys=True, allow_nan=False
        )

    @classmethod
    def loads(cls, text: str) -> PrescribedTorqueProfile:
        """Parse a profile JSON string with strict object validation."""
        require(isinstance(text, str), "profile JSON must be text", text)
        try:
            data = json.loads(text, object_pairs_hook=_unique_json_object)
        except (TypeError, json.JSONDecodeError) as exc:
            require(False, "invalid profile JSON", str(exc))
            raise AssertionError("unreachable") from exc
        return cls.from_json_dict(data)


__all__ = [
    "COEFFICIENT_ORDER",
    "TORQUE_UNIT",
    "TORQUE_PROFILE_SCHEMA_VERSION",
    "FitMetadata",
    "JointTorqueAssignment",
    "PrescribedTorqueProfile",
    "TorquePolynomial",
    "TorqueProfileSource",
    "evaluate_ascending_polynomial",
]
