"""Primitive validators shared by ground material profile value records."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_float

_EnumT = TypeVar("_EnumT")
_RecordT = TypeVar("_RecordT")
_MIN_CANONICAL_POSITIVE = 0.00000000001
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
_TEXT_EDGE_WHITESPACE = " \t\r\n\f\v"
_PARAMETER_UNITS = {
    "normal_restitution": "1",
    "static_friction": "1",
    "kinetic_friction": "1",
    "rolling_resistance": "1",
    "firmness_pa": "Pa",
    "hardness_fraction": "1",
    "grass_height_m": "m",
    "compressibility_fraction": "1",
    "compression_damping_fraction": "1",
    "turf_density_kg_m3": "kg/m^3",
    "moisture_fraction": "1",
}


class ProfileDocument:
    """Delegate strict profile wire behavior without coupling value records."""

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible v1 mapping."""
        from .profile_wire import document_to_dict

        return document_to_dict(self)

    def to_json(self) -> str:
        """Return deterministic compact canonical JSON."""
        from .profile_wire import document_to_json

        return document_to_json(self)

    def canonical_sha256(self) -> str:
        """Return the SHA-256 identity of canonical UTF-8 JSON."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Any:
        """Parse one exact-field v1 mapping for this document type."""
        from .profile_wire import document_from_dict

        return document_from_dict(cls, payload)


def _raw_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    if isinstance(value, int) and abs(value) > _MAX_SAFE_INTEGER:
        raise ValueError(f"{name} exceeds the cross-runtime safe range")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if abs(number) > _MAX_SAFE_INTEGER:
        raise ValueError(f"{name} exceeds the cross-runtime safe range")
    return number


def finite_number(value: object, name: str) -> float:
    """Validate raw identity safety, then return one canonical finite float."""
    return float(canonical_numeric_float(_raw_finite_number(value, name)))


def bounded_number(value: object, name: str, bounds: tuple[float, float]) -> float:
    """Return a canonical number inside inclusive ``bounds``."""
    number = _raw_finite_number(value, name)
    lower, upper = bounds
    if not lower <= number <= upper:
        raise ValueError(f"{name} must lie within [{lower:g}, {upper:g}]")
    return float(canonical_numeric_float(number))


def nonnegative_number(value: object, name: str) -> float:
    """Return a canonical nonnegative number."""
    number = _raw_finite_number(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return float(canonical_numeric_float(number))


def positive_number(value: object, name: str) -> float:
    """Return a canonical representable positive number."""
    number = _raw_finite_number(value, name)
    if number < _MIN_CANONICAL_POSITIVE:
        raise ValueError(f"{name} must be at least {_MIN_CANONICAL_POSITIVE:g}")
    return float(canonical_numeric_float(number))


def parameter_unit(parameter_id: str) -> str:
    """Return the immutable SI unit for one canonical parameter."""
    return _PARAMETER_UNITS[parameter_id]


def parameter_value(parameter_id: str, value: object) -> float:
    """Validate one material parameter against its physical v1 bounds."""
    fractions = {
        "normal_restitution",
        "rolling_resistance",
        "hardness_fraction",
        "compressibility_fraction",
        "compression_damping_fraction",
        "moisture_fraction",
    }
    if parameter_id in fractions:
        return bounded_number(value, "value_si", (0.0, 1.0))
    if parameter_id in {"static_friction", "kinetic_friction"}:
        return bounded_number(value, "value_si", (0.0, 5.0))
    if parameter_id == "firmness_pa":
        return positive_number(value, "value_si")
    if parameter_id in {"grass_height_m", "turf_density_kg_m3"}:
        return nonnegative_number(value, "value_si")
    raise ValueError(f"unknown ground parameter_id: {parameter_id}")


def parameter_validity(
    parameter_id: str, value: float, lower: object, upper: object
) -> tuple[float, float]:
    """Return physical validity limits that enclose the declared value."""
    normalized_lower = parameter_value(parameter_id, lower)
    normalized_upper = parameter_value(parameter_id, upper)
    if not normalized_lower <= value <= normalized_upper:
        raise ValueError("validity bounds must enclose value_si")
    return normalized_lower, normalized_upper


def _validity_sources_traceable(profile: Any) -> bool:
    coverage = {
        evidence.evidence_id: set(evidence.parameter_ids)
        for evidence in profile.evidence
    }
    return all(
        all(
            parameter.parameter_id in coverage.get(evidence_id, set())
            for evidence_id in (
                *parameter.validity_lower_evidence_ids,
                *parameter.validity_upper_evidence_ids,
            )
        )
        for parameter in profile.parameters
    )


def validity_source_ids(parameters: Iterable[Any]) -> set[str]:
    """Return all lower/upper validity evidence identities."""
    return {
        evidence_id
        for parameter in parameters
        for evidence_id in (
            *parameter.validity_lower_evidence_ids,
            *parameter.validity_upper_evidence_ids,
        )
    }


def qualification_decisions(
    profile: Any, canonical_ids: tuple[Any, ...]
) -> tuple[bool, ...]:
    """Derive the seven stable qualification gates from validated records."""
    covered = {item for evidence in profile.evidence for item in evidence.parameter_ids}
    evidence_ids = {item.evidence_id for item in profile.evidence}
    rights_ok = (
        profile.rights.redistribution_allowed and profile.rights.derivative_use_allowed
    )
    referenced = tuple(
        item
        for item in profile.evidence
        if item.evidence_id in profile.calibration.evidence_ids
    )
    referenced_coverage = {
        item for evidence in referenced for item in evidence.parameter_ids
    }
    calibration_parameters = set(profile.calibration.parameter_ids)
    calibration_ok = (
        set(profile.calibration.evidence_ids) <= evidence_ids
        and calibration_parameters == set(canonical_ids)
        and referenced_coverage >= calibration_parameters
    )
    uncertainty_ok = all(
        item.confidence_level > 0.0 and item.coverage_factor >= 1.0
        for item in profile.parameters
    )
    return (
        covered == set(canonical_ids),
        _validity_sources_traceable(profile),
        rights_ok,
        uncertainty_ok,
        calibration_ok,
        profile.applicability.moisture_min_fraction
        <= profile.parameter_value("moisture_fraction")
        <= profile.applicability.moisture_max_fraction,
        bool(profile.provenance.source_sha256),
    )


def calibrated_model_use(profile: Any, canonical_ids: tuple[Any, ...]) -> bool:
    """Return scientific calibration status independently of reuse rights."""
    decisions = qualification_decisions(profile, canonical_ids)
    scientifically_ready = all((*decisions[:2], *decisions[3:]))
    referenced = tuple(
        item
        for item in profile.evidence
        if item.evidence_id in profile.calibration.evidence_ids
    )
    measured = any(str(item.kind) == "measured_dataset" for item in referenced)
    return scientifically_ready and measured


def strict_text(value: object, name: str) -> str:
    """Return nonempty scalar text without edge whitespace or surrogates."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    if value != value.strip(_TEXT_EDGE_WHITESPACE):
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError(f"{name} must not contain control characters")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} must not contain surrogate code points")
    return value


def strict_boolean(value: object, name: str) -> bool:
    """Return a real JSON boolean rather than a truthy scalar."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def sha256_digest(value: object, name: str) -> str:
    """Return one lowercase SHA-256 hex digest or fail closed."""
    digest = strict_text(value, name)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return digest


def sorted_unique_texts(values: Iterable[object], name: str) -> tuple[str, ...]:
    """Return a nonempty, strictly sorted tuple of unique text values."""
    normalized = tuple(strict_text(value, name) for value in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError(f"{name} must be sorted and unique")
    return normalized


def canonical_enum_subset(
    values: Iterable[object],
    enum_type: Callable[[Any], _EnumT],
    canonical: tuple[_EnumT, ...],
    name: str,
) -> tuple[_EnumT, ...]:
    """Return a nonempty enum subset in the declared canonical order."""
    normalized = tuple(enum_type(value) for value in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    expected = tuple(item for item in canonical if item in normalized)
    if len(set(normalized)) != len(normalized) or normalized != expected:
        raise ValueError(
            f"{name} must use canonical parameter order without duplicates"
        )
    return normalized


def exact_fields(payload: dict[str, Any], fields: set[str], name: str) -> None:
    """Reject omitted and unknown fields at a wire boundary."""
    if set(payload) != fields:
        raise ValueError(f"{name} fields do not match v1 schema")


def object_mapping(value: object, name: str) -> dict[str, Any]:
    """Return a string-keyed JSON object mapping."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return value


def array_value(value: object, name: str) -> list[Any]:
    """Return a JSON array without accepting tuples at the wire boundary."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def exact_record(value: object, expected: type[_RecordT], name: str) -> _RecordT:
    """Return one exact record type, rejecting subclasses and duck types."""
    if type(value) is not expected:
        raise TypeError(f"{name} must be an exact {expected.__name__}")
    return value


def exact_records(
    values: Iterable[object], expected: type[_RecordT], name: str
) -> tuple[_RecordT, ...]:
    """Return an exact-type tuple without allowing wire-extending subclasses."""
    return tuple(exact_record(item, expected, name) for item in values)


__all__ = [
    "array_value",
    "bounded_number",
    "calibrated_model_use",
    "canonical_enum_subset",
    "exact_fields",
    "exact_record",
    "exact_records",
    "finite_number",
    "nonnegative_number",
    "object_mapping",
    "parameter_unit",
    "parameter_validity",
    "parameter_value",
    "ProfileDocument",
    "qualification_decisions",
    "positive_number",
    "sha256_digest",
    "sorted_unique_texts",
    "strict_boolean",
    "strict_text",
    "validity_source_ids",
]
