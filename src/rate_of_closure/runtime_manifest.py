"""Strict, immutable calculation-runtime manifest contract.

The contract accepts only caller-supplied identities. It deliberately does not
inspect Git, installed packages, sibling repositories, or mutable environment
state; delivery adapters own that evidence collection.
"""

from __future__ import annotations

import json
import math
import re
from enum import Enum
from typing import Any, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)

from rate_of_closure.application._workspace_validation import unique_json_object
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

SCHEMA_VERSION = "calculation-runtime-manifest/v1"
MAX_SAFE_INTEGER = 9_007_199_254_740_991
_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_SEMVER = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_SHA = re.compile(r"^[0-9a-f]{40}$")
_PLACEHOLDER = re.compile(r"\b(?:fixme|placeholder|tbd|todo|unknown)\b", re.I)
_DOMAIN_ORDER = ("impact", "flight", "ground")
_AUTHORITY_FIELDS = (
    "model_id",
    "model_version",
    "implementation_authority",
    "backend",
    "integrator",
    "request_schema",
    "result_schema",
    "frame_id",
    "unit_system_id",
)


def _validated_text(value: str, name: str, *, stable_id: bool = False) -> str:
    if not value.strip():
        raise ValueError(f"{name} must be nonempty")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} must not contain surrogate code points")
    if _PLACEHOLDER.search(value):
        raise ValueError(f"{name} must not contain a placeholder")
    if stable_id and not _STABLE_ID.fullmatch(value):
        raise ValueError(f"{name} must be a stable lowercase identifier")
    return value


class StrictModel(BaseModel):
    """Immutable base for exact runtime-manifest records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SurfaceId(str, Enum):  # noqa: UP042 - Python 3.10 consumer support
    """Stable identities for the four audited product surfaces."""

    TOOLS_PYQT6 = "tools.pyqt6"
    TOOLS_REACT = "tools.react"
    UPSTREAMDRIFT_PYQT6 = "upstreamdrift.pyqt6"
    UPSTREAMDRIFT_REACT = "upstreamdrift.react"


class CalculationDomain(str, Enum):  # noqa: UP042
    """Required calculation domains recorded by every manifest."""

    IMPACT = "impact"
    FLIGHT = "flight"
    GROUND = "ground"


class Availability(str, Enum):  # noqa: UP042
    """Truthful calculation-availability state."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class SourceKind(str, Enum):  # noqa: UP042
    """Declared origin of the build identity."""

    INSTALLED_PACKAGE = "installed_package"
    SOURCE_CHECKOUT = "source_checkout"
    EMBEDDED_WEB_BUILD = "embedded_web_build"
    TEST_FIXTURE = "test_fixture"


RuntimeOptionValue = bool | int | float | str


class RuntimeOption(StrictModel):
    """One unit-explicit numerical or categorical runtime option."""

    option_id: str
    value: RuntimeOptionValue
    unit: str | None

    @field_validator("option_id")
    @classmethod
    def _option_id(cls, value: str) -> str:
        return _validated_text(value, "option_id", stable_id=True)

    @field_validator("value", mode="before")
    @classmethod
    def _value(cls, value: object) -> object:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            if abs(value) > MAX_SAFE_INTEGER:
                raise ValueError(
                    "option value exceeds the cross-runtime safe integer range"
                )
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("option value must be finite")
            return value
        if isinstance(value, str):
            return _validated_text(value, "option value")
        raise ValueError("option value must be a JSON scalar")

    @field_validator("unit")
    @classmethod
    def _unit(cls, value: str | None) -> str | None:
        return value if value is None else _validated_text(value, "option unit")

    @model_validator(mode="after")
    def _unit_semantics(self) -> Self:
        numeric = isinstance(self.value, (int, float)) and not isinstance(
            self.value, bool
        )
        if numeric != (self.unit is not None):
            raise ValueError(
                "numeric options require a unit; text/bool options require null"
            )
        return self


class RuntimeBuild(StrictModel):
    """Explicit package and immutable Tools revision identity."""

    package_name: str
    package_version: str
    tools_commit: str
    build_id: str

    @field_validator("package_name", "build_id")
    @classmethod
    def _stable_ids(cls, value: str, info: ValidationInfo) -> str:
        return _validated_text(value, info.field_name, stable_id=True)

    @field_validator("package_version")
    @classmethod
    def _package_version(cls, value: str) -> str:
        if not _SEMVER.fullmatch(value):
            raise ValueError("package_version must be semantic version text")
        return value

    @field_validator("tools_commit")
    @classmethod
    def _tools_commit(cls, value: str) -> str:
        if not _SHA.fullmatch(value):
            raise ValueError("tools_commit must be an exact lowercase 40-character SHA")
        return value


class CalculationAuthority(StrictModel):
    """Authority and numerical identity for one calculation domain."""

    domain: CalculationDomain
    status: Availability
    reason: str | None
    model_id: str | None
    model_version: str | None
    implementation_authority: str | None
    backend: str | None
    integrator: str | None
    request_schema: str | None
    result_schema: str | None
    frame_id: str | None
    unit_system_id: str | None
    numerical_options: tuple[RuntimeOption, ...]

    @field_validator("domain", mode="before")
    @classmethod
    def _domain(cls, value: object) -> object:
        return CalculationDomain(value) if isinstance(value, str) else value

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, value: object) -> object:
        return Availability(value) if isinstance(value, str) else value

    @field_validator("numerical_options", mode="before")
    @classmethod
    def _options(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator(*_AUTHORITY_FIELDS)
    @classmethod
    def _authority_identity(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _validated_text(value, info.field_name, stable_id=True)

    @field_validator("reason")
    @classmethod
    def _reason(cls, value: str | None) -> str | None:
        return value if value is None else _validated_text(value, "reason")

    @model_validator(mode="after")
    def _availability_invariant(self) -> Self:
        identities = tuple(getattr(self, field) for field in _AUTHORITY_FIELDS)
        option_ids = tuple(option.option_id for option in self.numerical_options)
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("numerical option IDs must be unique")
        if self.status is Availability.AVAILABLE:
            if self.reason is not None or any(value is None for value in identities):
                raise ValueError(
                    "available calculation requires all identities and null reason"
                )
        elif self.reason is None or any(value is not None for value in identities):
            raise ValueError(
                "unavailable calculation requires reason and null identities"
            )
        elif self.numerical_options:
            raise ValueError("unavailable calculation requires empty numerical_options")
        return self


class RuntimeProvenance(StrictModel):
    """Auditable, caller-supplied origin and evidence identities."""

    source_kind: SourceKind
    source_reference: str
    evidence_ids: tuple[str, ...]

    @field_validator("source_kind", mode="before")
    @classmethod
    def _source_kind(cls, value: object) -> object:
        return SourceKind(value) if isinstance(value, str) else value

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _evidence_sequence(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("source_reference")
    @classmethod
    def _source_reference(cls, value: str) -> str:
        return _validated_text(value, "source_reference", stable_id=True)

    @field_validator("evidence_ids")
    @classmethod
    def _evidence_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(
            _validated_text(value, "evidence_id", stable_id=True) for value in values
        )
        if not validated or len(validated) != len(set(validated)):
            raise ValueError("evidence_ids must be nonempty and unique")
        return validated


class CalculationRuntimeManifest(StrictModel):
    """Complete calculation authority manifest for one product run."""

    schema_version: Literal["calculation-runtime-manifest/v1"]
    surface_id: SurfaceId
    build: RuntimeBuild
    calculations: tuple[CalculationAuthority, ...]
    provenance: RuntimeProvenance

    @field_validator("surface_id", mode="before")
    @classmethod
    def _surface_id(cls, value: object) -> object:
        return SurfaceId(value) if isinstance(value, str) else value

    @field_validator("calculations", mode="before")
    @classmethod
    def _calculation_sequence(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _complete_domain_ledger(self) -> Self:
        domains = tuple(calculation.domain.value for calculation in self.calculations)
        if domains != _DOMAIN_ORDER:
            raise ValueError(
                "calculations must contain impact, flight, ground in order"
            )
        return self

    def to_wire(self) -> dict[str, Any]:
        """Return a detached JSON-compatible record."""
        return cast(dict[str, Any], self.model_dump(mode="json"))

    def to_json(self) -> str:
        """Return deterministic cross-runtime canonical JSON."""
        return stable_runtime_manifest_json(self)


def create_runtime_manifest(
    *,
    surface_id: SurfaceId,
    build: RuntimeBuild,
    calculations: tuple[CalculationAuthority, ...],
    provenance: RuntimeProvenance,
) -> CalculationRuntimeManifest:
    """Build a manifest only from explicit, already sourced identities."""
    return CalculationRuntimeManifest(
        schema_version=SCHEMA_VERSION,
        surface_id=surface_id,
        build=build,
        calculations=calculations,
        provenance=provenance,
    )


def runtime_manifest_from_json(text: str) -> CalculationRuntimeManifest:
    """Parse strict JSON while rejecting duplicate fields at every depth."""
    try:
        payload = json.loads(text, object_pairs_hook=unique_json_object)
    except json.JSONDecodeError as exc:
        raise ValueError("runtime manifest JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("runtime manifest JSON must be an object")
    return cast(
        CalculationRuntimeManifest,
        CalculationRuntimeManifest.model_validate(payload),
    )


def stable_runtime_manifest_json(manifest: CalculationRuntimeManifest) -> str:
    """Serialize a validated manifest with stable keys and numeric tokens."""
    if not isinstance(manifest, CalculationRuntimeManifest):
        raise TypeError("manifest must be a CalculationRuntimeManifest")
    return cast(str, canonical_numeric_json(manifest.to_wire()))


__all__ = [
    "Availability",
    "CalculationAuthority",
    "CalculationDomain",
    "CalculationRuntimeManifest",
    "RuntimeBuild",
    "RuntimeOption",
    "RuntimeProvenance",
    "SCHEMA_VERSION",
    "SourceKind",
    "SurfaceId",
    "create_runtime_manifest",
    "runtime_manifest_from_json",
    "stable_runtime_manifest_json",
]
