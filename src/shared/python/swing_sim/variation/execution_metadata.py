"""Strict resolved-base and registry identity for variation execution."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from shared.python.contracts import require

from .registry import keys_for_mode, variable_registry
from .spec import MAX_SAFE_INTEGER, SCHEMA_VERSION, VariationPlan

EXECUTION_DOCUMENT_SCHEMA_ID = "rate-of-closure/variation-execution-document"
EXECUTION_DOCUMENT_SCHEMA_VERSION = 1
EXECUTION_METADATA_SCHEMA_ID = "rate-of-closure/variation-execution-metadata"
EXECUTION_METADATA_SCHEMA_VERSION = 1
VARIABLE_REGISTRY_SCHEMA_ID = "swing-sim/variation-variable-registry"
VARIABLE_REGISTRY_SCHEMA_VERSION = 1
LEGACY_CURRENT_REGISTRY_WARNING = (
    "Legacy plan has no historical execution sidecar; resolved against the "
    "current variable registry. This is not evidence of historical reproducibility."
)

_DOCUMENT_FIELDS = frozenset({"schema_id", "schema_version", "plan", "metadata"})
_METADATA_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "plan_sha256",
        "mode",
        "flight_model",
        "registry_schema_id",
        "registry_schema_version",
        "registry_sha256",
        "resolved_variables",
    }
)
_VARIABLE_FIELDS = frozenset({"variable_key", "value", "unit", "dimension"})
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "base_variables",
        "noise",
        "n_runs",
        "seed",
        "flight_model",
        "groups",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ResolvedVariableSnapshot:
    """One resolved execution value with registered unit semantics."""

    variable_key: str
    value: float
    unit: str
    dimension: str

    def to_json_dict(self) -> dict[str, object]:
        """Return the canonical wire record."""
        return {
            "variable_key": self.variable_key,
            "value": self.value,
            "unit": self.unit,
            "dimension": self.dimension,
        }


@dataclass(frozen=True)
class VariationExecutionMetadata:
    """Immutable sidecar binding a plan to current resolved registry state."""

    plan_sha256: str
    mode: str
    flight_model: str
    registry_sha256: str
    resolved_variables: tuple[ResolvedVariableSnapshot, ...]
    schema_id: str = EXECUTION_METADATA_SCHEMA_ID
    schema_version: int = EXECUTION_METADATA_SCHEMA_VERSION
    registry_schema_id: str = VARIABLE_REGISTRY_SCHEMA_ID
    registry_schema_version: int = VARIABLE_REGISTRY_SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, object]:
        """Return the canonical snake-case wire record."""
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "mode": self.mode,
            "flight_model": self.flight_model,
            "registry_schema_id": self.registry_schema_id,
            "registry_schema_version": self.registry_schema_version,
            "registry_sha256": self.registry_sha256,
            "resolved_variables": [
                item.to_json_dict() for item in self.resolved_variables
            ],
        }


@dataclass(frozen=True)
class ExecutionMetadataResolution:
    """Validated metadata plus any legacy-resolution warning."""

    metadata: VariationExecutionMetadata
    warning: str | None


@dataclass(frozen=True)
class VariationExecutionDocument:
    """Strictly decoded plan and exact execution sidecar."""

    plan: VariationPlan
    metadata: VariationExecutionMetadata
    warning: str | None = None


def _f64_hex(value: float) -> str:
    return struct.pack(">d", value).hex()


def _normalized_float(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _digest_value(value: object) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        require(abs(value) <= MAX_SAFE_INTEGER, "digest integer must be safe", value)
        numeric = float(value)
        return {"$f64": _f64_hex(numeric)}
    if isinstance(value, float):
        numeric = _normalized_float(value)
        require(math.isfinite(numeric), "digest numbers must be finite", value)
        return {"$f64": _f64_hex(numeric)}
    if isinstance(value, Mapping):
        require(
            all(isinstance(key, str) for key in value),
            "digest mapping keys must be strings",
        )
        return {
            key: _digest_value(value[key])
            for key in sorted(cast(Mapping[str, object], value))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_digest_value(item) for item in value]
    raise TypeError(f"unsupported canonical digest value: {type(value).__name__}")


def _sha256(value: object) -> str:
    canonical = json.dumps(
        _digest_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _plan_sha256(plan: VariationPlan) -> str:
    return _sha256(plan.to_json_dict())


def _resolved_variables(plan: VariationPlan) -> tuple[ResolvedVariableSnapshot, ...]:
    registry = variable_registry()
    resolved = plan.resolved_base()
    return tuple(
        ResolvedVariableSnapshot(
            variable_key=key,
            value=_normalized_float(float(resolved[key])),
            unit=registry[key].unit,
            dimension=registry[key].dimension,
        )
        for key in sorted(keys_for_mode(plan.mode))
    )


def _registry_sha256(plan: VariationPlan) -> str:
    registry = variable_registry()
    records = [
        {
            "variable_key": key,
            "default": registry[key].default,
            "unit": registry[key].unit,
            "dimension": registry[key].dimension,
        }
        for key in sorted(keys_for_mode(plan.mode))
    ]
    return _sha256(
        {
            "schema_id": VARIABLE_REGISTRY_SCHEMA_ID,
            "schema_version": VARIABLE_REGISTRY_SCHEMA_VERSION,
            "variables": records,
        }
    )


def make_execution_metadata(plan: VariationPlan) -> VariationExecutionMetadata:
    """Snapshot the plan's complete resolved base and registry semantics."""
    require(isinstance(plan, VariationPlan), "plan must be a VariationPlan", plan)
    return VariationExecutionMetadata(
        plan_sha256=_plan_sha256(plan),
        mode=plan.mode,
        flight_model=plan.flight_model,
        registry_sha256=_registry_sha256(plan),
        resolved_variables=_resolved_variables(plan),
    )


def validate_execution_metadata(
    plan: VariationPlan, metadata: VariationExecutionMetadata
) -> VariationExecutionMetadata:
    """Reject any plan, resolved-value, unit, dimension, or registry drift."""
    expected = make_execution_metadata(plan)
    require(metadata.schema_id == expected.schema_id, "metadata schema_id mismatch")
    require(
        metadata.schema_version == expected.schema_version,
        "metadata schema_version mismatch",
    )
    require(metadata.mode == expected.mode, "metadata mode mismatch")
    require(
        metadata.flight_model == expected.flight_model, "metadata flight_model mismatch"
    )
    require(metadata.plan_sha256 == expected.plan_sha256, "plan digest mismatch")
    require(
        metadata.registry_schema_id == expected.registry_schema_id,
        "registry schema_id mismatch",
    )
    require(
        metadata.registry_schema_version == expected.registry_schema_version,
        "registry schema_version mismatch",
    )
    require(
        metadata.resolved_variables == expected.resolved_variables,
        "resolved variable snapshot mismatch",
    )
    require(
        metadata.registry_sha256 == expected.registry_sha256, "registry digest mismatch"
    )
    return metadata


def resolve_execution_metadata(
    plan: VariationPlan, metadata: VariationExecutionMetadata | None
) -> ExecutionMetadataResolution:
    """Validate supplied metadata or explicitly resolve a legacy raw plan."""
    if metadata is None:
        return ExecutionMetadataResolution(
            make_execution_metadata(plan), LEGACY_CURRENT_REGISTRY_WARNING
        )
    return ExecutionMetadataResolution(
        validate_execution_metadata(plan, metadata), None
    )


def execution_document_to_json_dict(
    plan: VariationPlan, metadata: VariationExecutionMetadata | None = None
) -> dict[str, object]:
    """Write canonical plan v2 plus a fresh or validated strict sidecar."""
    resolved = resolve_execution_metadata(plan, metadata)
    return {
        "schema_id": EXECUTION_DOCUMENT_SCHEMA_ID,
        "schema_version": EXECUTION_DOCUMENT_SCHEMA_VERSION,
        "plan": plan.to_json_dict(),
        "metadata": resolved.metadata.to_json_dict(),
    }


def _mapping(value: object, name: str, fields: frozenset[str]) -> Mapping[str, object]:
    require(isinstance(value, Mapping), f"{name} must be an object", value)
    item = cast(Mapping[str, object], value)
    require(set(item) == fields, f"{name} fields mismatch", tuple(item))
    return item


def _text(value: object, name: str) -> str:
    require(isinstance(value, str), f"{name} must be text", value)
    return cast(str, value)


def _integer(value: object, name: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
        value,
    )
    return cast(int, value)


def _metadata_from_json_dict(value: object) -> VariationExecutionMetadata:
    item = _mapping(value, "metadata", _METADATA_FIELDS)
    raw_variables = item["resolved_variables"]
    require(isinstance(raw_variables, list), "resolved_variables must be an array")
    snapshots: list[ResolvedVariableSnapshot] = []
    for index, raw in enumerate(cast(list[object], raw_variables)):
        record = _mapping(raw, f"resolved_variables[{index}]", _VARIABLE_FIELDS)
        numeric = record["value"]
        require(
            isinstance(numeric, (int, float))
            and not isinstance(numeric, bool)
            and math.isfinite(float(numeric)),
            "resolved variable value must be finite",
            numeric,
        )
        snapshots.append(
            ResolvedVariableSnapshot(
                variable_key=_text(record["variable_key"], "variable_key"),
                value=float(cast(int | float, numeric)),
                unit=_text(record["unit"], "unit"),
                dimension=_text(record["dimension"], "dimension"),
            )
        )
    plan_digest = _text(item["plan_sha256"], "plan_sha256")
    registry_digest = _text(item["registry_sha256"], "registry_sha256")
    require(
        bool(_SHA256.fullmatch(plan_digest)), "plan_sha256 must be lowercase SHA-256"
    )
    require(
        bool(_SHA256.fullmatch(registry_digest)),
        "registry_sha256 must be lowercase SHA-256",
    )
    return VariationExecutionMetadata(
        schema_id=_text(item["schema_id"], "metadata schema_id"),
        schema_version=_integer(item["schema_version"], "metadata schema_version"),
        plan_sha256=plan_digest,
        mode=_text(item["mode"], "metadata mode"),
        flight_model=_text(item["flight_model"], "metadata flight_model"),
        registry_schema_id=_text(item["registry_schema_id"], "registry schema_id"),
        registry_schema_version=_integer(
            item["registry_schema_version"], "registry schema_version"
        ),
        registry_sha256=registry_digest,
        resolved_variables=tuple(snapshots),
    )


def execution_document_from_json_dict(value: object) -> VariationExecutionDocument:
    """Strictly parse and validate one execution document against this registry."""
    item = _mapping(value, "execution document", _DOCUMENT_FIELDS)
    require(
        item["schema_id"] == EXECUTION_DOCUMENT_SCHEMA_ID, "document schema_id mismatch"
    )
    require(
        _integer(item["schema_version"], "document schema_version")
        == EXECUTION_DOCUMENT_SCHEMA_VERSION,
        "document schema_version mismatch",
    )
    plan_item = _mapping(item["plan"], "plan", _PLAN_FIELDS)
    require(
        plan_item["schema_version"] == SCHEMA_VERSION,
        "execution document requires canonical plan v2",
    )
    plan = VariationPlan.from_json_dict(plan_item)
    require(
        plan.to_json_dict() == dict(plan_item),
        "execution document plan is not canonical v2",
    )
    metadata = _metadata_from_json_dict(item["metadata"])
    return VariationExecutionDocument(plan, validate_execution_metadata(plan, metadata))


__all__ = [
    "EXECUTION_DOCUMENT_SCHEMA_ID",
    "EXECUTION_DOCUMENT_SCHEMA_VERSION",
    "EXECUTION_METADATA_SCHEMA_ID",
    "EXECUTION_METADATA_SCHEMA_VERSION",
    "LEGACY_CURRENT_REGISTRY_WARNING",
    "VARIABLE_REGISTRY_SCHEMA_ID",
    "VARIABLE_REGISTRY_SCHEMA_VERSION",
    "ExecutionMetadataResolution",
    "ResolvedVariableSnapshot",
    "VariationExecutionDocument",
    "VariationExecutionMetadata",
    "execution_document_from_json_dict",
    "execution_document_to_json_dict",
    "make_execution_metadata",
    "resolve_execution_metadata",
    "validate_execution_metadata",
]
