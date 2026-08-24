"""Canonical and explicitly legacy variation-plan persistence."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shared.python.contracts import require

from ._execution_metadata_schema import (
    EXECUTION_DOCUMENT_SCHEMA_ID,
    EXECUTION_DOCUMENT_SCHEMA_VERSION,
    LEGACY_CURRENT_REGISTRY_WARNING,
    LEGACY_EXECUTION_DOCUMENT_MIGRATION_ERROR,
)
from .execution_metadata import (
    VariationExecutionMetadata,
    execution_document_dumps,
    execution_document_from_json_dict,
    execution_document_to_json_dict,
)
from .execution_provenance import (
    PYTHON_DEFAULT_PROVENANCE,
    PlanProducerProvenance,
)
from .spec import VariationPlan


@dataclass(frozen=True)
class PersistedPlanResolution:
    """A plan plus only the historical evidence actually present on disk."""

    plan: VariationPlan
    metadata: VariationExecutionMetadata | None
    provenance: PlanProducerProvenance | None
    warning: str | None


PLAN_BINDING_SCHEMA_ID = "rate-of-closure/variation-plan-binding"
PLAN_BINDING_SCHEMA_VERSION = 1
_BINDING_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "state",
        "document",
        "legacy_plan",
        "legacy_warning",
    }
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _json_value(text: str) -> object:
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


def _legacy_document_plan(value: Mapping[str, object]) -> VariationPlan:
    raw_plan = value.get("plan")
    require(isinstance(raw_plan, Mapping), "legacy document plan must be an object")
    return VariationPlan.from_json_dict(cast(Mapping[str, Any], raw_plan))


def persisted_plan_loads(text: str) -> PersistedPlanResolution:
    """Read canonical v3 or migrate a raw/sidecar legacy plan without invention."""
    value = _json_value(text)
    if isinstance(value, Mapping) and value.get("schema_id") == (
        EXECUTION_DOCUMENT_SCHEMA_ID
    ):
        if value.get("schema_version") == EXECUTION_DOCUMENT_SCHEMA_VERSION:
            document = execution_document_from_json_dict(value)
            return PersistedPlanResolution(
                document.plan,
                document.metadata,
                document.provenance,
                document.warning,
            )
        return PersistedPlanResolution(
            _legacy_document_plan(cast(Mapping[str, object], value)),
            None,
            None,
            LEGACY_EXECUTION_DOCUMENT_MIGRATION_ERROR,
        )
    require(isinstance(value, Mapping), "variation plan must be an object", value)
    return PersistedPlanResolution(
        VariationPlan.from_json_dict(cast(Mapping[str, Any], value)),
        None,
        None,
        LEGACY_CURRENT_REGISTRY_WARNING,
    )


def persisted_plan_dumps(
    plan: VariationPlan,
    *,
    provenance: PlanProducerProvenance = PYTHON_DEFAULT_PROVENANCE,
) -> str:
    return execution_document_dumps(plan, provenance=provenance)


def persisted_plan_binding_to_json_dict(
    value: VariationPlan | PersistedPlanResolution,
) -> dict[str, object]:
    """Encode canonical evidence or an explicit legacy state without invention."""
    resolution = (
        persisted_plan_loads(persisted_plan_dumps(value))
        if isinstance(value, VariationPlan)
        else value
    )
    require(isinstance(resolution, PersistedPlanResolution), "invalid plan binding")
    metadata = resolution.metadata
    provenance = resolution.provenance
    warning = resolution.warning
    canonical = False
    if metadata is not None and provenance is not None and warning is None:
        canonical = True
        document = execution_document_to_json_dict(
            resolution.plan,
            metadata,
            provenance=provenance,
        )
        legacy_plan: object = None
        legacy_warning: object = None
    else:
        require(
            metadata is None and provenance is None and warning is not None,
            "partial plan evidence cannot be persisted",
        )
        document = None
        legacy_plan = resolution.plan.to_json_dict()
        legacy_warning = warning
    return {
        "schema_id": PLAN_BINDING_SCHEMA_ID,
        "schema_version": PLAN_BINDING_SCHEMA_VERSION,
        "state": "canonical" if canonical else "legacy",
        "document": document,
        "legacy_plan": legacy_plan,
        "legacy_warning": legacy_warning,
    }


def persisted_plan_binding_from_json_dict(value: object) -> PersistedPlanResolution:
    """Parse a strict binding and verify all plan/evidence cohesion."""
    require(isinstance(value, Mapping), "plan binding must be an object", value)
    item = cast(Mapping[str, object], value)
    require(set(item) == _BINDING_FIELDS, "plan binding fields mismatch", sorted(item))
    require(item["schema_id"] == PLAN_BINDING_SCHEMA_ID, "plan binding schema mismatch")
    require(
        type(item["schema_version"]) is int
        and item["schema_version"] == PLAN_BINDING_SCHEMA_VERSION,
        "plan binding version mismatch",
    )
    if item["state"] == "canonical":
        require(
            item["legacy_plan"] is None and item["legacy_warning"] is None,
            "canonical binding must not contain legacy evidence",
        )
        document = execution_document_from_json_dict(item["document"])
        return PersistedPlanResolution(
            document.plan, document.metadata, document.provenance, None
        )
    require(item["state"] == "legacy", "plan binding state mismatch")
    require(item["document"] is None, "legacy binding must not contain a document")
    resolution = persisted_plan_loads(
        json.dumps(item["legacy_plan"], sort_keys=True, separators=(",", ":"))
    )
    require(
        resolution.metadata is None
        and resolution.provenance is None
        and item["legacy_warning"] == resolution.warning,
        "legacy warning does not match the retained evidence",
    )
    return resolution


def write_persisted_plan(
    path: str | Path,
    plan: VariationPlan,
    *,
    provenance: PlanProducerProvenance = PYTHON_DEFAULT_PROVENANCE,
) -> None:
    """Atomically replace one plan after a durable same-directory write."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(persisted_plan_dumps(plan, provenance=provenance))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "PLAN_BINDING_SCHEMA_ID",
    "PLAN_BINDING_SCHEMA_VERSION",
    "PersistedPlanResolution",
    "persisted_plan_binding_from_json_dict",
    "persisted_plan_binding_to_json_dict",
    "persisted_plan_dumps",
    "persisted_plan_loads",
    "write_persisted_plan",
]
