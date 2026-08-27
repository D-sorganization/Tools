"""Typed authority for whole-run and localized variation execution."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Literal, cast

TimeWindowPolicy = Literal["forbidden", "required_half_open_seconds"]
PointLocusPolicy = Literal["forbidden", "required_exact_topological"]

_SCHEMA_VERSION = "rate-locus-execution-capabilities/v1"
RATE_EXTENSION_VARIABLE_KEYS = frozenset(
    {
        "swing_sim.flight.launch.ground_normal_restitution",
        "swing_sim.flight.launch.ground_rolling_resistance",
    }
)
_MANAGED_NAMESPACE_PREFIXES = ("swing_sim.", "golf_club.")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "source_kind",
        "point_id_semantics",
        "time_window_semantics",
        "capabilities",
    }
)
_CAPABILITY_FIELDS = frozenset(
    {
        "variable_key",
        "supported",
        "adapter_id",
        "whole_run",
        "time_window_policy",
        "point_locus_policy",
        "point_ids",
        "unsupported_reason",
    }
)


class LocusContractError(ValueError):
    """Raised when execution-capability authority is incomplete or inconsistent."""


def _record(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise LocusContractError(f"{label} must be an object with string keys")
    return cast(Mapping[str, Any], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise LocusContractError(f"{label} must be a non-empty trimmed string")
    return value


def _nullable_text(value: object, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise LocusContractError(f"{label} must be Boolean")
    return value


def _point_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise LocusContractError("point_ids must be an array")
    points = tuple(_text(point, "point_id") for point in value)
    if len(points) != len(set(points)):
        raise LocusContractError("point_ids must be unique")
    return points


@dataclass(frozen=True)
class LocusExecutionCapability:
    """Execution semantics for one registered variation variable."""

    variable_key: str
    supported: bool
    adapter_id: str | None
    whole_run: bool
    time_window_policy: TimeWindowPolicy
    point_locus_policy: PointLocusPolicy
    point_ids: tuple[str, ...]
    unsupported_reason: str | None

    def to_wire(self) -> dict[str, object]:
        """Return the canonical JSON-compatible representation."""
        return {
            "variable_key": self.variable_key,
            "supported": self.supported,
            "adapter_id": self.adapter_id,
            "whole_run": self.whole_run,
            "time_window_policy": self.time_window_policy,
            "point_locus_policy": self.point_locus_policy,
            "point_ids": list(self.point_ids),
            "unsupported_reason": self.unsupported_reason,
        }


@dataclass(frozen=True)
class LocusExecutionContract:
    """Versioned exhaustive capability map across registered execution adapters."""

    schema_version: str
    mode: str
    source_kind: str
    point_id_semantics: str
    time_window_semantics: str
    capabilities: Mapping[str, LocusExecutionCapability]

    def to_wire(self) -> dict[str, object]:
        """Return the canonical JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "source_kind": self.source_kind,
            "point_id_semantics": self.point_id_semantics,
            "time_window_semantics": self.time_window_semantics,
            "capabilities": [row.to_wire() for row in self.capabilities.values()],
        }


def _parse_capability(raw: object) -> LocusExecutionCapability:
    row = _record(raw, "capability")
    if set(row) != _CAPABILITY_FIELDS:
        raise LocusContractError("capability fields do not match the v1 schema")
    time_policy = _text(row["time_window_policy"], "time_window_policy")
    if time_policy not in ("forbidden", "required_half_open_seconds"):
        raise LocusContractError("unsupported time_window_policy")
    point_policy = _text(row["point_locus_policy"], "point_locus_policy")
    if point_policy not in ("forbidden", "required_exact_topological"):
        raise LocusContractError("unsupported point_locus_policy")
    capability = LocusExecutionCapability(
        variable_key=_text(row["variable_key"], "variable_key"),
        supported=_boolean(row["supported"], "supported"),
        adapter_id=_nullable_text(row["adapter_id"], "adapter_id"),
        whole_run=_boolean(row["whole_run"], "whole_run"),
        time_window_policy=cast(TimeWindowPolicy, time_policy),
        point_locus_policy=cast(PointLocusPolicy, point_policy),
        point_ids=_point_ids(row["point_ids"]),
        unsupported_reason=_nullable_text(
            row["unsupported_reason"], "unsupported_reason"
        ),
    )
    _validate_capability(capability)
    return capability


def _validate_capability(capability: LocusExecutionCapability) -> None:
    if capability.whole_run and capability.time_window_policy != "forbidden":
        raise LocusContractError("whole-run capability cannot require a time window")
    if capability.whole_run and capability.point_locus_policy != "forbidden":
        raise LocusContractError("whole-run capability cannot require a point locus")
    if capability.point_locus_policy == "forbidden" and capability.point_ids:
        raise LocusContractError("forbidden point locus must have no point_ids")
    if capability.point_locus_policy == "required_exact_topological" and not (
        capability.time_window_policy == "required_half_open_seconds"
        and len(capability.point_ids) == 1
    ):
        raise LocusContractError("localized capability requires one exact point_id")
    if capability.supported:
        if capability.adapter_id is None or capability.unsupported_reason is not None:
            raise LocusContractError("supported capability requires only adapter_id")
    elif (
        capability.adapter_id is not None
        or capability.whole_run
        or capability.time_window_policy != "forbidden"
        or capability.point_locus_policy != "forbidden"
        or capability.point_ids
        or capability.unsupported_reason is None
    ):
        raise LocusContractError(
            "unsupported capability requires unsupported_reason and no execution locus"
        )


def parse_locus_execution_contract(
    payload: object,
    *,
    registered_keys: Iterable[str],
) -> LocusExecutionContract:
    """Parse authority and require exact registry coverage."""
    document = _record(payload, "locus execution contract")
    if set(document) != _TOP_LEVEL_FIELDS:
        raise LocusContractError("top-level fields do not match the v1 schema")
    if document["schema_version"] != _SCHEMA_VERSION:
        raise LocusContractError("unsupported schema_version")
    raw_capabilities = document["capabilities"]
    if not isinstance(raw_capabilities, list):
        raise LocusContractError("capabilities must be an array")
    capabilities: dict[str, LocusExecutionCapability] = {}
    for raw in raw_capabilities:
        capability = _parse_capability(raw)
        if capability.variable_key in capabilities:
            raise LocusContractError(
                f"duplicate variable_key: {capability.variable_key}"
            )
        capabilities[capability.variable_key] = capability
    expected = set(registered_keys)
    observed = set(capabilities)
    if observed != expected:
        raise LocusContractError(
            f"registry coverage mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return LocusExecutionContract(
        schema_version=_SCHEMA_VERSION,
        mode=_text(document["mode"], "mode"),
        source_kind=_text(document["source_kind"], "source_kind"),
        point_id_semantics=_text(document["point_id_semantics"], "point_id_semantics"),
        time_window_semantics=_text(
            document["time_window_semantics"], "time_window_semantics"
        ),
        capabilities=MappingProxyType(capabilities),
    )


@lru_cache(maxsize=1)
def load_locus_execution_contract() -> LocusExecutionContract:
    """Load packaged authority and bind it to the live shared registry."""
    from shared.python.swing_sim.variation import variable_registry

    resource = files("rate_of_closure").joinpath("locus_execution_capabilities.v1.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return parse_locus_execution_contract(
        payload,
        registered_keys=managed_registry_keys(variable_registry()),
    )


def managed_registry_keys(registry: Mapping[str, object]) -> frozenset[str]:
    """Return keys governed by this authority, excluding third-party namespaces."""
    managed = {key for key in registry if key.startswith(_MANAGED_NAMESPACE_PREFIXES)}
    return frozenset(managed | RATE_EXTENSION_VARIABLE_KEYS)


def capability_for(variable_key: str) -> LocusExecutionCapability:
    """Return one declared capability or fail closed for an unknown key."""
    capability = load_locus_execution_contract().capabilities.get(variable_key)
    if capability is None:
        raise LocusContractError(f"variable is not declared: {variable_key}")
    return capability


__all__ = [
    "LocusContractError",
    "LocusExecutionCapability",
    "LocusExecutionContract",
    "RATE_EXTENSION_VARIABLE_KEYS",
    "capability_for",
    "load_locus_execution_contract",
    "managed_registry_keys",
    "parse_locus_execution_contract",
]
