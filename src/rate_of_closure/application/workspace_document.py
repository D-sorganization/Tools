"""Versioned, UI-neutral whole-workspace persistence contract.

The document composes the existing prescribed-torque and variation contracts.
Other evolving domains use explicit versioned payload envelopes so the shell
does not need to know about widgets or duplicate their domain schemas.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from shared.python.swing_sim.torque_profiles import PrescribedTorqueProfile
from shared.python.swing_sim.variation import VariationPlan

from ._workspace_validation import (
    FrozenJsonValue,
    exact_mapping,
    freeze_object,
    positive_version,
    stable_id,
    thaw_json,
    unique_json_object,
    utc_datetime,
    valid_stable_id,
)

WORKSPACE_SCHEMA = "rate_of_closure.workspace"
WORKSPACE_SCHEMA_VERSION = 2
_SUPPORTED_WORKSPACE_VERSIONS = frozenset({1, WORKSPACE_SCHEMA_VERSION})

_PAYLOAD_FIELDS = frozenset({"schema", "schema_version", "data"})
_METADATA_FIELDS = frozenset(
    {
        "document_id",
        "title",
        "created_at_utc",
        "modified_at_utc",
        "app_version",
        "provenance",
    }
)
_LAYOUT_V1_FIELDS = frozenset(
    {"module_order", "visible_module_ids", "active_module_id"}
)
_LAYOUT_FIELDS = _LAYOUT_V1_FIELDS | {"view_workspace"}
_ROOT_V1_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "metadata",
        "model_session",
        "torque_profiles",
        "club_configuration",
        "variation_plan",
        "layout",
    }
)
_ROOT_FIELDS = (_ROOT_V1_FIELDS - {"torque_profiles"}) | {"prescribed_torque_profiles"}
_VARIATION_V1_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "base_variables",
        "noise",
        "n_runs",
        "seed",
        "flight_model",
    }
)
_VARIATION_FIELDS = _VARIATION_V1_FIELDS | {"groups"}


@dataclass(frozen=True)
class VersionedPayload:
    """Generic versioned JSON payload owned by another domain contract."""

    schema: str
    schema_version: int
    data: Mapping[str, FrozenJsonValue]

    def __post_init__(self) -> None:
        """Validate identity and freeze payload data defensively."""
        stable_id(self.schema, "payload.schema")
        positive_version(self.schema_version, "payload.schema_version")
        object.__setattr__(self, "data", freeze_object(self.data, "payload.data"))

    def to_json_dict(self) -> dict[str, Any]:
        """Return a detached strict-JSON representation."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "data": thaw_json(self.data),
        }

    @classmethod
    def from_json_dict(cls, value: object) -> VersionedPayload:
        """Parse an exact generic-payload object."""
        data = exact_mapping(value, _PAYLOAD_FIELDS, "payload")
        return cls(
            schema=stable_id(data["schema"], "payload.schema"),
            schema_version=positive_version(
                data["schema_version"], "payload.schema_version"
            ),
            data=data["data"],
        )


@dataclass(frozen=True)
class WorkspaceMetadata:
    """Identity, timestamps, version, and provenance for one workspace."""

    document_id: str
    title: str
    created_at_utc: str
    modified_at_utc: str
    app_version: str
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate metadata and make provenance immutable."""
        stable_id(self.document_id, "metadata.document_id")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("metadata.title must be non-empty")
        if not isinstance(self.app_version, str) or not self.app_version.strip():
            raise ValueError("metadata.app_version must be non-empty")
        created = utc_datetime(self.created_at_utc, "metadata.created_at_utc")
        modified = utc_datetime(self.modified_at_utc, "metadata.modified_at_utc")
        if modified < created:
            raise ValueError("metadata.modified_at_utc must not precede creation")
        if not isinstance(self.provenance, Mapping) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str)
            for key, value in self.provenance.items()
        ):
            raise TypeError("metadata.provenance must map non-empty strings to strings")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def to_json_dict(self) -> dict[str, Any]:
        """Return a detached strict-JSON representation."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "created_at_utc": self.created_at_utc,
            "modified_at_utc": self.modified_at_utc,
            "app_version": self.app_version,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_json_dict(cls, value: object) -> WorkspaceMetadata:
        """Parse exact metadata fields."""
        data = exact_mapping(value, _METADATA_FIELDS, "metadata")
        provenance = data["provenance"]
        if not isinstance(provenance, Mapping):
            raise TypeError("metadata.provenance must be a JSON object")
        return cls(
            document_id=data["document_id"],
            title=data["title"],
            created_at_utc=data["created_at_utc"],
            modified_at_utc=data["modified_at_utc"],
            app_version=data["app_version"],
            provenance=provenance,
        )


@dataclass(frozen=True)
class WorkspaceLayout:
    """Generic module visibility/order with optional versioned view state."""

    module_order: tuple[str, ...]
    visible_module_ids: tuple[str, ...]
    active_module_id: str
    view_workspace: VersionedPayload | None = None

    def __post_init__(self) -> None:
        """Enforce a usable, deterministic module state."""
        order = tuple(self.module_order)
        visible = tuple(self.visible_module_ids)
        if not order or len(set(order)) != len(order):
            raise ValueError("layout.module_order must be non-empty and unique")
        if any(not valid_stable_id(item) for item in order):
            raise ValueError("layout.module_order contains an invalid module ID")
        if not visible or len(set(visible)) != len(visible):
            raise ValueError("layout.visible_module_ids must be non-empty and unique")
        if not set(visible).issubset(order):
            raise ValueError("visible modules must occur in module_order")
        if self.active_module_id not in visible:
            raise ValueError("active module must be visible")
        if self.view_workspace is not None and not isinstance(
            self.view_workspace, VersionedPayload
        ):
            raise TypeError("layout.view_workspace must be a VersionedPayload")
        object.__setattr__(self, "module_order", order)
        object.__setattr__(self, "visible_module_ids", visible)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the current version's detached JSON representation."""
        view = self.view_workspace
        return {
            "module_order": list(self.module_order),
            "visible_module_ids": list(self.visible_module_ids),
            "active_module_id": self.active_module_id,
            "view_workspace": None if view is None else view.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, value: object) -> WorkspaceLayout:
        """Parse an exact current-version layout object."""
        data = exact_mapping(value, _LAYOUT_FIELDS, "layout")
        order = data["module_order"]
        visible = data["visible_module_ids"]
        if not isinstance(order, list) or not isinstance(visible, list):
            raise TypeError("layout module collections must be JSON arrays")
        raw_view = data["view_workspace"]
        return cls(
            module_order=tuple(order),
            visible_module_ids=tuple(visible),
            active_module_id=data["active_module_id"],
            view_workspace=(
                None if raw_view is None else VersionedPayload.from_json_dict(raw_view)
            ),
        )


@dataclass(frozen=True)
class WorkspaceDocument:
    """Complete persistent application state without presentation details."""

    metadata: WorkspaceMetadata
    model_session: VersionedPayload
    prescribed_torque_profiles: tuple[PrescribedTorqueProfile, ...]
    club_configuration: VersionedPayload
    variation_plan: VariationPlan | None
    layout: WorkspaceLayout

    def __post_init__(self) -> None:
        """Validate all composed domain values and unique profile identity."""
        if not isinstance(self.metadata, WorkspaceMetadata):
            raise TypeError("metadata must be WorkspaceMetadata")
        if not isinstance(self.model_session, VersionedPayload):
            raise TypeError("model_session must be VersionedPayload")
        if not isinstance(self.club_configuration, VersionedPayload):
            raise TypeError("club_configuration must be VersionedPayload")
        if not isinstance(self.layout, WorkspaceLayout):
            raise TypeError("layout must be WorkspaceLayout")
        if self.variation_plan is not None and not isinstance(
            self.variation_plan, VariationPlan
        ):
            raise TypeError("variation_plan must be VariationPlan")
        profiles = tuple(self.prescribed_torque_profiles)
        if any(not isinstance(item, PrescribedTorqueProfile) for item in profiles):
            raise TypeError("prescribed_torque_profiles contains an invalid profile")
        profile_ids = tuple(item.profile_id for item in profiles)
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("prescribed torque profile IDs must be unique")
        object.__setattr__(self, "prescribed_torque_profiles", profiles)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the canonical current-version workspace object."""
        plan = self.variation_plan
        return {
            "schema": WORKSPACE_SCHEMA,
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "metadata": self.metadata.to_json_dict(),
            "model_session": self.model_session.to_json_dict(),
            "prescribed_torque_profiles": [
                profile.to_json_dict() for profile in self.prescribed_torque_profiles
            ],
            "club_configuration": self.club_configuration.to_json_dict(),
            "variation_plan": None if plan is None else plan.to_json_dict(),
            "layout": self.layout.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, value: object) -> WorkspaceDocument:
        """Validate, migrate, and parse a supported workspace object."""
        current = _migrate_workspace(value)
        data = exact_mapping(current, _ROOT_FIELDS, "workspace")
        raw_profiles = data["prescribed_torque_profiles"]
        if not isinstance(raw_profiles, list):
            raise TypeError("prescribed_torque_profiles must be a JSON array")
        raw_plan = data["variation_plan"]
        return cls(
            metadata=WorkspaceMetadata.from_json_dict(data["metadata"]),
            model_session=VersionedPayload.from_json_dict(data["model_session"]),
            prescribed_torque_profiles=tuple(
                PrescribedTorqueProfile.from_json_dict(item) for item in raw_profiles
            ),
            club_configuration=VersionedPayload.from_json_dict(
                data["club_configuration"]
            ),
            variation_plan=(
                None if raw_plan is None else _variation_from_json(raw_plan)
            ),
            layout=WorkspaceLayout.from_json_dict(data["layout"]),
        )


def _variation_from_json(value: object) -> VariationPlan:
    if not isinstance(value, Mapping):
        raise TypeError("variation_plan must be a JSON object")
    version = value.get("schema_version")
    expected = _VARIATION_V1_FIELDS if version == 1 else _VARIATION_FIELDS
    data = exact_mapping(value, expected, "variation_plan")
    return VariationPlan.from_json_dict(data)


def _migrate_workspace(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("workspace must be a JSON object")
    if value.get("schema") != WORKSPACE_SCHEMA:
        raise ValueError("unsupported workspace schema")
    version = value.get("schema_version")
    if type(version) is not int or version not in _SUPPORTED_WORKSPACE_VERSIONS:
        raise ValueError(f"unsupported workspace schema_version {version!r}")
    if version == WORKSPACE_SCHEMA_VERSION:
        current: Mapping[str, Any] = exact_mapping(value, _ROOT_FIELDS, "workspace")
        return current
    legacy = exact_mapping(value, _ROOT_V1_FIELDS, "workspace v1")
    legacy_layout = exact_mapping(legacy["layout"], _LAYOUT_V1_FIELDS, "layout v1")
    migrated = dict(legacy)
    migrated["schema_version"] = WORKSPACE_SCHEMA_VERSION
    migrated["prescribed_torque_profiles"] = migrated.pop("torque_profiles")
    migrated["layout"] = {**legacy_layout, "view_workspace": None}
    return migrated


def workspace_to_json(document: WorkspaceDocument) -> str:
    """Serialize a validated workspace deterministically as strict JSON."""
    if not isinstance(document, WorkspaceDocument):
        raise TypeError("document must be a WorkspaceDocument")
    return (
        json.dumps(document.to_json_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"invalid non-finite JSON number: {value}")


def workspace_from_json(text: str) -> WorkspaceDocument:
    """Parse strict JSON, rejecting duplicate keys before migration."""
    if not isinstance(text, str):
        raise TypeError("workspace JSON must be text")
    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_json_object,
            parse_constant=_reject_non_finite_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("invalid workspace JSON") from exc
    return WorkspaceDocument.from_json_dict(value)


__all__ = [
    "WORKSPACE_SCHEMA",
    "WORKSPACE_SCHEMA_VERSION",
    "VersionedPayload",
    "WorkspaceDocument",
    "WorkspaceLayout",
    "WorkspaceMetadata",
    "workspace_from_json",
    "workspace_to_json",
]
