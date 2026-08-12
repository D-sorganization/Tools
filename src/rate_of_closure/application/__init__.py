"""Lazy UI-neutral application exports for optional-dependency safety."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULE = {
    "APP_COMMAND_IDS": ".commands",
    "AppCommandId": ".commands",
    "CommandAvailability": ".commands",
    "CommandUnavailableError": ".commands",
    "WORKSPACE_SCHEMA": ".workspace_document",
    "WORKSPACE_SCHEMA_VERSION": ".workspace_document",
    "VersionedPayload": ".workspace_document",
    "WorkspaceDocument": ".workspace_document",
    "WorkspaceLayout": ".workspace_document",
    "WorkspaceMetadata": ".workspace_document",
    "workspace_from_json": ".workspace_document",
    "workspace_to_json": ".workspace_document",
    "read_workspace": ".workspace_files",
    "write_workspace_atomic": ".workspace_files",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULE:
        raise AttributeError(name)
    return getattr(import_module(_EXPORT_MODULE[name], __name__), name)


__all__ = [
    "APP_COMMAND_IDS",
    "WORKSPACE_SCHEMA",
    "WORKSPACE_SCHEMA_VERSION",
    "AppCommandId",
    "CommandAvailability",
    "CommandUnavailableError",
    "VersionedPayload",
    "WorkspaceDocument",
    "WorkspaceLayout",
    "WorkspaceMetadata",
    "read_workspace",
    "workspace_from_json",
    "workspace_to_json",
    "write_workspace_atomic",
]
