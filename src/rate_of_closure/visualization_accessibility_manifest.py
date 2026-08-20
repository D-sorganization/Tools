"""Strict reader for the cross-toolkit accessibility-evidence authority."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from rate_of_closure.visualization_tab_manifest import (
    load_visualization_tab_manifest,
)

_EVIDENCE = {
    "react": "axe-core-wcag-a-aa-through-2.2",
    "pyqt": "named-visible-focusable-semantic-controls",
}


class AccessibilityManifestError(ValueError):
    """Raised when accessibility evidence does not match exact v1 authority."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AccessibilityManifestError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AccessibilityManifestError(f"non-finite JSON value: {value}")


def _object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AccessibilityManifestError(f"{context} fields must be exact")
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise AccessibilityManifestError(f"{context} must be bounded text")
    return value


@dataclass(frozen=True)
class AccessibilityEvidenceTab:
    """One surface/tab automated-evidence contract."""

    surface: str
    tab_id: str
    evidence: str


@dataclass(frozen=True)
class VisualizationAccessibilityManifest:
    """Immutable accessibility evidence and open qualification boundary."""

    schema_id: str
    schema_version: int
    automated_claim: str
    manual_at_status: str
    manual_at_protocol_path: str
    tabs: tuple[AccessibilityEvidenceTab, ...]

    def for_surface(self, surface: str) -> tuple[AccessibilityEvidenceTab, ...]:
        return tuple(entry for entry in self.tabs if entry.surface == surface)


def _parse_tab(value: object) -> AccessibilityEvidenceTab:
    tab = _object(value, {"surface", "tab_id", "evidence"}, "tab")
    surface = _text(tab["surface"], "surface")
    if surface not in _EVIDENCE:
        raise AccessibilityManifestError("unknown accessibility surface")
    evidence = _text(tab["evidence"], "evidence")
    if evidence != _EVIDENCE[surface]:
        raise AccessibilityManifestError("unsupported accessibility evidence")
    return AccessibilityEvidenceTab(surface, _text(tab["tab_id"], "tab id"), evidence)


def load_visualization_accessibility_manifest() -> VisualizationAccessibilityManifest:
    """Load, validate, and cross-check the packaged accessibility authority."""

    resource = files("rate_of_closure").joinpath("visualization_accessibility.v1.json")
    try:
        raw = json.loads(
            resource.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AccessibilityManifestError(
            "accessibility manifest is unreadable"
        ) from exc
    document = _object(
        raw,
        {"schema_id", "schema_version", "automated_claim", "manual_at", "tabs"},
        "manifest",
    )
    manual = _object(document["manual_at"], {"status", "protocol_path"}, "manual AT")
    if not isinstance(document["tabs"], list):
        raise AccessibilityManifestError("tabs must be an array")
    tabs = tuple(_parse_tab(value) for value in document["tabs"])
    identities = tuple((entry.surface, entry.tab_id) for entry in tabs)
    expected = tuple(
        (entry.surface, entry.tab_id)
        for entry in load_visualization_tab_manifest().tabs
    )
    if identities != expected or len(set(identities)) != len(identities):
        raise AccessibilityManifestError(
            "accessibility tabs must exactly match visibility authority"
        )
    version = document["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise AccessibilityManifestError("schema version must be an integer")
    manifest = VisualizationAccessibilityManifest(
        _text(document["schema_id"], "schema id"),
        version,
        _text(document["automated_claim"], "automated claim"),
        _text(manual["status"], "manual AT status"),
        _text(manual["protocol_path"], "manual AT protocol path"),
        tabs,
    )
    if (
        manifest.schema_id != "rate-of-closure/visualization-accessibility-evidence"
        or manifest.schema_version != 1
        or manifest.automated_claim
        != "protected-automated-semantics-not-manual-at-qualification"
        or manifest.manual_at_status != "protocol-ready-human-execution-required"
        or manifest.manual_at_protocol_path
        != "docs/development/rate-visualization-at-protocol.md"
    ):
        raise AccessibilityManifestError("unsupported accessibility manifest")
    return manifest


__all__ = [
    "AccessibilityEvidenceTab",
    "AccessibilityManifestError",
    "VisualizationAccessibilityManifest",
    "load_visualization_accessibility_manifest",
]
