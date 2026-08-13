"""Strict reader and governance audit for visualization-tab manifest v1."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any

_SURFACES = {"react", "pyqt"}
_CLASSIFICATIONS = {
    "visual-first",
    "form-led-live-preview",
    "form-led-evidence",
    "reference-utility",
}
_LANDMARK_KINDS = {"visual", "semantic-content"}
_STATE_KEYS = {"empty", "loading", "result", "error"}
_MAX_SAFE_INTEGER = 9_007_199_254_740_991


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ManifestContractError(f"{context} fields must be exact")


def _positive_int(value: object, context: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > _MAX_SAFE_INTEGER
    ):
        raise ManifestContractError(f"{context} must be a positive safe integer")
    return value


def _dimension(value: object, context: str) -> int:
    parsed = _positive_int(value, context)
    if parsed > 10_000:
        raise ManifestContractError(f"{context} exceeds the bounded viewport domain")
    return parsed


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ManifestContractError(f"{context} must be bounded nonempty text")
    return value


def _reject_constant(value: str) -> None:
    raise ManifestContractError(f"non-finite JSON value: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestContractError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


class ManifestContractError(ValueError):
    """Raised when the committed visualization manifest is not exact v1 data."""


@dataclass(frozen=True)
class ReferenceEnvironment:
    """One surface's deterministic reference viewport or DPI scales."""

    viewport_px: tuple[int, int] | None
    additional_viewports_px: tuple[tuple[int, int], ...]
    responsive_minimum_visible_height_px: int | None
    minimum_visible_width_px: int
    responsive_minimum_visible_width_px: int | None
    responsive_control_locators: Mapping[str, str]
    dpi_scales: tuple[float, ...]


@dataclass(frozen=True)
class VisualizationTabEntry:
    """One registered primary tab and its first-screen audit contract."""

    surface: str
    tab_id: str
    classification: str
    landmark_kind: str
    minimum_visible_height_px: int
    primary_visual_locator: str
    states: Mapping[str, str]


@dataclass(frozen=True)
class VisualizationTabManifest:
    """Immutable decoded form of visualization-tab-visibility@1."""

    schema_id: str
    schema_version: int
    artifact_policy: str
    reference_environments: Mapping[str, ReferenceEnvironment]
    tabs: tuple[VisualizationTabEntry, ...]

    def for_surface(self, surface: str) -> tuple[VisualizationTabEntry, ...]:
        """Return entries for one exact surface in manifest order."""
        return tuple(entry for entry in self.tabs if entry.surface == surface)

    def validate(self) -> None:
        """Reject duplicate identities and incomplete state declarations."""
        identities = [(entry.surface, entry.tab_id) for entry in self.tabs]
        if len(identities) != len(set(identities)):
            raise ManifestContractError("duplicate visualization tab identity")
        if any(set(entry.states) != _STATE_KEYS for entry in self.tabs):
            raise ManifestContractError("every tab must declare all four states")
        if set(self.reference_environments) != _SURFACES:
            raise ManifestContractError(
                "reference environments must cover both surfaces"
            )
        for surface, environment in self.reference_environments.items():
            if environment.viewport_px is None:
                raise ManifestContractError(f"{surface} viewport is required")
            for dimension in environment.viewport_px:
                _dimension(dimension, f"{surface} viewport dimension")
            for viewport in environment.additional_viewports_px:
                if len(viewport) != 2:
                    raise ManifestContractError("additional viewport must be a pair")
                for dimension in viewport:
                    _dimension(dimension, "additional viewport dimension")
            _positive_int(
                environment.minimum_visible_width_px,
                f"{surface} minimum visible width",
            )
            for name, value in (
                (
                    "responsive minimum visible height",
                    environment.responsive_minimum_visible_height_px,
                ),
                (
                    "responsive minimum visible width",
                    environment.responsive_minimum_visible_width_px,
                ),
            ):
                if value is not None:
                    _positive_int(value, name)
            if any(
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(scale)
                or scale <= 0
                for scale in environment.dpi_scales
            ):
                raise ManifestContractError("DPI scales must be positive numbers")
        for entry in self.tabs:
            if entry.surface not in _SURFACES:
                raise ManifestContractError("unknown visualization surface")
            if entry.classification not in _CLASSIFICATIONS:
                raise ManifestContractError("unknown visualization classification")
            if entry.landmark_kind not in _LANDMARK_KINDS:
                raise ManifestContractError("unknown landmark kind")
            if (
                entry.classification == "reference-utility"
                and entry.landmark_kind != "semantic-content"
            ):
                raise ManifestContractError(
                    "reference utilities require semantic content"
                )
            if (
                entry.classification
                in {
                    "visual-first",
                    "form-led-live-preview",
                    "form-led-evidence",
                }
                and entry.landmark_kind != "visual"
            ):
                raise ManifestContractError(
                    f"{entry.classification} tabs require visual landmarks"
                )
            _positive_int(entry.minimum_visible_height_px, "minimum visible height")
            expected = 240 if entry.landmark_kind == "visual" else 1
            if entry.minimum_visible_height_px != expected:
                raise ManifestContractError("landmark minimum does not match its kind")
            if entry.surface == "pyqt" and entry.primary_visual_locator.endswith(
                ("_scroll", "_tabs", "_view")
            ):
                raise ManifestContractError("PyQt locator must identify a content leaf")
        expected_react_controls = {
            entry.tab_id
            for entry in self.tabs
            if entry.surface == "react" and entry.landmark_kind == "visual"
        }
        if (
            set(self.reference_environments["react"].responsive_control_locators)
            != expected_react_controls
        ):
            raise ManifestContractError(
                "React responsive control locators must exactly cover visual tabs"
            )
        if self.reference_environments["pyqt"].responsive_control_locators:
            raise ManifestContractError(
                "PyQt reference environment cannot declare responsive controls"
            )


def _environment(value: object, surface: str) -> ReferenceEnvironment:
    if not isinstance(value, dict):
        raise ManifestContractError("reference environment must be an object")
    expected = {"viewport_px", "minimum_visible_width_px", "dpi_scales"}
    if surface == "react":
        expected.update(
            {
                "additional_viewports_px",
                "responsive_minimum_visible_height_px",
                "responsive_minimum_visible_width_px",
                "responsive_control_locators",
            }
        )
    if set(value) != expected:
        raise ManifestContractError("reference environment fields are invalid")
    viewport = value.get("viewport_px")
    if not isinstance(viewport, list) or len(viewport) != 2:
        raise ManifestContractError("viewport must contain two dimensions")
    dimensions = (
        _dimension(viewport[0], "viewport width"),
        _dimension(viewport[1], "viewport height"),
    )
    additional = value.get("additional_viewports_px", ())
    if not isinstance(additional, (list, tuple)):
        raise ManifestContractError("additional viewports must be a sequence")
    if any(not isinstance(item, list) or len(item) != 2 for item in additional):
        raise ManifestContractError(
            "each additional viewport must contain two dimensions"
        )
    dpi = value.get("dpi_scales", ())
    if not isinstance(dpi, (list, tuple)) or any(
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
        for scale in dpi
    ):
        raise ManifestContractError("DPI scales must be positive numbers")
    responsive = value.get("responsive_minimum_visible_height_px")
    if responsive is not None:
        responsive = _positive_int(responsive, "responsive minimum")
    minimum_width = _positive_int(
        value.get("minimum_visible_width_px"), "minimum width"
    )
    responsive_width = value.get("responsive_minimum_visible_width_px")
    if responsive_width is not None:
        responsive_width = _positive_int(responsive_width, "responsive minimum width")
    controls = value.get("responsive_control_locators", {})
    if not isinstance(controls, dict):
        raise ManifestContractError("responsive control locators must be an object")
    return ReferenceEnvironment(
        viewport_px=dimensions,
        additional_viewports_px=tuple(
            (
                _dimension(item[0], "viewport width"),
                _dimension(item[1], "viewport height"),
            )
            for item in additional
        ),
        responsive_minimum_visible_height_px=responsive,
        minimum_visible_width_px=minimum_width,
        responsive_minimum_visible_width_px=responsive_width,
        responsive_control_locators=MappingProxyType(
            {
                _text(key, "control tab id"): _text(locator, "control locator")
                for key, locator in controls.items()
            }
        ),
        dpi_scales=tuple(float(scale) for scale in dpi),
    )


def load_visualization_tab_manifest() -> VisualizationTabManifest:
    """Load the packaged machine-readable manifest through its strict v1 shape."""
    path = files("rate_of_closure").joinpath("visualization_tabs.v1.json")
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(raw, dict):
        raise ManifestContractError("manifest root must be an object")
    _exact_keys(
        raw,
        {
            "schema_id",
            "schema_version",
            "artifact_policy",
            "reference_environments",
            "tabs",
        },
        "manifest",
    )
    environments = raw["reference_environments"]
    tabs = raw["tabs"]
    if not isinstance(environments, dict) or set(environments) != _SURFACES:
        raise ManifestContractError("reference environments must cover both surfaces")
    if not isinstance(tabs, list) or not all(isinstance(entry, dict) for entry in tabs):
        raise ManifestContractError("tabs must be an object array")
    for entry in tabs:
        _exact_keys(
            entry,
            {
                "surface",
                "tab_id",
                "classification",
                "landmark_kind",
                "minimum_visible_height_px",
                "primary_visual_locator",
                "states",
            },
            "tab",
        )
        if not isinstance(entry["states"], dict):
            raise ManifestContractError("states must be an object")
        _exact_keys(entry["states"], _STATE_KEYS, "states")
    manifest = VisualizationTabManifest(
        schema_id=_text(raw["schema_id"], "schema id"),
        schema_version=_positive_int(raw["schema_version"], "schema version"),
        artifact_policy=_text(raw["artifact_policy"], "artifact policy"),
        reference_environments=MappingProxyType(
            {key: _environment(value, key) for key, value in environments.items()}
        ),
        tabs=tuple(
            VisualizationTabEntry(
                surface=_text(entry["surface"], "surface"),
                tab_id=_text(entry["tab_id"], "tab id"),
                classification=_text(entry["classification"], "classification"),
                landmark_kind=_text(entry["landmark_kind"], "landmark kind"),
                minimum_visible_height_px=_positive_int(
                    entry["minimum_visible_height_px"], "minimum visible height"
                ),
                primary_visual_locator=_text(
                    entry["primary_visual_locator"], "locator"
                ),
                states=MappingProxyType(
                    {
                        key: _text(value, f"{key} state")
                        for key, value in entry["states"].items()
                    }
                ),
            )
            for entry in tabs
        ),
    )
    if (
        manifest.schema_id != "rate-of-closure/visualization-tab-visibility"
        or manifest.schema_version != 1
        or manifest.artifact_policy != "diagnostic-only-not-approved-golden"
    ):
        raise ManifestContractError("unsupported visualization manifest schema")
    manifest.validate()
    return manifest


def audit_registered_tabs(
    manifest: VisualizationTabManifest, surface: str, registered: tuple[str, ...]
) -> tuple[str, ...]:
    """Return deterministic missing/unregistered governance findings."""
    documented = tuple(entry.tab_id for entry in manifest.for_surface(surface))
    findings = [
        f"missing manifest entry for {surface} tab {tab_id}"
        for tab_id in registered
        if tab_id not in documented
    ]
    findings.extend(
        f"unregistered manifest entry for {surface} tab {tab_id}"
        for tab_id in documented
        if tab_id not in registered
    )
    return tuple(findings)


__all__ = [
    "ManifestContractError",
    "ReferenceEnvironment",
    "VisualizationTabEntry",
    "VisualizationTabManifest",
    "audit_registered_tabs",
    "load_visualization_tab_manifest",
]
