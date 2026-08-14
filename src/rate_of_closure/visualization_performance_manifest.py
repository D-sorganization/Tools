"""Strict cross-surface visualization performance-budget authority."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from rate_of_closure.visualization_tab_manifest import (
    load_visualization_tab_manifest,
)

_SURFACES = {"react", "pyqt"}
_WORKLOAD = "initial-production-state"
_MAX_SAFE_INTEGER = 9_007_199_254_740_991


class PerformanceManifestError(ValueError):
    """Raised when performance-budget evidence is malformed or incomplete."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PerformanceManifestError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PerformanceManifestError(f"non-finite JSON value: {value}")


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise PerformanceManifestError(f"{context} fields must be exact")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise PerformanceManifestError(f"{context} must be bounded nonempty text")
    return value


def _integer(value: object, context: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
        or value > _MAX_SAFE_INTEGER
    ):
        raise PerformanceManifestError(f"{context} is outside its integer domain")
    return value


@dataclass(frozen=True)
class SurfacePerformanceBudget:
    """One toolkit's contention-tolerant layout performance envelope."""

    tab_open_budget_ms: int
    resize_settle_budget_ms: int
    stable_frame_count: int
    stability_tolerance_px: int
    max_post_settle_shift_px: int
    max_layout_shift_score_microunits: int | None


@dataclass(frozen=True)
class PerformanceTabEntry:
    """One visibility-authority tab and its measured workload."""

    surface: str
    tab_id: str
    workload: str


@dataclass(frozen=True)
class VisualizationPerformanceManifest:
    """Immutable decoded visualization performance contract."""

    schema_id: str
    schema_version: int
    measurement_policy: str
    surfaces: Mapping[str, SurfacePerformanceBudget]
    tabs: tuple[PerformanceTabEntry, ...]

    def for_surface(self, surface: str) -> tuple[PerformanceTabEntry, ...]:
        """Return exact entries for one registered toolkit."""
        return tuple(entry for entry in self.tabs if entry.surface == surface)

    def validate(self) -> None:
        """Reject coverage drift, duplicate identities, and dishonest budgets."""
        if set(self.surfaces) != _SURFACES:
            raise PerformanceManifestError("surfaces must exactly cover both toolkits")
        identities = tuple((entry.surface, entry.tab_id) for entry in self.tabs)
        if len(identities) != len(set(identities)):
            raise PerformanceManifestError("duplicate performance tab identity")
        if any(entry.surface not in _SURFACES for entry in self.tabs):
            raise PerformanceManifestError("unknown performance surface")
        if any(entry.workload != _WORKLOAD for entry in self.tabs):
            raise PerformanceManifestError("unsupported performance workload")
        expected = tuple(
            (entry.surface, entry.tab_id)
            for entry in load_visualization_tab_manifest().tabs
        )
        if identities != expected:
            raise PerformanceManifestError(
                "performance tabs must exactly match visibility authority"
            )
        for surface, budget in self.surfaces.items():
            _integer(budget.tab_open_budget_ms, "tab-open budget", 1, 10_000)
            _integer(budget.resize_settle_budget_ms, "resize budget", 1, 10_000)
            _integer(budget.stable_frame_count, "stable frames", 2, 10)
            _integer(budget.stability_tolerance_px, "stability tolerance", 0, 10)
            _integer(budget.max_post_settle_shift_px, "post-settle shift", 0, 20)
            cls = budget.max_layout_shift_score_microunits
            if surface == "react":
                _integer(cls, "layout-shift score", 0, 1_000_000)
            elif cls is not None:
                raise PerformanceManifestError("PyQt cannot declare browser CLS")


def _budget(value: object, surface: str) -> SurfacePerformanceBudget:
    if not isinstance(value, dict):
        raise PerformanceManifestError("surface budget must be an object")
    _exact_keys(
        value,
        {
            "tab_open_budget_ms",
            "resize_settle_budget_ms",
            "stable_frame_count",
            "stability_tolerance_px",
            "max_post_settle_shift_px",
            "max_layout_shift_score_microunits",
        },
        "surface budget",
    )
    cls = value["max_layout_shift_score_microunits"]
    if cls is not None:
        cls = _integer(cls, "layout-shift score", 0, 1_000_000)
    budget = SurfacePerformanceBudget(
        tab_open_budget_ms=_integer(
            value["tab_open_budget_ms"], "tab-open budget", 1, 10_000
        ),
        resize_settle_budget_ms=_integer(
            value["resize_settle_budget_ms"], "resize budget", 1, 10_000
        ),
        stable_frame_count=_integer(
            value["stable_frame_count"], "stable frames", 2, 10
        ),
        stability_tolerance_px=_integer(
            value["stability_tolerance_px"], "stability tolerance", 0, 10
        ),
        max_post_settle_shift_px=_integer(
            value["max_post_settle_shift_px"], "post-settle shift", 0, 20
        ),
        max_layout_shift_score_microunits=cls,
    )
    if surface == "react" and cls is None:
        raise PerformanceManifestError("React must declare browser CLS")
    if surface == "pyqt" and cls is not None:
        raise PerformanceManifestError("PyQt cannot declare browser CLS")
    return budget


def load_visualization_performance_manifest() -> VisualizationPerformanceManifest:
    """Load the packaged v1 performance contract and bind it to visibility."""
    path = files("rate_of_closure").joinpath("visualization_performance.v1.json")
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(raw, dict):
        raise PerformanceManifestError("manifest root must be an object")
    _exact_keys(
        raw,
        {"schema_id", "schema_version", "measurement_policy", "surfaces", "tabs"},
        "manifest",
    )
    surfaces = raw["surfaces"]
    tabs = raw["tabs"]
    if not isinstance(surfaces, dict) or set(surfaces) != _SURFACES:
        raise PerformanceManifestError("surfaces must exactly cover both toolkits")
    if not isinstance(tabs, list) or not all(isinstance(entry, dict) for entry in tabs):
        raise PerformanceManifestError("tabs must be an object array")
    for entry in tabs:
        _exact_keys(entry, {"surface", "tab_id", "workload"}, "tab")
    manifest = VisualizationPerformanceManifest(
        schema_id=_text(raw["schema_id"], "schema id"),
        schema_version=_integer(raw["schema_version"], "schema version", 1, 1),
        measurement_policy=_text(raw["measurement_policy"], "measurement policy"),
        surfaces=MappingProxyType(
            {surface: _budget(value, surface) for surface, value in surfaces.items()}
        ),
        tabs=tuple(
            PerformanceTabEntry(
                surface=_text(entry["surface"], "surface"),
                tab_id=_text(entry["tab_id"], "tab id"),
                workload=_text(entry["workload"], "workload"),
            )
            for entry in tabs
        ),
    )
    if (
        manifest.schema_id != "rate-of-closure/visualization-performance-budgets"
        or manifest.schema_version != 1
        or manifest.measurement_policy
        != "protected-diagnostic-not-user-hardware-qualification"
    ):
        raise PerformanceManifestError("unsupported performance manifest")
    manifest.validate()
    return manifest


__all__ = [
    "PerformanceManifestError",
    "PerformanceTabEntry",
    "SurfacePerformanceBudget",
    "VisualizationPerformanceManifest",
    "load_visualization_performance_manifest",
]
