"""Versioned all-tab visualization performance-budget authority."""

from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import pytest

from rate_of_closure.ui.pyqt6.navigation_state import DEFAULT_TAB_IDS
from rate_of_closure.visualization_performance_manifest import (
    PerformanceManifestError,
    load_visualization_performance_manifest,
)
from rate_of_closure.visualization_tab_manifest import (
    load_visualization_tab_manifest,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _document() -> dict[str, object]:
    path = (
        Path(__file__).parents[2]
        / "src"
        / "rate_of_closure"
        / "visualization_performance.v1.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_document(value: object) -> object:
    text = json.dumps(value)
    resource = type("Resource", (), {"read_text": lambda self, **_kwargs: text})()
    with patch("rate_of_closure.visualization_performance_manifest.files") as package:
        package.return_value.joinpath.return_value = resource
        return load_visualization_performance_manifest()


def test_performance_manifest_exactly_covers_visibility_authority() -> None:
    performance = load_visualization_performance_manifest()
    visibility = load_visualization_tab_manifest()

    assert performance.schema_id == (
        "rate-of-closure/visualization-performance-budgets"
    )
    assert performance.schema_version == 1
    assert performance.measurement_policy == (
        "protected-diagnostic-not-user-hardware-qualification"
    )
    expected = tuple((entry.surface, entry.tab_id) for entry in visibility.tabs)
    assert (
        tuple((entry.surface, entry.tab_id) for entry in performance.tabs) == expected
    )
    assert (
        tuple(entry.tab_id for entry in performance.for_surface("pyqt"))
        == DEFAULT_TAB_IDS
    )
    assert all(
        entry.workload == "initial-production-state" for entry in performance.tabs
    )


def test_surface_budgets_are_bounded_and_toolkit_honest() -> None:
    manifest = load_visualization_performance_manifest()
    react = manifest.surfaces["react"]
    pyqt = manifest.surfaces["pyqt"]

    assert react.tab_open_budget_ms == 2_500
    assert react.resize_settle_budget_ms == 1_500
    assert react.stable_frame_count == 3
    assert react.stability_tolerance_px == 1
    assert react.max_post_settle_shift_px == 2
    assert react.max_layout_shift_score_microunits == 100_000
    assert pyqt.tab_open_budget_ms == 5_000
    assert pyqt.resize_settle_budget_ms == 4_000
    assert pyqt.max_layout_shift_score_microunits is None
    assert isinstance(manifest.surfaces, MappingProxyType)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tab_open_budget_ms", 0),
        ("resize_settle_budget_ms", 10_001),
        ("stable_frame_count", 1),
        ("stability_tolerance_px", -1),
        ("max_post_settle_shift_px", True),
        ("max_layout_shift_score_microunits", 1_000_001),
    ],
)
def test_reader_rejects_unbounded_or_forged_budgets(field: str, value: object) -> None:
    document = _document()
    surfaces = document["surfaces"]
    assert isinstance(surfaces, dict)
    react = surfaces["react"]
    assert isinstance(react, dict)
    react[field] = value

    with pytest.raises(PerformanceManifestError):
        _load_document(document)


def test_reader_rejects_missing_duplicate_or_unknown_tab_identity() -> None:
    document = _document()
    tabs = document["tabs"]
    assert isinstance(tabs, list)
    tabs.pop()
    with pytest.raises(PerformanceManifestError, match="visibility authority"):
        _load_document(document)

    document = _document()
    tabs = document["tabs"]
    assert isinstance(tabs, list)
    tabs.append(dict(tabs[0]))
    with pytest.raises(PerformanceManifestError, match="duplicate"):
        _load_document(document)

    manifest = load_visualization_performance_manifest()
    entry = manifest.tabs[0]
    with pytest.raises(PerformanceManifestError, match="surface"):
        replace(
            manifest, tabs=(replace(entry, surface="desktop"), *manifest.tabs[1:])
        ).validate()


def test_manifest_is_deeply_immutable() -> None:
    manifest = load_visualization_performance_manifest()
    assert isinstance(manifest.surfaces, MappingProxyType)
    with pytest.raises(TypeError):
        manifest.surfaces["react"] = manifest.surfaces["pyqt"]  # type: ignore[index]


def test_all_visualization_authorities_are_declared_as_package_data() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    packaged = set(project["tool"]["setuptools"]["package-data"]["rate_of_closure"])
    authorities = {
        "visualization_tabs.v1.json",
        "visualization_performance.v1.json",
        "visualization_accessibility.v1.json",
        "visual_baselines.v1.json",
        "visual_baselines/v1/react/*.png",
        "visual_baselines/v1/pyqt/*.png",
    }
    # Every visualization authority must be declared -- that is this test's
    # contract. It is deliberately not an equality check: the built web
    # distribution is also legitimate package data, and pinning the whole list
    # to one feature's entries makes any other feature's packaging a failure.
    # Anything outside both sets is still rejected, so drift is still caught.
    assert authorities <= packaged
    assert not {
        entry for entry in packaged - authorities if not entry.startswith("web/dist/")
    }
