"""Versioned cross-surface visualization-tab manifest contract."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import pytest

from rate_of_closure.ui.pyqt6.navigation_state import DEFAULT_TAB_IDS
from rate_of_closure.visualization_tab_manifest import (
    ManifestContractError,
    audit_registered_tabs,
    load_visualization_tab_manifest,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _manifest_document() -> dict[str, object]:
    path = (
        Path(__file__).parents[2]
        / "src"
        / "rate_of_closure"
        / "visualization_tabs.v1.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_document(value: object) -> object:
    text = json.dumps(value)
    resource = type("Resource", (), {"read_text": lambda self, **_kwargs: text})()
    with patch("rate_of_closure.visualization_tab_manifest.files") as package:
        package.return_value.joinpath.return_value = resource
        return load_visualization_tab_manifest()


def test_manifest_v1_covers_every_registered_pyqt_tab() -> None:
    manifest = load_visualization_tab_manifest()

    assert manifest.schema_id == "rate-of-closure/visualization-tab-visibility"
    assert manifest.schema_version == 1
    assert manifest.artifact_policy == "diagnostic-only-not-approved-golden"
    assert manifest.reference_environments["react"].viewport_px == (1440, 900)
    assert manifest.reference_environments["react"].additional_viewports_px == (
        (1280, 720),
        (390, 844),
    )
    assert manifest.reference_environments["pyqt"].dpi_scales == (1.0, 1.5)
    entries = manifest.for_surface("pyqt")
    assert tuple(entry.tab_id for entry in entries) == DEFAULT_TAB_IDS
    assert audit_registered_tabs(manifest, "pyqt", DEFAULT_TAB_IDS) == ()
    assert all(entry.primary_visual_locator for entry in entries)
    assert all(
        set(entry.states) == {"empty", "loading", "result", "error"}
        for entry in entries
    )


def test_flight_manifest_names_synchronous_atomic_inspector_states() -> None:
    manifest = load_visualization_tab_manifest()
    react = next(
        entry
        for entry in manifest.tabs
        if entry.surface == "react" and entry.tab_id == "flight"
    )
    pyqt = next(
        entry
        for entry in manifest.tabs
        if entry.surface == "pyqt" and entry.tab_id == "flight_explorer"
    )
    assert react.states == MappingProxyType(
        {
            "empty": "placeholder-canvas",
            "loading": "synchronous",
            "result": "bounded-synchronized-sample-inspector",
            "error": "alert-and-prior-or-empty-inspector",
        }
    )
    assert pyqt.states["result"] == "bounded-synchronized-sample-inspector"
    assert pyqt.states["error"] == (
        "status-and-prior-or-empty-inspector;stale-warning-on-restoration-failure"
    )


def test_simulation_manifest_names_honest_synchronous_retained_states() -> None:
    manifest = load_visualization_tab_manifest()
    react = next(
        entry
        for entry in manifest.tabs
        if entry.surface == "react" and entry.tab_id == "simulation"
    )
    pyqt = next(
        entry
        for entry in manifest.tabs
        if entry.surface == "pyqt" and entry.tab_id == "simulation"
    )
    assert react.states["loading"] == "synchronous-not-observable"
    assert pyqt.states["loading"] == "synchronous-not-observable"
    assert react.states["error"] == "status-and-prior-or-empty-scene"
    assert pyqt.states["error"] == "status-and-prior-or-empty-scene"


def test_governance_rejects_a_missing_or_duplicate_registered_tab() -> None:
    manifest = load_visualization_tab_manifest()
    pyqt = manifest.for_surface("pyqt")
    missing = replace(
        manifest,
        tabs=tuple(entry for entry in manifest.tabs if entry is not pyqt[0]),
    )
    assert audit_registered_tabs(missing, "pyqt", DEFAULT_TAB_IDS) == (
        f"missing manifest entry for pyqt tab {pyqt[0].tab_id}",
    )

    duplicate = replace(manifest, tabs=manifest.tabs + (pyqt[0],))
    with pytest.raises(ManifestContractError, match="duplicate"):
        duplicate.validate()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"surface": "unknown"}, "surface"),
        ({"classification": "container-led"}, "classification"),
        ({"minimum_visible_height_px": -1}, "minimum"),
        ({"primary_visual_locator": "attr:_scroll"}, "content leaf"),
    ],
)
def test_manifest_validation_rejects_adversarial_entries(
    changes: dict[str, object],
    message: str,
) -> None:
    manifest = load_visualization_tab_manifest()
    entry = manifest.for_surface("pyqt")[0]
    entries = list(manifest.tabs)
    entries[entries.index(entry)] = replace(entry, **changes)
    tampered = replace(manifest, tabs=tuple(entries))

    with pytest.raises(ManifestContractError, match=message):
        tampered.validate()


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_id":"a","schema_id":"b"}',
        '{"schema_id":NaN}',
        "[]",
    ],
)
def test_manifest_reader_rejects_malformed_raw_json(text: str) -> None:
    resource = type("Resource", (), {"read_text": lambda self, **_kwargs: text})()
    with patch("rate_of_closure.visualization_tab_manifest.files") as package:
        package.return_value.joinpath.return_value = resource
        with pytest.raises(ManifestContractError):
            load_visualization_tab_manifest()


def test_reference_environments_reject_cross_surface_and_unbounded_fields() -> None:
    manifest = load_visualization_tab_manifest()
    react = manifest.reference_environments["react"]
    pyqt = manifest.reference_environments["pyqt"]
    assert react.responsive_control_locators == {
        "explorer": "section[aria-label='Scenario inputs']",
        "simulation": "section[aria-label='Simulation setup']",
        "plots": "section[aria-label='Plot management']",
        "flight": "section[aria-label='Flight explorer inputs']",
        "launch-monitor-analytics": "section[aria-label='Analysis contract']",
        "variation": "section[aria-label='Variation setup']",
        "putting": "section[aria-label='Putt setup']",
        "neural-model-lab": "section[aria-label='Neural Model Lab']",
    }
    assert pyqt.responsive_control_locators == {}
    assert max(react.viewport_px or ()) <= 10_000
    assert react.responsive_minimum_visible_height_px == 180


def test_classification_landmark_relationships_are_exact() -> None:
    manifest = load_visualization_tab_manifest()
    entry = manifest.for_surface("react")[0]
    with pytest.raises(ManifestContractError, match="visual-first"):
        replace(
            manifest,
            tabs=(
                replace(
                    entry, landmark_kind="semantic-content", minimum_visible_height_px=1
                ),
                *manifest.tabs[1:],
            ),
        ).validate()


def test_manifest_is_deeply_immutable() -> None:
    manifest = load_visualization_tab_manifest()
    assert isinstance(manifest.reference_environments, MappingProxyType)
    assert isinstance(manifest.tabs[0].states, MappingProxyType)
    assert isinstance(
        manifest.reference_environments["react"].responsive_control_locators,
        MappingProxyType,
    )
    with pytest.raises(TypeError):
        manifest.tabs[0].states["empty"] = "tampered"  # type: ignore[index]


def test_exact_react_control_map_and_classification_compatibility() -> None:
    manifest = load_visualization_tab_manifest()
    react = manifest.reference_environments["react"]
    assert set(react.responsive_control_locators) == {
        entry.tab_id
        for entry in manifest.for_surface("react")
        if entry.landmark_kind == "visual"
    }
    entry = next(
        item for item in manifest.tabs if item.classification == "form-led-live-preview"
    )
    entries = list(manifest.tabs)
    entries[entries.index(entry)] = replace(
        entry, landmark_kind="semantic-content", minimum_visible_height_px=1
    )
    with pytest.raises(ManifestContractError, match="live-preview"):
        replace(manifest, tabs=tuple(entries)).validate()

    evidence = replace(entry, classification="form-led-evidence")
    entries = list(manifest.tabs)
    entries[entries.index(entry)] = replace(
        evidence, landmark_kind="semantic-content", minimum_visible_height_px=1
    )
    with pytest.raises(ManifestContractError, match="form-led-evidence"):
        replace(manifest, tabs=tuple(entries)).validate()


@pytest.mark.parametrize("mutation", ["missing", "typo", "extra"])
def test_reader_requires_exact_react_responsive_control_keys(mutation: str) -> None:
    document = _manifest_document()
    environments = document["reference_environments"]
    assert isinstance(environments, dict)
    react = environments["react"]
    assert isinstance(react, dict)
    controls = react["responsive_control_locators"]
    assert isinstance(controls, dict)
    if mutation == "missing":
        controls.pop("putting")
    elif mutation == "typo":
        controls["puting"] = controls.pop("putting")
    else:
        controls["calculation"] = "section[aria-label='Calculation']"

    with pytest.raises(ManifestContractError, match="exactly cover visual tabs"):
        _load_document(document)


def test_reader_rejects_responsive_fields_on_pyqt_environment() -> None:
    document = _manifest_document()
    environments = document["reference_environments"]
    assert isinstance(environments, dict)
    pyqt = environments["pyqt"]
    assert isinstance(pyqt, dict)
    pyqt["responsive_control_locators"] = {}

    with pytest.raises(ManifestContractError, match="environment fields"):
        _load_document(document)


def test_reader_rejects_one_pixel_responsive_visual_height() -> None:
    document = _manifest_document()
    environments = document["reference_environments"]
    assert isinstance(environments, dict)
    react = environments["react"]
    assert isinstance(react, dict)
    react["responsive_minimum_visible_height_px"] = 1

    with pytest.raises(ManifestContractError, match="meaningful visual height"):
        _load_document(document)


@pytest.mark.parametrize("value", [2**60, 2**53, True, 1.5, "240", float("inf")])
def test_pixel_domains_reject_non_shared_safe_integers(value: object) -> None:
    document = _manifest_document()
    environments_document = document["reference_environments"]
    assert isinstance(environments_document, dict)
    react_document = environments_document["react"]
    assert isinstance(react_document, dict)
    react_document["responsive_minimum_visible_height_px"] = value
    with pytest.raises(ManifestContractError):
        _load_document(document)

    manifest = load_visualization_tab_manifest()
    entry = manifest.tabs[0]
    entries = (replace(entry, minimum_visible_height_px=value), *manifest.tabs[1:])
    with pytest.raises(ManifestContractError):
        replace(manifest, tabs=entries).validate()

    react = manifest.reference_environments["react"]
    environments = dict(manifest.reference_environments)
    environments["react"] = replace(react, minimum_visible_width_px=value)
    with pytest.raises(ManifestContractError):
        replace(
            manifest, reference_environments=MappingProxyType(environments)
        ).validate()
