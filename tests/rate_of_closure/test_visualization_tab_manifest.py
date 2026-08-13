"""Versioned cross-surface visualization-tab manifest contract."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from rate_of_closure.ui.pyqt6.navigation_state import DEFAULT_TAB_IDS
from rate_of_closure.visualization_tab_manifest import (
    ManifestContractError,
    audit_registered_tabs,
    load_visualization_tab_manifest,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


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
    }
    assert pyqt.responsive_control_locators == {}
    assert max(react.viewport_px or ()) <= 10_000


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
