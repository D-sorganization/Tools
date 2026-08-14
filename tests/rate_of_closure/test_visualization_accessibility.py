"""Cross-toolkit accessibility authority and PyQt semantic audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR, QSettings  # noqa: E402

from rate_of_closure.ui.pyqt6.accessibility_audit import (  # noqa: E402
    audit_visible_focusable_controls,
)
from rate_of_closure.ui.pyqt6.main_window import (  # noqa: E402
    RateOfClosureMainWindow,
)
from rate_of_closure.visualization_accessibility_manifest import (  # noqa: E402
    AccessibilityManifestError,
    load_visualization_accessibility_manifest,
)
from rate_of_closure.visualization_tab_manifest import (  # noqa: E402
    load_visualization_tab_manifest,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_EXPECTED_INITIAL_CONTROL_COUNTS = {
    "clubhead": frozenset({8}),
    "plots": frozenset({16, 17}),
    "calculation_description": frozenset({0}),
    "simulation": frozenset({35}),
    "flight_explorer": frozenset({37}),
    "launch_monitor_analytics": frozenset({15}),
    "variation": frozenset({17}),
    "putting": frozenset({8}),
    "glossary": frozenset({2}),
}


def _document() -> dict[str, object]:
    path = (
        Path(__file__).parents[2]
        / "src"
        / "rate_of_closure"
        / "visualization_accessibility.v1.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load(value: object) -> object:
    text = json.dumps(value)
    resource = type("Resource", (), {"read_text": lambda self, **_kwargs: text})()
    with patch("rate_of_closure.visualization_accessibility_manifest.files") as package:
        package.return_value.joinpath.return_value = resource
        return load_visualization_accessibility_manifest()


def test_accessibility_manifest_exactly_covers_visibility_authority() -> None:
    manifest = load_visualization_accessibility_manifest()
    visibility = load_visualization_tab_manifest()

    assert tuple((entry.surface, entry.tab_id) for entry in manifest.tabs) == tuple(
        (entry.surface, entry.tab_id) for entry in visibility.tabs
    )
    assert manifest.automated_claim == (
        "protected-automated-semantics-not-manual-at-qualification"
    )
    assert manifest.manual_at_status == "protocol-ready-human-execution-required"
    assert manifest.manual_at_protocol_path.endswith(
        "rate-visualization-at-protocol.md"
    )


def test_pyqt_control_count_authority_has_one_bounded_platform_envelope() -> None:
    assert _EXPECTED_INITIAL_CONTROL_COUNTS["plots"] == frozenset({16, 17})
    assert all(
        len(counts) == 1
        for tab_id, counts in _EXPECTED_INITIAL_CONTROL_COUNTS.items()
        if tab_id != "plots"
    )


def test_accessibility_manifest_rejects_coverage_and_claim_drift() -> None:
    missing = _document()
    tabs = missing["tabs"]
    assert isinstance(tabs, list)
    tabs.pop()
    with pytest.raises(AccessibilityManifestError, match="visibility authority"):
        _load(missing)

    false_claim = _document()
    false_claim["automated_claim"] = "manual-screen-reader-approved"
    with pytest.raises(AccessibilityManifestError, match="unsupported"):
        _load(false_claim)

    false_version = _document()
    false_version["schema_version"] = True
    with pytest.raises(AccessibilityManifestError, match="integer"):
        _load(false_version)

    false_protocol = _document()
    manual = false_protocol["manual_at"]
    assert isinstance(manual, dict)
    manual["protocol_path"] = "docs/approved-at-result.md"
    with pytest.raises(AccessibilityManifestError, match="unsupported"):
        _load(false_protocol)


def test_every_visible_focusable_pyqt_control_has_a_bounded_name(
    qtbot,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    settings = QSettings(str(tmp_path / "navigation.ini"), QSettings.Format.IniFormat)
    window = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(window)
    window.resize(1440, 900)
    window.show()

    expected_ids = [
        entry.tab_id
        for entry in load_visualization_accessibility_manifest().for_surface("pyqt")
    ]
    assert window.primary_tab_ids() == expected_ids
    evidence: list[dict[str, object]] = []
    for index, tab_id in enumerate(expected_ids):
        window._tabs.setCurrentIndex(index)
        qtbot.wait(0)
        page = window._tabs.currentWidget()
        assert page is not None
        result = audit_visible_focusable_controls(page)
        assert result.control_count in _EXPECTED_INITIAL_CONTROL_COUNTS[tab_id]
        assert result.findings == (), tab_id
        evidence.append(
            {
                "tab_id": tab_id,
                "audited_control_count": result.control_count,
                "findings": [],
            }
        )

    output_root = os.environ.get("RATE_PYQT_EVIDENCE_DIR")
    if output_root:
        output = Path(output_root)
        output.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_id": "rate-of-closure/pyqt-accessibility-audit-result",
            "schema_version": 1,
            "policy": "automated-semantics-not-manual-at-qualification",
            "build_identity": os.environ.get("GITHUB_SHA", "local-diagnostic"),
            "qt_version": QT_VERSION_STR,
            "pyqt_version": PYQT_VERSION_STR,
            "tabs": evidence,
        }
        (output / "visualization-accessibility.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
