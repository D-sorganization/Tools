"""Full primary-tab visual geometry at 100 and 150 percent DPI."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from rate_of_closure.ui.pyqt6.morris_tab import MorrisScreeningTab
from rate_of_closure.ui.pyqt6.variation_tab import VariationTab
from rate_of_closure.visualization_performance_manifest import (
    load_visualization_performance_manifest,
)

pytest.importorskip("PyQt6")
pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_initial_morris_result_selectors_are_visible_and_fail_closed(qtbot) -> None:
    """Do not present a fabricated target or source before a completed report."""

    tab = MorrisScreeningTab(None)
    qtbot.addWidget(tab)
    tab.resize(900, 600)
    tab.show()

    assert tab._target_combo.accessibleName() == "Morris result target"
    assert tab._source_combo.accessibleName() == "Morris result source"
    assert tab._target_combo.isVisible()
    assert tab._source_combo.isVisible()
    assert tab._target_combo.count() == 0
    assert tab._source_combo.count() == 0
    assert tab._target_detail.text() == "Run a study to choose a target."


def test_initial_variation_execution_policy_is_visible_and_bounded(qtbot) -> None:
    """Expose the governed computation policy without changing the saved plan."""

    tab = VariationTab(None)
    qtbot.addWidget(tab)
    tab.resize(1000, 700)
    tab.show()

    assert tab._analysis_combo.accessibleName() == (
        "Variation analysis execution policy"
    )
    assert tab._analysis_label.text() == "Analysis Policy"
    assert tab._analysis_label.buddy() is tab._analysis_combo
    assert tab._analysis_label.isVisible()
    assert tab._analysis_combo.isVisible()
    assert [tab._analysis_combo.itemData(index) for index in range(3)] == [
        "all_together",
        "individual",
        "both",
    ]
    assert tab._analysis_combo.currentData() == "both"


def _probe(
    output_root: Path, scale: float, candidate_root: Path | None = None
) -> dict[str, Any]:
    output = output_root / f"scale-{scale:g}"
    stale_settings = output / "qsettings" / "stale.ini"
    stale_settings.parent.mkdir(parents=True, exist_ok=True)
    stale_settings.write_text("[ImpactScene]\nvisible_layers_v1=[]\n", encoding="utf-8")
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "QT_SCALE_FACTOR": str(scale),
            "MPLBACKEND": "qtagg",
            "PYTHONPATH": os.pathsep.join((str(repository / "src"), str(repository))),
        }
    )
    if candidate_root is not None:
        environment["RATE_VISUAL_BASELINE_CANDIDATE_DIR"] = str(candidate_root)
        environment.setdefault(
            "RATE_VISUAL_BASELINE_SOURCE_COMMIT",
            "a" * 40,
        )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("pyqt_visualization_tab_probe.py")),
            "--output",
            str(output),
            "--scale",
            str(scale),
        ],
        check=True,
        env=environment,
        timeout=120,
    )
    assert not stale_settings.exists()
    return cast(
        dict[str, Any],
        json.loads((output / "manifest.json").read_text(encoding="utf-8")),
    )


@pytest.mark.timeout(300)
def test_all_primary_tab_visuals_are_visible_and_nonoverlapping_at_both_dpis(
    tmp_path: Path,
) -> None:
    output = Path(os.environ.get("RATE_PYQT_EVIDENCE_DIR", str(tmp_path)))
    candidate_root = Path(
        os.environ.get(
            "RATE_VISUAL_BASELINE_CANDIDATE_DIR", str(tmp_path / "candidates")
        )
    )
    manifests = [
        _probe(output, 1.0, candidate_root),
        _probe(output, 1.5),
    ]
    budget = load_visualization_performance_manifest().surfaces["pyqt"]
    for scale, manifest in zip((1.0, 1.5), manifests, strict=True):
        assert manifest["artifact_policy"] == "diagnostic-only-not-approved-golden"
        assert manifest["measurement_policy"] == (
            "protected-diagnostic-not-user-hardware-qualification"
        )
        assert manifest["requested_scale"] == scale
        assert manifest["device_pixel_ratio"] == pytest.approx(scale)
        assert manifest["logical_window_size"] == [1440, 900]
        assert manifest["font"] == {
            "font_family": "DejaVu Sans",
            "font_ascii_supported": True,
        }
        render_environment = manifest["render_environment"]
        assert render_environment["settings_policy"] == "isolated-ini-user-scope"
        assert render_environment["qt_version"]
        assert render_environment["pyqt_version"]
        assert render_environment["matplotlib_version"]
        assert render_environment["font_family"] == "DejaVu Sans"
        assert len(manifest["tabs"]) == 10
        for tab in manifest["tabs"]:
            assert tab["workload"] == "initial-production-state", tab["tab_id"]
            assert tab["tab_open_ms"] <= budget.tab_open_budget_ms, tab["tab_id"]
            assert tab["resize_settle_ms"] <= budget.resize_settle_budget_ms, tab[
                "tab_id"
            ]
            assert tab["post_settle_shift_px"] <= (budget.max_post_settle_shift_px), (
                tab["tab_id"]
            )
            assert tab["max_open_step_px"] >= 0, tab["tab_id"]
            assert tab["max_resize_step_px"] >= 0, tab["tab_id"]
            assert tab["screenshot_bytes"] > 10_000, tab["tab_id"]
            assert tab["visual_visible"] is True, tab["tab_id"]
            if tab["landmark_kind"] == "visual":
                assert "Canvas" in tab["visual_class"], tab["tab_id"]
            else:
                assert tab["semantic_text"].strip(), tab["tab_id"]
            assert tab["tab_rect"][2] > 0 and tab["tab_rect"][3] > 0
            assert tab["visible_intersection"][2] >= tab["minimum_visible_width_px"], (
                tab["tab_id"]
            )
            assert tab["visible_intersection"][3] >= tab["minimum_visible_height_px"], (
                tab["tab_id"]
            )
            assert tab["tab_bar_overlap"][2:] == [0, 0], tab["tab_id"]
            assert tab["interactive_overlaps"] == [], tab["tab_id"]

    candidate_path = candidate_root / "pyqt" / "manifest.json"
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidates["schema_id"] == ("rate-of-closure/visual-baseline-candidates")
    assert candidates["schema_version"] == 1
    assert candidates["artifact_policy"] == (
        "candidate-diagnostic-not-approved-until-protected-merge"
    )
    assert candidates["source_commit"] == os.environ.get(
        "RATE_VISUAL_BASELINE_SOURCE_COMMIT", "a" * 40
    )
    assert candidates["surface"] == "pyqt"
    environment = candidates["environment"]
    assert environment.startswith("posix-offscreen-qt-") or environment.startswith(
        "nt-offscreen-qt-"
    )
    for token in ("-pyqt-", "-matplotlib-", "-font-dejavu-sans-dpi-1.0-1440x900"):
        assert token in environment
    captures = candidates["captures"]
    assert {entry["tab_id"] for entry in captures} == {
        tab["tab_id"] for tab in manifests[0]["tabs"]
    }
    for entry in captures:
        image = candidate_path.parent / entry["file"]
        assert image.stat().st_size > 10_000
        assert hashlib.sha256(image.read_bytes()).hexdigest() == entry["sha256"]
