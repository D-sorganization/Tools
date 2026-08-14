"""DPI-isolated full-workspace visualization-tab geometry probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import matplotlib
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QFont, QFontDatabase, QFontMetrics
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget

from rate_of_closure.ui.pyqt6.launch_monitor_analytics_tab import (
    LaunchMonitorAnalyticsTab,
)
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow
from rate_of_closure.ui.pyqt6.visualization_tab_audit import (
    interactive_overlaps,
    mapped_rect,
    resolve_visual_widget,
    visible_intersection,
)
from rate_of_closure.visualization_performance_manifest import (
    SurfacePerformanceBudget,
    load_visualization_performance_manifest,
)
from rate_of_closure.visualization_tab_manifest import (
    load_visualization_tab_manifest,
)


def _install_evidence_font(application: QApplication) -> dict[str, object]:
    font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    families = QFontDatabase.applicationFontFamilies(font_id)
    if font_id < 0 or not families:
        raise RuntimeError("could not load the bundled all-tab evidence font")
    font = QFont(families[0])
    metrics = QFontMetrics(font)
    if not all(metrics.inFontUcs4(ord(char)) for char in "Tabs 0123 m/s"):
        raise RuntimeError("the all-tab evidence font lacks required ASCII glyphs")
    application.setFont(font)
    return {"font_family": families[0], "font_ascii_supported": True}


class MemorySettings:
    """Minimal isolated navigation settings for the rendered probe."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, default_value: object = None) -> object:
        """Return one in-memory setting."""
        return self.values.get(key, default_value)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        """Store one in-memory setting."""
        self.values[key] = value


def _rect(rect: QRect) -> list[int]:
    return [rect.x(), rect.y(), rect.width(), rect.height()]


def _rect_shift(left: QRect, right: QRect) -> int:
    return max(
        abs(left.x() - right.x()),
        abs(left.y() - right.y()),
        abs(left.width() - right.width()),
        abs(left.height() - right.height()),
    )


def _stable_visual_rect(
    tab: QWidget,
    locator: str,
    budget_ms: int,
    stable_frames: int,
    tolerance_px: int,
) -> tuple[QRect, float, int]:
    started = time.perf_counter()
    visual = resolve_visual_widget(tab, locator)
    previous = mapped_rect(visual, tab)
    stable = 0
    max_step = 0
    while (time.perf_counter() - started) * 1000 <= budget_ms:
        QTest.qWait(10)
        QApplication.processEvents()
        current = mapped_rect(resolve_visual_widget(tab, locator), tab)
        step = _rect_shift(previous, current)
        max_step = max(max_step, step)
        stable = stable + 1 if step <= tolerance_px else 0
        previous = current
        if stable >= stable_frames:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return current, elapsed_ms, max_step
    raise RuntimeError(f"visual rectangle did not settle within {budget_ms} ms")


def _semantic_text(widget: QWidget) -> str:
    """Read a semantic landmark without treating an empty container as evidence."""
    for accessor in ("text", "toPlainText"):
        value = getattr(widget, accessor, None)
        if callable(value):
            text = str(value()).strip()
            if text:
                return text
    return widget.accessibleName().strip()


def _audit_tab(
    window: RateOfClosureMainWindow,
    tab_id: str,
    locator: str,
    landmark_kind: str,
    minimum_visible_height_px: int,
    minimum_visible_width_px: int,
    workload: str,
    budget: SurfacePerformanceBudget,
    output: Path,
) -> dict[str, object]:
    index = window.primary_tab_ids().index(tab_id)
    opened_at = time.perf_counter()
    window._tabs.setCurrentIndex(index)
    QApplication.processEvents()
    tab = window._tabs.widget(index)
    if not isinstance(tab, QWidget):
        raise TypeError(f"registered tab is not QWidget: {tab_id}")
    _, open_settle_ms, max_open_step = _stable_visual_rect(
        tab,
        locator,
        budget.tab_open_budget_ms,
        budget.stable_frame_count,
        budget.stability_tolerance_px,
    )
    tab_open_ms = (time.perf_counter() - opened_at) * 1000
    visual = resolve_visual_widget(tab, locator)
    tab_rect = mapped_rect(tab, window)
    visual_rect = mapped_rect(visual, tab)
    QTest.qWait(100)
    QApplication.processEvents()
    quiet_rect = mapped_rect(resolve_visual_widget(tab, locator), tab)
    post_settle_shift = _rect_shift(visual_rect, quiet_rect)
    original_size = window.size()
    resize_started = time.perf_counter()
    window.resize(max(1, original_size.width() - 8), original_size.height())
    _, shrunk_ms, shrunk_step = _stable_visual_rect(
        tab,
        locator,
        budget.resize_settle_budget_ms,
        budget.stable_frame_count,
        budget.stability_tolerance_px,
    )
    window.resize(original_size)
    _, restored_ms, restored_step = _stable_visual_rect(
        tab,
        locator,
        budget.resize_settle_budget_ms,
        budget.stable_frame_count,
        budget.stability_tolerance_px,
    )
    resize_settle_ms = (time.perf_counter() - resize_started) * 1000
    visual = resolve_visual_widget(tab, locator)
    visual_rect = mapped_rect(visual, tab)
    intersection = visible_intersection(visual, tab)
    tab_bar_rect = mapped_rect(window._tabs.tabBar(), window)
    screenshot = output / f"tab-{tab_id}.png"
    if not window.grab().save(str(screenshot), "PNG"):
        raise RuntimeError(f"could not save {tab_id} diagnostic")
    selected_screenshot = ""
    if tab_id == "launch_monitor_analytics":
        if not isinstance(tab, LaunchMonitorAnalyticsTab):
            raise TypeError("launch-monitor analytics tab has unexpected type")
        preview = tab.preview
        QTest.keyClick(preview, Qt.Key.Key_End)
        QApplication.processEvents()
        selected = output / "tab-launch_monitor_analytics-selected.png"
        if not window.grab().save(str(selected), "PNG"):
            raise RuntimeError("could not save selected linked-scatter diagnostic")
        selected_screenshot = selected.name
        QTest.keyClick(preview, Qt.Key.Key_Escape)
    return {
        "tab_id": tab_id,
        "workload": workload,
        "tab_open_ms": round(tab_open_ms, 3),
        "open_settle_ms": round(open_settle_ms, 3),
        "resize_settle_ms": round(resize_settle_ms, 3),
        "resize_shrink_ms": round(shrunk_ms, 3),
        "resize_restore_ms": round(restored_ms, 3),
        "max_open_step_px": max_open_step,
        "max_resize_step_px": max(shrunk_step, restored_step),
        "post_settle_shift_px": post_settle_shift,
        "locator": locator,
        "landmark_kind": landmark_kind,
        "minimum_visible_height_px": minimum_visible_height_px,
        "minimum_visible_width_px": minimum_visible_width_px,
        "tab_rect": _rect(tab_rect),
        "visual_rect": _rect(visual_rect),
        "visible_intersection": _rect(intersection),
        "tab_bar_overlap": _rect(tab_rect.intersected(tab_bar_rect)),
        "visual_visible": visual.isVisible(),
        "screenshot": screenshot.name,
        "screenshot_bytes": screenshot.stat().st_size,
        "selected_screenshot": selected_screenshot,
        "visual_class": type(visual).__name__,
        "semantic_text": _semantic_text(visual)
        if landmark_kind == "semantic-content"
        else "",
        "interactive_overlaps": list(interactive_overlaps(tab)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    application = QApplication.instance() or QApplication([])
    if application is None:
        raise RuntimeError("could not initialize QApplication")
    font = _install_evidence_font(application)
    window = RateOfClosureMainWindow(navigation_settings=MemorySettings())
    window.resize(1440, 900)
    window.show()
    QApplication.processEvents()
    manifest = load_visualization_tab_manifest()
    performance = load_visualization_performance_manifest()
    entries = manifest.for_surface("pyqt")
    performance_entries = {
        entry.tab_id: entry for entry in performance.for_surface("pyqt")
    }
    performance_budget = performance.surfaces["pyqt"]
    minimum_width = manifest.reference_environments["pyqt"].minimum_visible_width_px
    evidence = [
        _audit_tab(
            window,
            entry.tab_id,
            entry.primary_visual_locator,
            entry.landmark_kind,
            entry.minimum_visible_height_px,
            minimum_width if entry.landmark_kind == "visual" else 1,
            performance_entries[entry.tab_id].workload,
            performance_budget,
            args.output,
        )
        for entry in entries
    ]
    candidate_root_text = os.environ.get("RATE_VISUAL_BASELINE_CANDIDATE_DIR")
    if candidate_root_text and args.scale == 1.0:
        candidate_root = Path(candidate_root_text) / "pyqt"
        candidate_root.mkdir(parents=True, exist_ok=True)
        candidates: list[dict[str, str]] = []
        for entry in evidence:
            tab_id = str(entry["tab_id"])
            file = f"initial-{tab_id}-dpi-1.0.png"
            source = args.output / str(entry["screenshot"])
            target = candidate_root / file
            shutil.copyfile(source, target)
            candidates.append(
                {
                    "tab_id": tab_id,
                    "file": file,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
        candidate_document = {
            "schema_id": "rate-of-closure/visual-baseline-candidates",
            "schema_version": 1,
            "artifact_policy": (
                "candidate-diagnostic-not-approved-until-protected-merge"
            ),
            "source_commit": os.environ.get("GITHUB_SHA", "local-diagnostic"),
            "surface": "pyqt",
            "environment": (
                f"{os.name}-{os.environ.get('QT_QPA_PLATFORM', 'default')}"
                "-qt-dpi-1.0-1440x900"
            ),
            "captures": candidates,
        }
        (candidate_root / "manifest.json").write_text(
            json.dumps(candidate_document, indent=2) + "\n", encoding="utf-8"
        )
    pixmap = window.grab()
    document = {
        "artifact_policy": "diagnostic-only-not-approved-golden",
        "measurement_policy": performance.measurement_policy,
        "requested_scale": args.scale,
        "device_pixel_ratio": pixmap.devicePixelRatio(),
        "logical_window_size": [window.width(), window.height()],
        "font": font,
        "tabs": evidence,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
