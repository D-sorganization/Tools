"""DPI-isolated full-workspace visualization-tab geometry probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PyQt6.QtCore import QRect, Qt
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
from rate_of_closure.visualization_tab_manifest import (
    load_visualization_tab_manifest,
)


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
    output: Path,
) -> dict[str, object]:
    index = window.primary_tab_ids().index(tab_id)
    window._tabs.setCurrentIndex(index)
    QApplication.processEvents()
    tab = window._tabs.widget(index)
    if not isinstance(tab, QWidget):
        raise TypeError(f"registered tab is not QWidget: {tab_id}")
    visual = resolve_visual_widget(tab, locator)
    tab_rect = mapped_rect(tab, window)
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
    window = RateOfClosureMainWindow(navigation_settings=MemorySettings())
    window.resize(1440, 900)
    window.show()
    QApplication.processEvents()
    manifest = load_visualization_tab_manifest()
    entries = manifest.for_surface("pyqt")
    minimum_width = manifest.reference_environments["pyqt"].minimum_visible_width_px
    evidence = [
        _audit_tab(
            window,
            entry.tab_id,
            entry.primary_visual_locator,
            entry.landmark_kind,
            entry.minimum_visible_height_px,
            minimum_width if entry.landmark_kind == "visual" else 1,
            args.output,
        )
        for entry in entries
    ]
    pixmap = window.grab()
    document = {
        "artifact_policy": "diagnostic-only-not-approved-golden",
        "requested_scale": args.scale,
        "device_pixel_ratio": pixmap.devicePixelRatio(),
        "logical_window_size": [window.width(), window.height()],
        "tabs": evidence,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
