"""DPI-isolated production Putting inspector evidence probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QScrollArea

import rate_of_closure.ui.pyqt6.putting_tab as putting_tab_module
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow
from rate_of_closure.ui.pyqt6.putting_tab import PuttingTab
from rate_of_closure.ui.pyqt6.visualization_tab_audit import visible_intersection


class MemorySettings:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, default_value: object = None) -> object:
        return self.values.get(key, default_value)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        self.values[key] = value


def _capture(
    window: RateOfClosureMainWindow, tab: PuttingTab, output: Path, state: str
) -> dict[str, object]:
    QApplication.processEvents()
    target = output / f"putting-{state}.png"
    if not window.grab().save(str(target), "PNG"):
        raise RuntimeError(f"could not capture Putting {state}")
    visible = visible_intersection(tab._canvas, tab)
    visual = QRect(tab._canvas.mapTo(tab, QPoint()), tab._canvas.size())
    controls = tab.findChild(QScrollArea)
    if controls is None:
        raise RuntimeError("Putting setup scroll area is unavailable")
    control_rect = QRect(controls.mapTo(tab, QPoint()), controls.size())
    return {
        "state": state,
        "selected_raw_index": tab._plot_view.selected_raw_index(),
        "selected_marker_count": len(tab._plot_view.selected_artists()),
        "status": tab._plot_view.status_text(),
        "error": tab._plot_view.error_text(),
        "context": tab._plot_view.context_text(),
        "visible_visual": [visible.x(), visible.y(), visible.width(), visible.height()],
        "tab_size": [tab.width(), tab.height()],
        "control_overlap": visual.intersects(control_rect),
        "canvas_has_focus": QApplication.focusWidget() is tab._canvas,
        "screenshot": target.name,
        "bytes": target.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    window = RateOfClosureMainWindow(navigation_settings=MemorySettings())
    window.resize(1440, 900)
    window.show()
    window._tabs.setCurrentIndex(window.primary_tab_ids().index("putting"))
    tab = window._putting_tab
    tab._canvas.setFocus()
    QTest.keyClick(tab._canvas, Qt.Key.Key_Home)
    accepted = tab.result()
    states = [_capture(window, tab, args.output, "selected-result")]

    original = putting_tab_module.simulate_putt_on_surface

    def fail(*_args: object, **_kwargs: object) -> None:
        raise ValueError("diagnostic solver authority unavailable")

    putting_tab_module.simulate_putt_on_surface = fail
    tab.green_controls().grade_spin.setValue(1.0)
    if tab.result() is not accepted:
        raise RuntimeError("failed recompute discarded accepted evidence")
    states.append(_capture(window, tab, args.output, "error-prior"))
    putting_tab_module.simulate_putt_on_surface = original
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_policy": "diagnostic PNG; semantic tests are authority",
                "requested_scale": args.scale,
                "requested_window": [1440, 900],
                "actual_window": [window.width(), window.height()],
                "states": states,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
