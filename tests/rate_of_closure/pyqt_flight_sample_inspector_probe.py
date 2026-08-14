"""DPI-isolated production Flight inspector evidence probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QFont, QFontDatabase, QFontMetrics
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QScrollArea

import rate_of_closure.ui.pyqt6.flight_explorer_run as run_module
from rate_of_closure.ui.pyqt6.flight_explorer_tab import FlightExplorerTab
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow
from rate_of_closure.ui.pyqt6.visualization_tab_audit import visible_intersection


class MemorySettings:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, default_value: object = None) -> object:
        return self.values.get(key, default_value)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        self.values[key] = value


def _install_evidence_font(application: QApplication) -> dict[str, object]:
    font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    if font_id < 0:
        raise RuntimeError("could not load the bundled Flight evidence font")
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        raise RuntimeError("the bundled Flight evidence font has no family")
    font = QFont(families[0])
    metrics = QFontMetrics(font)
    sanity_text = "Flight 0123 m/s"
    if not all(metrics.inFontUcs4(ord(character)) for character in sanity_text):
        raise RuntimeError("the Flight evidence font lacks required ASCII glyphs")
    application.setFont(font)
    return {
        "font_id": font_id,
        "font_family": families[0],
        "font_ascii_supported": True,
    }


def _capture(
    window: RateOfClosureMainWindow,
    tab: FlightExplorerTab,
    output: Path,
    state: str,
) -> dict[str, object]:
    QApplication.processEvents()
    window_target = output / f"flight-{state}.png"
    canvas_target = output / f"flight-{state}-canvas.png"
    if not window.grab().save(str(window_target), "PNG"):
        raise RuntimeError(f"could not capture Flight {state}")
    canvas = tab.flight_view()._canvas
    if not canvas.grab().save(str(canvas_target), "PNG"):
        raise RuntimeError(f"could not capture Flight canvas {state}")
    canvas_bytes = canvas_target.read_bytes()
    visible = visible_intersection(canvas, tab)
    visual = QRect(canvas.mapTo(tab, QPoint()), canvas.size())
    controls = tab.findChild(QScrollArea)
    if controls is None:
        raise RuntimeError("Flight setup scroll area is unavailable")
    control_rect = QRect(controls.mapTo(tab, QPoint()), controls.size())
    accepted = tab.accepted_study()
    return {
        "state": state,
        "accepted_generation": None if accepted is None else accepted.generation,
        "accepted_context": None if accepted is None else accepted.context.label(),
        "raw_sample_count": None if accepted is None else accepted.plan.raw_count,
        "selected_raw_index": tab.flight_view().selected_raw_index(),
        "playback_time_s": tab._flight_panel.controls.current_time_s(),
        "status": tab._sample_status.text(),
        "error": tab._error_status.text(),
        "context": tab._context_status.text(),
        "visible_visual": [
            visible.x(),
            visible.y(),
            visible.width(),
            visible.height(),
        ],
        "tab_size": [tab.width(), tab.height()],
        "control_overlap": visual.intersects(control_rect),
        "canvas_has_focus": QApplication.focusWidget() is canvas,
        "device_pixel_ratio": canvas.devicePixelRatioF(),
        "window_screenshot": window_target.name,
        "window_bytes": window_target.stat().st_size,
        "canvas_screenshot": canvas_target.name,
        "canvas_bytes": len(canvas_bytes),
        "canvas_sha256": hashlib.sha256(canvas_bytes).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    font_evidence = _install_evidence_font(application)
    window = RateOfClosureMainWindow(navigation_settings=MemorySettings())
    window.resize(1440, 900)
    window.show()
    window._tabs.setCurrentIndex(window.primary_tab_ids().index("flight_explorer"))
    tab = window._flight_explorer_tab
    if tab.run_now() is None:
        raise RuntimeError("Flight probe could not produce an accepted flight")
    canvas = tab.flight_view()._canvas
    canvas.setFocus()
    QTest.keyClick(canvas, Qt.Key.Key_Home)
    accepted = tab.accepted_study()
    states = [_capture(window, tab, args.output, "selected-result")]

    original = run_module.explore_with_optional_wind

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("diagnostic flight authority unavailable")

    run_module.explore_with_optional_wind = fail
    tab._direct_spins["launch_angle_deg"].stepUp()
    if tab.run_now() is not None or tab.accepted_study() is not accepted:
        raise RuntimeError("failed flight recompute discarded accepted evidence")
    states.append(_capture(window, tab, args.output, "error-prior"))
    run_module.explore_with_optional_wind = original
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_policy": "diagnostic PNG; semantic tests are authority",
                "requested_scale": args.scale,
                "requested_window": [1440, 900],
                "actual_window": [window.width(), window.height()],
                "font": font_evidence,
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
