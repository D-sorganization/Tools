"""DPI-isolated production Simulation scrub/error evidence probe."""

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

import rate_of_closure.ui.pyqt6.simulation_tab_publication as publication
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow
from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab
from rate_of_closure.ui.pyqt6.visualization_tab_audit import visible_intersection


class MemorySettings:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, default_value: object = None) -> object:
        return self.values.get(key, default_value)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        self.values[key] = value


def _install_font(application: QApplication) -> dict[str, object]:
    font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    families = QFontDatabase.applicationFontFamilies(font_id)
    if font_id < 0 or not families:
        raise RuntimeError("could not load the bundled Simulation evidence font")
    font = QFont(families[0])
    metrics = QFontMetrics(font)
    if not all(metrics.inFontUcs4(ord(character)) for character in "Simulation 0123"):
        raise RuntimeError("Simulation evidence font lacks required ASCII glyphs")
    application.setFont(font)
    return {"font_id": font_id, "font_family": families[0], "ascii": True}


def _capture(
    window: RateOfClosureMainWindow,
    tab: SimulationTab,
    output: Path,
    state: str,
) -> dict[str, object]:
    QApplication.processEvents()
    window_target = output / f"simulation-{state}.png"
    canvas_target = output / f"simulation-{state}-canvas.png"
    canvas = tab.view()._canvas
    if not window.grab().save(str(window_target), "PNG"):
        raise RuntimeError(f"could not capture Simulation {state}")
    if not canvas.grab().save(str(canvas_target), "PNG"):
        raise RuntimeError(f"could not capture Simulation canvas {state}")
    canvas_bytes = canvas_target.read_bytes()
    visible = visible_intersection(canvas, tab)
    visual = QRect(canvas.mapTo(tab, QPoint()), canvas.size())
    controls = tab.findChild(QScrollArea)
    if controls is None:
        raise RuntimeError("Simulation setup scroll area is unavailable")
    control_rect = QRect(controls.mapTo(tab, QPoint()), controls.size())
    status_visible = visible_intersection(tab._run_status, tab)
    run = tab.last_run()
    return {
        "state": state,
        "run_identity": None if run is None else id(run),
        "requested_impact_time_s": None if run is None else run.config.impact_time_s,
        "resolved_impact_time_s": None if run is None else run.impact_time_s,
        "status": tab._run_status.text(),
        "status_visible": [
            status_visible.x(),
            status_visible.y(),
            status_visible.width(),
            status_visible.height(),
        ],
        "controls_scroll_y": controls.verticalScrollBar().value(),
        "visible_visual": [visible.x(), visible.y(), visible.width(), visible.height()],
        "tab_size": [tab.width(), tab.height()],
        "control_overlap": visual.intersects(control_rect),
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
    font = _install_font(application)
    window = RateOfClosureMainWindow(navigation_settings=MemorySettings())
    window.resize(1440, 900)
    window.show()
    window._tabs.setCurrentIndex(window.primary_tab_ids().index("simulation"))
    tab = window._simulation_tab
    tab._display_tabs.setCurrentWidget(tab.view())
    accepted = tab.last_run()
    if accepted is None:
        raise RuntimeError("Simulation probe has no accepted startup run")
    tab._scrub_slider.setFocus()
    prior_value = tab._scrub_slider.value()
    QTest.keyClick(tab._scrub_slider, Qt.Key.Key_Right)
    if tab.last_run() is accepted or tab._scrub_slider.value() != prior_value + 1:
        raise RuntimeError("Simulation keyboard scrub did not publish a new run")
    tab._auto_tau_button.click()
    accepted = tab.last_run()
    if accepted is None or accepted.config.impact_time_s is not None:
        raise RuntimeError("Simulation Auto tau did not publish null request authority")
    states = [_capture(window, tab, args.output, "result-auto")]

    original = publication.run_simulation

    def fail(_config: object) -> None:
        raise OSError("diagnostic simulation authority unavailable")

    publication.run_simulation = fail
    try:
        if tab.run_now() is not None or tab.last_run() is not accepted:
            raise RuntimeError(
                "failed Simulation recompute discarded accepted evidence"
            )
        tab._controls_scroll.ensureWidgetVisible(tab._run_status, 0, 16)
        QApplication.processEvents()
        states.append(_capture(window, tab, args.output, "error-prior"))
    finally:
        publication.run_simulation = original
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_policy": "diagnostic PNG; semantic tests are authority",
                "requested_scale": args.scale,
                "window": [window.width(), window.height()],
                "font": font,
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
