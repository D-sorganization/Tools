"""DPI-isolated production Plots exact-evidence probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QFont, QFontDatabase, QFontMetrics
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

import rate_of_closure.ui.pyqt6.plots_tab as plots_module
from rate_of_closure.plotting import PlotData
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow
from rate_of_closure.ui.pyqt6.plots_tab import PlotsTab
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
        raise RuntimeError("could not load the bundled Plots evidence font")
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        raise RuntimeError("the bundled Plots evidence font has no family")
    font = QFont(families[0])
    metrics = QFontMetrics(font)
    sanity_text = "Plots 0123 m/s"
    if not all(metrics.inFontUcs4(ord(character)) for character in sanity_text):
        raise RuntimeError("the Plots evidence font lacks required ASCII glyphs")
    application.setFont(font)
    return {
        "font_id": font_id,
        "font_family": families[0],
        "font_ascii_supported": True,
    }


def _data_digest(data: PlotData | None) -> str | None:
    if data is None:
        return None
    digest = hashlib.sha256(data.x.tobytes())
    for label, values in data.series.items():
        digest.update(label.encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _capture(
    window: RateOfClosureMainWindow,
    tab: PlotsTab,
    output: Path,
    state: str,
) -> dict[str, object]:
    QApplication.processEvents()
    pane = tab.plot_panes()[0]
    canvas = pane.canvas()
    window_target = output / f"plots-{state}.png"
    canvas_target = output / f"plots-{state}-canvas.png"
    if not window.grab().save(str(window_target), "PNG"):
        raise RuntimeError(f"could not capture Plots {state}")
    if not canvas.grab().save(str(canvas_target), "PNG"):
        raise RuntimeError(f"could not capture Plots canvas {state}")
    canvas_bytes = canvas_target.read_bytes()
    visible = visible_intersection(canvas, tab)
    visual = QRect(canvas.mapTo(tab, QPoint()), canvas.size())
    controls = tab._plot_list.parentWidget()
    if controls is None:
        raise RuntimeError("Plots setup controls are unavailable")
    control_rect = QRect(controls.mapTo(tab, QPoint()), controls.size())
    selection = pane.selected_evidence()
    return {
        "state": state,
        "data_digest": _data_digest(tab.current_data()),
        "selected_evidence": repr(selection),
        "inspection_status": pane.inspection_status(),
        "error": tab._status.text(),
        "visible_visual": [visible.x(), visible.y(), visible.width(), visible.height()],
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
    window._tabs.setCurrentIndex(window.primary_tab_ids().index("plots"))
    tab = window._plots_tab
    tab.refresh()
    pane = tab.plot_panes()[0]
    canvas = pane.canvas()
    canvas.setFocus()
    QTest.keyClick(canvas, Qt.Key.Key_Home)
    accepted_data = tab.current_data()
    states = [_capture(window, tab, args.output, "selected-result")]

    original = plots_module.compute_plot_data

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("diagnostic plot authority unavailable")

    plots_module.compute_plot_data = fail
    run = tab.reference_run()
    if run is None:
        raise RuntimeError("Plots probe has no accepted reference run")
    tab.set_run(run)
    if tab.current_data() is not accepted_data:
        raise RuntimeError("failed plot recompute discarded accepted evidence")
    states.append(_capture(window, tab, args.output, "error-prior"))
    plots_module.compute_plot_data = original
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
