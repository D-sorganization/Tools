"""DPI-isolated Launch Monitor Analytics noninitial-state geometry probe.

Drives the registered PyQt ``launch_monitor_analytics`` tab through its
manifest states (``result``, ``error``, ``loading``, ``empty``) inside the
full 1440x900 main window and records, per state, whether the registered
primary visual still intersects the first viewport (#4433).

Preconditions:
    Runs as a script in a fresh process with ``QT_QPA_PLATFORM=offscreen``
    and ``QT_SCALE_FACTOR`` already set, because Qt fixes the scale factor
    when the ``QApplication`` is created.
Postconditions:
    Writes ``manifest.json`` into ``--output`` with one record per state in
    the order the states were driven. No modal dialog is ever shown: the
    tab's own file dialog and message box are replaced by recorders, so the
    production ``import_dialog`` path runs unattended.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import QApplication
from pyqt_visualization_tab_probe import (
    MemorySettings,
    _install_evidence_font,
    _isolate_render_environment,
    _rect,
)

import rate_of_closure.ui.pyqt6.launch_monitor_analytics_tab as analytics_module
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

TAB_ID = "launch_monitor_analytics"
IMPORTED_SOURCE = "imported-launch-monitor.csv"
MALFORMED_SOURCE = "malformed-launch-monitor.csv"


class _DialogRecorder:
    """Unattended stand-in for the tab's file dialog and message box."""

    next_path = ""
    messages: list[str] = []

    @classmethod
    def getOpenFileName(  # noqa: N802
        cls, *_args: object, **_kwargs: object
    ) -> tuple[str, str]:
        """Return the scripted import path as the user's selection."""
        return cls.next_path, ""

    @classmethod
    def critical(cls, _parent: object, _title: str, text: str) -> None:
        """Record the error message the tab would have shown."""
        cls.messages.append(text)


def _write_import_files(output: Path) -> tuple[Path, Path]:
    """Write one valid and one fail-closed launch-monitor import."""
    rows = 40
    valid = pd.DataFrame(
        {
            "club_speed": [40.0 + 0.2 * index for index in range(rows)],
            "attack_angle": [-3.0 + 0.1 * (index % 11) for index in range(rows)],
            "ball_speed": [58.0 + 0.3 * index for index in range(rows)],
            "monitor_vendor": [
                "TrackMan" if index % 2 else "Foresight" for index in range(rows)
            ],
        }
    )
    valid_path = output / IMPORTED_SOURCE
    valid.to_csv(valid_path, index=False)
    malformed_path = output / MALFORMED_SOURCE
    # One numeric column cannot form a relationship, so set_frame fails closed.
    malformed_path.write_text("ball_speed\n60\n61\n62\n", encoding="utf-8")
    return valid_path, malformed_path


def _measure(
    window: RateOfClosureMainWindow,
    tab: LaunchMonitorAnalyticsTab,
    locator: str,
    state: str,
    semantics: str,
) -> dict[str, object]:
    """Record first-viewport geometry of the registered visual for one state."""
    QApplication.processEvents()
    visual = resolve_visual_widget(tab, locator)
    tab_rect = mapped_rect(tab, window)
    tab_bar_rect = mapped_rect(window._tabs.tabBar(), window)
    return {
        "state": state,
        "manifest_semantics": semantics,
        "source_name": tab.source_name,
        "retained_rows": len(tab.frame),
        "result_rows": tab.result_table.rowCount(),
        "has_result": tab.last_result is not None,
        "messages": list(_DialogRecorder.messages),
        "visual_visible": visual.isVisible(),
        "visual_rect": _rect(mapped_rect(visual, tab)),
        "visible_intersection": _rect(visible_intersection(visual, tab)),
        "tab_rect": _rect(tab_rect),
        "tab_bar_overlap": _rect(tab_rect.intersected(tab_bar_rect)),
        "interactive_overlaps": list(interactive_overlaps(tab)),
    }


def _drive_states(
    window: RateOfClosureMainWindow,
    tab: LaunchMonitorAnalyticsTab,
    locator: str,
    states: dict[str, str],
    output: Path,
) -> list[dict[str, object]]:
    """Drive result, error, loading, and empty through production handlers."""
    valid_path, malformed_path = _write_import_files(output)
    # The probe process owns this module; no other caller observes the swap.
    vars(analytics_module).update(
        QFileDialog=_DialogRecorder, QMessageBox=_DialogRecorder
    )
    records: list[dict[str, object]] = []

    tab.run_button.click()
    records.append(_measure(window, tab, locator, "result", states["result"]))

    _DialogRecorder.next_path = str(malformed_path)
    tab.import_button.click()
    records.append(_measure(window, tab, locator, "error", states["error"]))

    original_read = analytics_module.read_launch_monitor_frame

    def pending_read(path: Path) -> pd.DataFrame:
        # The PyQt read is synchronous: measure while it is still pending.
        records.append(_measure(window, tab, locator, "loading", states["loading"]))
        return original_read(path)

    analytics_module.read_launch_monitor_frame = pending_read
    _DialogRecorder.next_path = str(valid_path)
    try:
        tab.import_button.click()
    finally:
        analytics_module.read_launch_monitor_frame = original_read
    if tab.source_name != IMPORTED_SOURCE:
        raise RuntimeError("the valid launch-monitor import did not complete")

    tab.demo_button.click()
    records.append(_measure(window, tab, locator, "empty", states["empty"]))
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    render_environment = _isolate_render_environment(args.output)
    application = QApplication.instance()
    if not isinstance(application, QApplication):
        application = QApplication([])
    font = _install_evidence_font(application)
    render_environment["font_family"] = str(font["font_family"])
    window = RateOfClosureMainWindow(navigation_settings=MemorySettings())
    window.resize(1440, 900)
    window.show()
    QApplication.processEvents()
    entry = next(
        item
        for item in load_visualization_tab_manifest().for_surface("pyqt")
        if item.tab_id == TAB_ID
    )
    index = window.primary_tab_ids().index(TAB_ID)
    window._tabs.setCurrentIndex(index)
    QApplication.processEvents()
    tab = window._tabs.widget(index)
    if not isinstance(tab, LaunchMonitorAnalyticsTab):
        raise TypeError("launch-monitor analytics tab has unexpected type")
    records = _drive_states(
        window, tab, entry.primary_visual_locator, dict(entry.states), args.output
    )
    pixmap = window.grab()
    document = {
        "artifact_policy": "diagnostic-only-not-approved-golden",
        "tab_id": TAB_ID,
        "locator": entry.primary_visual_locator,
        "minimum_visible_height_px": entry.minimum_visible_height_px,
        "requested_scale": args.scale,
        "device_pixel_ratio": pixmap.devicePixelRatio(),
        "logical_window_size": [window.width(), window.height()],
        "render_environment": render_environment,
        "states": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
