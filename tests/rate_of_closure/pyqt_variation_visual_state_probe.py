"""DPI-isolated Variation lifecycle visual evidence probe."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PyQt6.QtCore import QEventLoop, QPoint, QRect, QTimer
from PyQt6.QtWidgets import QApplication

from rate_of_closure.ui.pyqt6 import variation_worker
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow
from rate_of_closure.ui.pyqt6.variation_tab import VariationTab
from rate_of_closure.ui.pyqt6.visualization_tab_audit import visible_intersection
from shared.python.swing_sim.variation import (
    CATEGORY_LAUNCH,
    CancelledError,
    NoiseSpec,
    VariationPlan,
)


def _plan(runs: int) -> VariationPlan:
    return VariationPlan(
        mode="launch",
        noise=(NoiseSpec(f"{CATEGORY_LAUNCH}.ball_speed_mph", scale=1.0),),
        n_runs=runs,
        seed=4433,
    )


class MemorySettings:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, default_value: object = None) -> object:
        return self.values.get(key, default_value)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        self.values[key] = value


def _capture(
    window: RateOfClosureMainWindow,
    tab: VariationTab,
    output: Path,
    name: str,
) -> dict[str, object]:
    QApplication.processEvents()
    target = output / f"variation-{name}.png"
    if not window.grab().save(str(target), "PNG"):
        raise RuntimeError(f"could not capture {name}")
    strip = tab._visual_frame._state_strip.geometry()
    content = tab._visual_frame.content.geometry()
    strip_visible = tab._visual_frame._state_strip.isVisible()
    strip_in_tab = QRect(
        tab._visual_frame._state_strip.mapTo(tab, QPoint()),
        tab._visual_frame._state_strip.size(),
    )
    run_in_tab = QRect(tab._run_button.mapTo(tab, QPoint()), tab._run_button.size())
    visible = visible_intersection(tab._visual_frame.content, tab)
    return {
        "state": name,
        "phase": tab._visual_frame.property("visualPhase"),
        "origin": tab._visual_frame.property("visualOrigin"),
        "strip": [strip.x(), strip.y(), strip.width(), strip.height()],
        "content": [content.x(), content.y(), content.width(), content.height()],
        "visible_content": [
            visible.x(),
            visible.y(),
            visible.width(),
            visible.height(),
        ],
        "window_size": [window.width(), window.height()],
        "tab_size": [tab.width(), tab.height()],
        "strip_visible": strip_visible,
        "overlap": strip.intersects(content) if strip_visible else False,
        "control_overlap": (
            strip_in_tab.intersects(run_in_tab) if strip_visible else False
        ),
        "status": tab._status.text(),
        "screenshot": target.name,
        "bytes": target.stat().st_size,
    }


def _wait(tab: VariationTab) -> None:
    worker = tab._worker
    if worker is None:
        raise RuntimeError("variation worker was not created")
    if not worker.isFinished():
        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        QTimer.singleShot(60_000, loop.quit)
        loop.exec()
    if not worker.isFinished():
        raise TimeoutError("variation worker did not finish")
    QApplication.processEvents()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    window = RateOfClosureMainWindow(navigation_settings=MemorySettings())
    window.resize(1440, 900)
    window.show()
    index = window.primary_tab_ids().index("variation")
    window._tabs.setCurrentIndex(index)
    tab = window._variation_tab
    tab._sens_check.setChecked(False)
    evidence = [_capture(window, tab, args.output, "empty")]

    original_run = variation_worker.run_variation

    def blocking_run(*_args: object, **kwargs: object) -> None:
        cancel_event = kwargs["cancel_event"]
        deadline = time.monotonic() + 10.0
        while not cancel_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.005)
        raise CancelledError

    variation_worker.run_variation = blocking_run
    tab.load_plan(_plan(4))
    tab._on_run()
    evidence.append(_capture(window, tab, args.output, "loading-no-prior"))
    tab._on_cancel()
    _wait(tab)
    variation_worker.run_variation = original_run

    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("diagnostic engine failure")

    variation_worker.run_variation = fail_run
    tab.load_plan(_plan(4))
    tab._on_run()
    _wait(tab)
    evidence.append(_capture(window, tab, args.output, "error-empty"))
    variation_worker.run_variation = original_run

    tab.load_plan(_plan(4))
    tab._on_run()
    _wait(tab)
    evidence.append(_capture(window, tab, args.output, "result"))
    variation_worker.run_variation = blocking_run
    tab._on_run()
    evidence.append(_capture(window, tab, args.output, "loading-prior"))
    tab._on_cancel()
    _wait(tab)
    variation_worker.run_variation = original_run
    variation_worker.run_variation = fail_run
    tab._on_run()
    _wait(tab)
    evidence.append(_capture(window, tab, args.output, "error-prior"))
    variation_worker.run_variation = original_run

    document = {
        "artifact_policy": "diagnostic PNG; semantic manifest is test authority",
        "requested_scale": args.scale,
        "states": evidence,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
