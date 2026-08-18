"""Subprocess-owned rendered interaction probe for the PyQt variation views.

The caller sets ``QT_SCALE_FACTOR`` before Qt is imported.  PNGs are diagnostic
artifacts; the JSON manifest carries deterministic semantic assertions so CI
does not depend on fragile pixel equality across Qt platform plugins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QWidget,
)

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.plotting import PlotData, PlotSpec
from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.ui.pyqt6.plot_canvas_pane import PlotCanvasPane
from rate_of_closure.ui.pyqt6.variation_visualizations import ArcOverlayView
from rate_of_closure.variation.plot_data import (
    EnsemblePlotDataset,
    build_ensemble_plot_dataset,
)
from rate_of_closure.variation.simulation_adapter import (
    build_simulation_ensemble_request,
    run_simulation_ensemble,
)
from shared.python.swing_sim.variation import (
    CATEGORY_SWING,
    ELLIPSOID_VOLUME,
    NoiseSpec,
    VariationPlan,
)

_YAW = f"{CATEGORY_SWING}.yaw_deg"


def _plot_dataset() -> EnsemblePlotDataset:
    plan = VariationPlan(
        mode="swing",
        noise=(NoiseSpec(_YAW, distribution="uniform", scale=0.2),),
        n_runs=3,
        seed=4142,
    )
    base = SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=30.0),
        club=get_club("Sand Wedge"),
        source_kind="double_pendulum",
        swing_duration_s=0.05,
    )
    result = run_simulation_ensemble(build_simulation_ensemble_request(plan, base))
    return build_ensemble_plot_dataset(result)


def _visible_control_overlaps(root: QWidget) -> list[str]:
    """Return positive-area overlaps between visible interactive controls."""
    kinds = (QAbstractButton, QComboBox, QSpinBox, QDoubleSpinBox)
    controls = [
        widget
        for widget in root.findChildren(QWidget)
        if widget.isVisible() and isinstance(widget, kinds)
    ]
    conflicts: list[str] = []
    for left_index, left in enumerate(controls):
        left_rect = left.rect().translated(left.mapTo(root, left.rect().topLeft()))
        for right in controls[left_index + 1 :]:
            if left.isAncestorOf(right) or right.isAncestorOf(left):
                continue
            right_rect = right.rect().translated(
                right.mapTo(root, right.rect().topLeft())
            )
            intersection = left_rect.intersected(right_rect)
            if intersection.width() > 1 and intersection.height() > 1:
                left_name = left.accessibleName() or left.objectName()
                right_name = right.accessibleName() or right.objectName()
                conflicts.append(
                    f"{left_name or type(left).__name__} <> "
                    f"{right_name or type(right).__name__}"
                )
    return conflicts


def _save_widget(widget: QWidget, path: Path) -> dict[str, object]:
    QApplication.processEvents()
    pixmap = widget.grab()
    if not pixmap.save(str(path), "PNG"):
        raise RuntimeError(f"could not write {path}")
    image = QImage(str(path))
    return {
        "logical_size": [widget.width(), widget.height()],
        "pixel_size": [image.width(), image.height()],
        "device_pixel_ratio": pixmap.devicePixelRatio(),
        "bytes": path.stat().st_size,
        "overlaps": _visible_control_overlaps(widget),
    }


def _exercise_arc(output: Path) -> dict[str, object]:
    view = ArcOverlayView()
    view.resize(1500, 980)
    view.set_plot_dataset(_plot_dataset())
    view.show()
    QApplication.processEvents()
    view._metric_combo.setCurrentIndex(view._metric_combo.findData(ELLIPSOID_VOLUME))
    view._ellipsoid_mesh.setFocus()
    QTest.keyClick(view._ellipsoid_mesh, Qt.Key.Key_Space)  # type: ignore[call-overload]
    if not view._ellipsoid_mesh.isChecked():
        raise AssertionError("Space did not toggle the confidence-ellipsoid control")
    view._canvas.axes.view_init(elev=22.0, azim=-37.0)
    view._canvas.draw()
    path = output / "variation-arc.png"
    evidence = _save_widget(view, path)
    evidence.update(
        {
            "accessible_control": view._ellipsoid_mesh.accessibleName(),
            "camera": {
                "azimuth_deg": view._canvas.axes.azim,
                "elevation_deg": view._canvas.axes.elev,
            },
            "ellipsoid_toggle": view._ellipsoid_mesh.isChecked(),
            "metric": view._metric_combo.currentData(),
            "screenshot": path.name,
        }
    )
    view.close()
    return evidence


def _exercise_plot(output: Path) -> dict[str, object]:
    pane = PlotCanvasPane("Interaction Controls")
    x = np.linspace(0.0, 1.0, 101)
    pane.render_data(
        PlotData(
            spec=PlotSpec(
                kind="line",
                x_key="swing.time_s",
                y_keys=("swing.speed_mps",),
                title="Deterministic Zoom, Reset, and Legend Interaction",
            ),
            x=x,
            series={
                "Reference": np.sin(2.0 * np.pi * x),
                "Comparison": np.cos(2.0 * np.pi * x),
            },
            x_label="Normalized Time [-]",
            y_label="Normalized Response [-]",
        )
    )
    pane.resize(1100, 700)
    pane.show()
    QApplication.processEvents()
    pane._zoom_in.setFocus()
    QTest.keyClick(pane._zoom_in, Qt.Key.Key_Space)  # type: ignore[call-overload]
    if pane.zoom_percent() != 125:
        raise AssertionError("Space did not activate Zoom In")
    pane._auto_fit.setFocus()
    QTest.keyClick(pane._auto_fit, Qt.Key.Key_Space)  # type: ignore[call-overload]
    if pane.zoom_percent() != 100:
        raise AssertionError("Space did not activate Auto Fit")
    pane.set_legend_placement("hidden")
    if any(
        legend.get_visible()
        for axes in pane.figure().axes
        if (legend := axes.get_legend()) is not None
    ):
        raise AssertionError("hidden legend policy did not hide the legend")
    pane.set_legend_placement("outside_right")
    if not all(
        legend.get_visible()
        for axes in pane.figure().axes
        if (legend := axes.get_legend()) is not None
    ):
        raise AssertionError("outside legend policy did not restore the legend")
    pane.zoom_in()
    path = output / "plot-controls.png"
    evidence = _save_widget(pane, path)
    evidence.update(
        {
            "legend": pane.legend_placement(),
            "screenshot": path.name,
            "zoom_percent": pane.zoom_percent(),
        }
    )
    pane.close()
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Rate of Closure Render Probe")
    manifest = {
        "artifact_policy": "diagnostic PNG; semantic manifest is test authority",
        "requested_scale": args.scale,
        "arc": _exercise_arc(args.output),
        "plot": _exercise_plot(args.output),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
