"""Testable PNG, SVG, and plot-definition exports for variation figures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from matplotlib.figure import Figure
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton, QWidget

from rate_of_closure.variation.plot_data import EnsemblePlotDataset
from rate_of_closure.variation.plot_definition import (
    PlotDefinition,
    write_plot_definition,
)
from shared.python.swing_sim.variation import VariationDataset


def scatter_plot_definition(
    dataset: EnsemblePlotDataset | None,
    variation: VariationDataset | None,
    x_variable_key: str,
    y_variable_key: str,
    selected_trial_index: int | None,
) -> PlotDefinition:
    """Build a scalar definition from current selector state."""
    if variation is None:
        raise RuntimeError("no variation result is loaded")
    result_id = (
        dataset.result_id
        if dataset
        else (f"variation-{variation.plan.seed}-{variation.plan.n_runs}")
    )
    return PlotDefinition(
        result_id=result_id,
        plot_type="scalar_scatter",
        coordinate_frame=dataset.coordinate_frame if dataset else None,
        x_variable_key=x_variable_key,
        y_variable_key=y_variable_key,
        selected_trial_index=selected_trial_index,
    )


def arc_plot_definition(
    dataset: EnsemblePlotDataset | None,
    point_id: str,
    quiet_threshold_m: float,
    selected_trial_index: int | None,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
) -> PlotDefinition:
    """Build a geometric definition from current selector and camera state."""
    if dataset is None:
        raise RuntimeError("no swing ensemble is loaded")
    return PlotDefinition(
        result_id=dataset.result_id,
        plot_type="swing_arc_overlay",
        coordinate_frame=dataset.coordinate_frame,
        point_id=point_id,
        position_unit="m",
        alignment_basis="common_simulation_time_s",
        quiet_threshold_m=quiet_threshold_m,
        selected_trial_index=selected_trial_index,
        camera_yaw_deg=camera_yaw_deg,
        camera_pitch_deg=camera_pitch_deg,
    )


class VariationPlotExportControls(QWidget):
    """Compact export bar whose write methods also support headless tests."""

    def __init__(
        self,
        figure: Callable[[], Figure],
        definition: Callable[[], PlotDefinition],
        stem: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._figure = figure
        self._definition = definition
        self._stem = stem
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for label, kind, callback in (
            ("Export PNG…", "PNG (*.png)", self.write_png),
            ("Export SVG…", "SVG (*.svg)", self.write_svg),
            ("Plot Definition JSON…", "JSON (*.json)", self.write_definition),
        ):
            button = QPushButton(label)
            export_name = label.removesuffix("…")
            button.setToolTip(
                f"Save the current variation visualization as {export_name}."
            )
            button.clicked.connect(
                lambda _checked=False, file_filter=kind, writer=callback: self._choose(
                    file_filter, writer
                )
            )
            layout.addWidget(button)
        layout.addStretch(1)

    def _choose(self, file_filter: str, writer: Callable[[str | Path], None]) -> None:
        suffix = file_filter.split("*", 1)[-1].split(")", 1)[0]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Variation Plot",
            f"{self._stem}{suffix}",
            file_filter,
        )
        if path:
            writer(path)

    def write_png(self, path: str | Path) -> None:
        """Write the current figure as a 200-dpi PNG."""
        self._figure().savefig(path, dpi=200, format="png")

    def write_svg(self, path: str | Path) -> None:
        """Write the current figure as a vector SVG."""
        self._figure().savefig(path, format="svg")

    def write_definition(self, path: str | Path) -> None:
        """Write the current versioned plot state."""
        write_plot_definition(self._definition(), path)


__all__ = [
    "VariationPlotExportControls",
    "arc_plot_definition",
    "scatter_plot_definition",
]
