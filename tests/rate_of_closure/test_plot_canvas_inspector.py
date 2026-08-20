"""Atomic publication tests for one managed PyQt plot pane."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from rate_of_closure.plot_point_inspector import SeriesSelection
from rate_of_closure.plotting import PlotData, PlotSpec
from rate_of_closure.ui.pyqt6.plot_canvas_pane import PlotCanvasPane

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _data(offset: float = 0.0) -> PlotData:
    return PlotData(
        spec=PlotSpec(
            kind="line",
            x_key="swing.time_s",
            y_keys=("swing.speed_mps",),
            title="Atomic Evidence",
        ),
        x=np.array([0.0, 1.0, 2.0]),
        series={"Speed": np.array([10.0, 11.0, 12.0]) + offset},
        x_label="Time [s]",
        y_label="Speed [m/s]",
    )


def test_candidate_render_failure_retains_exact_prior_bundle(
    qtbot, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    pane = PlotCanvasPane("Atomic")
    qtbot.addWidget(pane)
    pane.render_data(_data())
    qtbot.keyClick(pane.canvas(), Qt.Key.Key_Home)
    prior_data = pane._data
    prior_figure = pane.figure()
    prior_selection = pane.selected_evidence()

    def fail(_data: PlotData, _figure: object) -> None:
        raise RuntimeError("planted candidate renderer failure")

    monkeypatch.setattr("rate_of_closure.ui.pyqt6.plot_canvas_pane.render_plot", fail)
    with pytest.raises(RuntimeError, match="candidate renderer"):
        pane.render_data(_data(5.0))
    assert pane._data is prior_data
    assert pane.figure() is prior_figure
    assert pane.selected_evidence() == prior_selection


def test_marker_failure_retains_selection_and_success_clears_error(
    qtbot, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    pane = PlotCanvasPane("Atomic")
    qtbot.addWidget(pane)
    pane.render_data(_data())
    pane._adopt_selection(SeriesSelection(0, 0))
    axes = pane.figure().axes[0]
    original = axes.scatter

    def fail(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("planted marker failure")

    monkeypatch.setattr(axes, "scatter", fail)
    pane._adopt_selection(SeriesSelection(0, 1))
    assert pane.selected_evidence() == SeriesSelection(0, 0)
    assert "Selection failed; prior evidence retained" in pane.inspection_status()

    monkeypatch.setattr(axes, "scatter", original)
    pane._adopt_selection(SeriesSelection(0, 1))
    assert pane.selected_evidence() == SeriesSelection(0, 1)
    assert "Selection failed" not in pane.inspection_status()
