"""MotionAnalysisPanel docked-legend layout regressions."""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.gui.plot_renderer import (
    plot_chain_curvature,
    plot_chain_energy,
    plot_chain_tension,
    plot_chain_tip_speed,
    plot_swing_angle,
    plot_swing_com_height,
    plot_swing_com_path,
    plot_swing_energy,
    plot_swing_joint_power,
    plot_swing_joint_torques,
)
from movement_optimizer.models.chain_forces import ChainForceHistory
from movement_optimizer.models.swingset import SWING_POLICY_JOINT_NAMES
from movement_optimizer.models.swingset_forces import SwingForceHistory

_T = 12
_N = 5
_NJOINTS = len(SWING_POLICY_JOINT_NAMES)


@pytest.fixture
def swing_history() -> SwingForceHistory:
    time = np.linspace(0.0, 1.0, _T)
    return SwingForceHistory(
        time_s=time,
        joint_torque_nm=np.ones((_T, _NJOINTS)),
        joint_power_w=np.ones((_T, _NJOINTS)),
        swing_angle_rad=np.linspace(-0.5, 0.5, _T),
        com_height_m=np.linspace(0.0, 0.3, _T),
        com_path_m=np.column_stack([np.sin(time), np.cos(time)]),
        energy_j=np.linspace(0.0, 10.0, _T),
    )


@pytest.fixture
def chain_history() -> ChainForceHistory:
    return ChainForceHistory(
        time_s=np.linspace(0.0, 1.0, _T),
        link_tension_n=np.ones((_T, _N)),
        max_tension_n=np.ones(_T),
        curvature_rad=np.zeros((_T, _N - 1)),
        max_curvature_rad=np.zeros(_T),
    )


def _assert_panel_legends_do_not_cover_plots(
    panel,
    *,
    figure_size: tuple[float, float] | None = None,
) -> None:
    if figure_size is None:
        minimum_size = panel.canvas.minimumSize()
        figure_size = (
            minimum_size.width() / 100.0,
            minimum_size.height() / 100.0,
        )
    panel.figure.set_size_inches(*figure_size, forward=True)
    panel.draw()
    panel.canvas.draw()
    renderer = panel.canvas.get_renderer()
    figure_box = panel.figure.bbox
    data_boxes = [axes.get_window_extent(renderer) for axes in panel.axes.values()]
    label_boxes = [
        artist.get_window_extent(renderer)
        for axes in panel.axes.values()
        for artist in (axes.title, axes.xaxis.label, axes.yaxis.label)
    ]
    tick_label_boxes = [
        tick.get_window_extent(renderer)
        for axes in panel.axes.values()
        for tick in (*axes.get_xticklabels(), *axes.get_yticklabels())
        if tick.get_visible()
    ]
    legends = [
        legend
        for legend_axis in panel.legend_axes.values()
        if (legend := legend_axis.get_legend()) is not None
    ]
    assert legends
    assert panel._figure_legend is None
    assert all(axes.get_legend() is None for axes in panel.axes.values())

    for legend in legends:
        legend_box = legend.get_window_extent(renderer)
        assert legend_box.x0 >= figure_box.x0 - 1.0
        assert legend_box.x1 <= figure_box.x1 + 1.0
        assert legend_box.y0 >= figure_box.y0 - 1.0
        assert legend_box.y1 <= figure_box.y1 + 1.0
        assert not any(data_box.overlaps(legend_box) for data_box in data_boxes)
        assert not any(label_box.overlaps(legend_box) for label_box in label_boxes)
        assert not any(tick_box.overlaps(legend_box) for tick_box in tick_label_boxes)


def _plot_swing_panel(panel, swing_history: SwingForceHistory, *, legend: bool) -> None:
    plot_swing_joint_torques(panel.axes["torques"], swing_history, legend=legend)
    plot_swing_joint_power(panel.axes["power"], swing_history, legend=legend)
    plot_swing_angle(panel.axes["angle"], swing_history, legend=legend)
    plot_swing_com_height(panel.axes["com_height"], swing_history, legend=legend)
    plot_swing_energy(panel.axes["energy"], swing_history, legend=legend)
    plot_swing_com_path(panel.axes["com_path"], swing_history, legend=legend)


def test_swingset_minimum_layout_preserves_curve_height(qapp, swing_history) -> None:
    from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

    panel = MotionAnalysisPanel(
        ["torques", "power", "angle", "com_height", "energy", "com_path"],
        rows=2,
        cols=3,
    )
    _plot_swing_panel(panel, swing_history, legend=True)
    minimum_size = panel.canvas.minimumSize()
    panel.figure.set_size_inches(
        minimum_size.width() / 100.0,
        minimum_size.height() / 100.0,
        forward=True,
    )
    panel.draw()
    panel.canvas.draw()
    renderer = panel.canvas.get_renderer()
    data_heights = [axes.get_window_extent(renderer).height for axes in panel.axes.values()]

    assert min(data_heights) >= 210.0
    _assert_panel_legends_do_not_cover_plots(panel)


def test_swingset_live_tab_layout_preserves_usable_plot_width(qapp, swing_history) -> None:
    from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

    panel = MotionAnalysisPanel(
        ["torques", "power", "angle", "com_height", "energy", "com_path"],
        rows=3,
        cols=2,
    )
    _plot_swing_panel(panel, swing_history, legend=False)
    minimum_size = panel.canvas.minimumSize()
    panel.figure.set_size_inches(
        minimum_size.width() / 100.0,
        minimum_size.height() / 100.0,
        forward=True,
    )
    panel.draw()
    renderer = panel.canvas.get_renderer()
    data_widths = [axes.get_window_extent(renderer).width for axes in panel.axes.values()]

    assert min(data_widths) >= 300.0
    _assert_panel_legends_do_not_cover_plots(
        panel,
        figure_size=(minimum_size.width() / 100.0, minimum_size.height() / 100.0),
    )


def test_swingset_legends_are_docked_in_reserved_rows(qapp, swing_history) -> None:
    from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

    panel = MotionAnalysisPanel(
        ["torques", "power", "angle", "com_height", "energy", "com_path"],
        rows=3,
        cols=2,
    )
    _plot_swing_panel(panel, swing_history, legend=False)
    panel.draw()
    renderer = panel.canvas.get_renderer()
    figure_box = panel.figure.bbox

    assert panel._figure_legend is None
    assert all(axes.get_legend() is None for axes in panel.axes.values())
    assert all(axes.get_legend() is not None for axes in panel.legend_axes.values())
    for legend_axis in panel.legend_axes.values():
        legend = legend_axis.get_legend()
        assert legend is not None
        legend_box = legend.get_window_extent(renderer)
        assert legend_box.x0 >= figure_box.x0 - 1.0
        assert legend_box.x1 <= figure_box.x1 + 1.0
        assert not any(
            legend_box.overlaps(axes.get_window_extent(renderer)) for axes in panel.axes.values()
        )


def test_swingset_legends_are_docked_outside_data_axes(qapp, swing_history) -> None:
    from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

    panel = MotionAnalysisPanel(
        ["torques", "power", "angle", "com_height", "energy", "com_path"],
        rows=2,
        cols=3,
    )
    _plot_swing_panel(panel, swing_history, legend=True)

    _assert_panel_legends_do_not_cover_plots(panel)
    assert all(axes.get_legend() is None for axes in panel.axes.values())


def test_swingset_docked_legends_clear_minimum_plot_size(qapp, swing_history) -> None:
    from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

    panel = MotionAnalysisPanel(
        ["torques", "power", "angle", "com_height", "energy", "com_path"],
        rows=2,
        cols=3,
    )
    _plot_swing_panel(panel, swing_history, legend=False)
    minimum_size = panel.canvas.minimumSize()

    _assert_panel_legends_do_not_cover_plots(
        panel,
        figure_size=(minimum_size.width() / 100.0, minimum_size.height() / 100.0),
    )
    assert all(axes.get_legend() is None for axes in panel.axes.values())


def test_swingset_docked_legends_clear_compressed_plot_size(qapp, swing_history) -> None:
    from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

    panel = MotionAnalysisPanel(
        ["torques", "power", "angle", "com_height", "energy", "com_path"],
        rows=3,
        cols=2,
    )
    _plot_swing_panel(panel, swing_history, legend=False)

    _assert_panel_legends_do_not_cover_plots(panel, figure_size=(5.2, 7.2))
    assert all(axes.get_legend() is None for axes in panel.axes.values())


def test_draw_enforces_minimum_render_size_before_docking_legends(qapp, swing_history) -> None:
    from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

    panel = MotionAnalysisPanel(
        ["torques", "power", "angle", "com_height", "energy", "com_path"],
        rows=3,
        cols=2,
    )
    _plot_swing_panel(panel, swing_history, legend=False)
    panel.figure.set_size_inches(3.6, 4.8, forward=True)
    panel.draw()
    minimum_size = panel.canvas.minimumSize()

    assert panel.figure.bbox.width >= minimum_size.width() - 1.0
    assert panel.figure.bbox.height >= minimum_size.height() - 1.0
    _assert_panel_legends_do_not_cover_plots(panel)


def test_chain_legends_are_docked_outside_data_axes(qapp, chain_history) -> None:
    from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

    panel = MotionAnalysisPanel(["tension", "curvature", "energy", "tip_speed"], rows=2, cols=2)
    plot_chain_tension(panel.axes["tension"], chain_history)
    plot_chain_curvature(panel.axes["curvature"], chain_history)
    plot_chain_energy(panel.axes["energy"], np.linspace(0, 1, _T), np.zeros(_T))
    plot_chain_tip_speed(panel.axes["tip_speed"], np.linspace(0, 1, _T), np.zeros(_T))

    _assert_panel_legends_do_not_cover_plots(panel)
    assert all(axes.get_legend() is None for axes in panel.axes.values())
