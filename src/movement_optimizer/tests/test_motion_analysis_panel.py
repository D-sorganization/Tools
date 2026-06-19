# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for the motion analysis panel and swing/chain plot renderers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure

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
def mock_ax():
    return MagicMock()


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


class TestSwingPlots:
    def test_joint_torques(self, mock_ax, swing_history) -> None:
        plot_swing_joint_torques(mock_ax, swing_history)
        assert mock_ax.plot.call_count == _NJOINTS
        assert mock_ax.axhline.call_count == 1
        assert mock_ax.set_title.call_count == 1

    def test_joint_power_includes_total(self, mock_ax, swing_history) -> None:
        plot_swing_joint_power(mock_ax, swing_history)
        assert mock_ax.plot.call_count == _NJOINTS + 1  # joints + total

    def test_angle(self, mock_ax, swing_history) -> None:
        plot_swing_angle(mock_ax, swing_history)
        assert mock_ax.plot.call_count == 1
        assert mock_ax.axhline.call_count == 1

    def test_com_height(self, mock_ax, swing_history) -> None:
        plot_swing_com_height(mock_ax, swing_history)
        assert mock_ax.plot.call_count == 1

    def test_energy(self, mock_ax, swing_history) -> None:
        plot_swing_energy(mock_ax, swing_history)
        assert mock_ax.plot.call_count == 1

    def test_com_path_marks_start_and_end(self, mock_ax, swing_history) -> None:
        plot_swing_com_path(mock_ax, swing_history)
        assert mock_ax.plot.call_count == 3  # path + start + end

    def test_legends_are_outside_plot_area(self, swing_history) -> None:
        plotters = (
            plot_swing_joint_torques,
            plot_swing_joint_power,
            plot_swing_angle,
            plot_swing_com_height,
            plot_swing_energy,
            plot_swing_com_path,
        )
        for plotter in plotters:
            figure = Figure()
            ax = figure.add_subplot(111)
            plotter(ax, swing_history)

            legend = ax.get_legend()
            assert legend is not None
            assert legend.get_bbox_to_anchor()._bbox.y0 < 0.0

    def test_panel_mode_suppresses_data_axis_legends(self, swing_history) -> None:
        plotters = (
            plot_swing_joint_torques,
            plot_swing_joint_power,
            plot_swing_angle,
            plot_swing_com_height,
            plot_swing_energy,
            plot_swing_com_path,
        )
        for plotter in plotters:
            figure = Figure()
            ax = figure.add_subplot(111)
            plotter(ax, swing_history, legend=False)

            assert ax.get_legend() is None
            handles, labels = ax.get_legend_handles_labels()
            assert handles
            assert labels


class TestChainPlots:
    def test_tension(self, mock_ax, chain_history) -> None:
        plot_chain_tension(mock_ax, chain_history)
        assert mock_ax.plot.call_count == 2  # max + mean
        assert mock_ax.set_title.call_count == 1

    def test_curvature(self, mock_ax, chain_history) -> None:
        plot_chain_curvature(mock_ax, chain_history)
        assert mock_ax.plot.call_count == 1

    def test_energy(self, mock_ax) -> None:
        plot_chain_energy(mock_ax, np.linspace(0, 1, _T), np.zeros(_T))
        assert mock_ax.plot.call_count == 1

    def test_tip_speed(self, mock_ax) -> None:
        plot_chain_tip_speed(mock_ax, np.linspace(0, 1, _T), np.zeros(_T))
        assert mock_ax.plot.call_count == 1

    def test_legends_are_outside_plot_area(self, chain_history) -> None:
        plotters = (
            lambda ax: plot_chain_tension(ax, chain_history),
            lambda ax: plot_chain_curvature(ax, chain_history),
            lambda ax: plot_chain_energy(ax, np.linspace(0, 1, _T), np.zeros(_T)),
            lambda ax: plot_chain_tip_speed(ax, np.linspace(0, 1, _T), np.zeros(_T)),
        )
        for plotter in plotters:
            figure = Figure()
            ax = figure.add_subplot(111)
            plotter(ax)

            legend = ax.get_legend()
            assert legend is not None
            assert legend.get_bbox_to_anchor()._bbox.y0 < 0.0

    def test_panel_mode_suppresses_data_axis_legends(self, chain_history) -> None:
        plotters = (
            lambda ax: plot_chain_tension(ax, chain_history, legend=False),
            lambda ax: plot_chain_curvature(ax, chain_history, legend=False),
            lambda ax: plot_chain_energy(ax, np.linspace(0, 1, _T), np.zeros(_T), legend=False),
            lambda ax: plot_chain_tip_speed(ax, np.linspace(0, 1, _T), np.zeros(_T), legend=False),
        )
        for plotter in plotters:
            figure = Figure()
            ax = figure.add_subplot(111)
            plotter(ax)

            assert ax.get_legend() is None
            handles, labels = ax.get_legend_handles_labels()
            assert handles
            assert labels


class TestMotionAnalysisPanel:
    @staticmethod
    def _assert_docked_legends_do_not_cover_plots(
        panel,
        *,
        figure_size: tuple[float, float] | None = None,
        min_legend_padding_px: float = 18.0,
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

        for legend_axis in panel.legend_axes.values():
            legend = legend_axis.get_legend()
            assert legend is not None
            assert legend.get_clip_on() is True
            assert legend.get_clip_box() is not None
            legend_box = legend.get_window_extent(renderer)
            legend_axis_box = legend_axis.get_window_extent(renderer)
            assert legend_box.x0 >= figure_box.x0 - 1.0
            assert legend_box.x1 <= figure_box.x1 + 1.0
            assert legend_box.y0 >= figure_box.y0 - 1.0
            assert legend_box.y1 <= figure_box.y1 + 1.0
            assert legend_box.y0 >= legend_axis_box.y0 - 1.0
            assert legend_box.y1 <= legend_axis_box.y1 + 1.0
            assert legend_axis_box.height >= legend_box.height + min_legend_padding_px
            assert not any(data_box.overlaps(legend_box) for data_box in data_boxes)
            assert not any(label_box.overlaps(legend_box) for label_box in label_boxes)
            assert not any(tick_box.overlaps(legend_box) for tick_box in tick_label_boxes)

    def test_axes_keys(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(["alpha", "beta", "gamma"], rows=2, cols=2)
        assert set(panel.axes) == {"alpha", "beta", "gamma"}
        assert set(panel.legend_axes) == {"alpha", "beta", "gamma"}
        assert panel.canvas is not None
        assert panel.toolbar is not None

    def test_minimum_canvas_size_preserves_swing_legend_room(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(
            ["torques", "power", "angle", "com_height", "energy", "com_path"],
            rows=2,
            cols=3,
        )

        assert panel.canvas.minimumWidth() >= 1200
        assert panel.canvas.minimumHeight() >= 832
        assert panel.minimumHeight() > panel.canvas.minimumHeight()

    def test_swingset_minimum_layout_preserves_curve_height(self, qapp, swing_history) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(
            ["torques", "power", "angle", "com_height", "energy", "com_path"],
            rows=2,
            cols=3,
        )
        plot_swing_joint_torques(panel.axes["torques"], swing_history)
        plot_swing_joint_power(panel.axes["power"], swing_history)
        plot_swing_angle(panel.axes["angle"], swing_history)
        plot_swing_com_height(panel.axes["com_height"], swing_history)
        plot_swing_energy(panel.axes["energy"], swing_history)
        plot_swing_com_path(panel.axes["com_path"], swing_history)

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
        legend_gaps = []
        for name, axes in panel.axes.items():
            legend = panel.legend_axes[name].get_legend()
            assert legend is not None
            legend_gaps.append(
                axes.get_window_extent(renderer).y0 - legend.get_window_extent(renderer).y1
            )

        assert min(data_heights) >= 210.0
        assert max(legend_gaps) <= 110.0

    def test_swingset_live_tab_layout_preserves_usable_plot_width(
        self, qapp, swing_history
    ) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(
            ["torques", "power", "angle", "com_height", "energy", "com_path"],
            rows=3,
            cols=2,
        )
        plot_swing_joint_torques(panel.axes["torques"], swing_history, legend=False)
        plot_swing_joint_power(panel.axes["power"], swing_history, legend=False)
        plot_swing_angle(panel.axes["angle"], swing_history, legend=False)
        plot_swing_com_height(panel.axes["com_height"], swing_history, legend=False)
        plot_swing_energy(panel.axes["energy"], swing_history, legend=False)
        plot_swing_com_path(panel.axes["com_path"], swing_history, legend=False)

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
        self._assert_docked_legends_do_not_cover_plots(
            panel,
            figure_size=(minimum_size.width() / 100.0, minimum_size.height() / 100.0),
            min_legend_padding_px=2.0,
        )

    def test_docked_joint_legends_use_compact_three_column_rows(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        assert MotionAnalysisPanel._legend_columns(0) == 1
        assert MotionAnalysisPanel._legend_columns(1) == 1
        assert MotionAnalysisPanel._legend_columns(2) == 2
        assert MotionAnalysisPanel._legend_columns(5) == 3

    def test_clear_rebuilds_axes(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(["a"], rows=1, cols=1)
        first = panel.axes["a"]
        panel.clear()
        assert set(panel.axes) == {"a"}
        assert panel.axes["a"] is not first

    def test_draw_runs(self, qapp, swing_history) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(["torques"], rows=1, cols=1)
        plot_swing_joint_torques(panel.axes["torques"], swing_history)
        panel.draw()  # should not raise

    def test_swingset_legends_are_docked_outside_data_axes(self, qapp, swing_history) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(
            ["torques", "power", "angle", "com_height", "energy", "com_path"],
            rows=2,
            cols=3,
        )
        plot_swing_joint_torques(panel.axes["torques"], swing_history)
        plot_swing_joint_power(panel.axes["power"], swing_history)
        plot_swing_angle(panel.axes["angle"], swing_history)
        plot_swing_com_height(panel.axes["com_height"], swing_history)
        plot_swing_energy(panel.axes["energy"], swing_history)
        plot_swing_com_path(panel.axes["com_path"], swing_history)

        self._assert_docked_legends_do_not_cover_plots(panel)
        assert all(axes.get_legend() is None for axes in panel.axes.values())
        assert all(
            panel.axes[name].get_zorder() > panel.legend_axes[name].get_zorder()
            for name in panel.axes
        )

    def test_swingset_docked_legends_clear_minimum_plot_size(self, qapp, swing_history) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(
            ["torques", "power", "angle", "com_height", "energy", "com_path"],
            rows=2,
            cols=3,
        )
        plot_swing_joint_torques(panel.axes["torques"], swing_history, legend=False)
        plot_swing_joint_power(panel.axes["power"], swing_history, legend=False)
        plot_swing_angle(panel.axes["angle"], swing_history, legend=False)
        plot_swing_com_height(panel.axes["com_height"], swing_history, legend=False)
        plot_swing_energy(panel.axes["energy"], swing_history, legend=False)
        plot_swing_com_path(panel.axes["com_path"], swing_history, legend=False)

        minimum_size = panel.canvas.minimumSize()
        self._assert_docked_legends_do_not_cover_plots(
            panel,
            figure_size=(minimum_size.width() / 100.0, minimum_size.height() / 100.0),
        )
        assert all(axes.get_legend() is None for axes in panel.axes.values())

    def test_swingset_docked_legends_clear_compressed_plot_size(self, qapp, swing_history) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(
            ["torques", "power", "angle", "com_height", "energy", "com_path"],
            rows=3,
            cols=2,
        )
        plot_swing_joint_torques(panel.axes["torques"], swing_history, legend=False)
        plot_swing_joint_power(panel.axes["power"], swing_history, legend=False)
        plot_swing_angle(panel.axes["angle"], swing_history, legend=False)
        plot_swing_com_height(panel.axes["com_height"], swing_history, legend=False)
        plot_swing_energy(panel.axes["energy"], swing_history, legend=False)
        plot_swing_com_path(panel.axes["com_path"], swing_history, legend=False)

        self._assert_docked_legends_do_not_cover_plots(
            panel,
            figure_size=(5.2, 7.2),
            min_legend_padding_px=2.0,
        )
        assert all(axes.get_legend() is None for axes in panel.axes.values())

    def test_draw_enforces_minimum_render_size_before_docking_legends(
        self, qapp, swing_history
    ) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(
            ["torques", "power", "angle", "com_height", "energy", "com_path"],
            rows=3,
            cols=2,
        )
        plot_swing_joint_torques(panel.axes["torques"], swing_history, legend=False)
        plot_swing_joint_power(panel.axes["power"], swing_history, legend=False)
        plot_swing_angle(panel.axes["angle"], swing_history, legend=False)
        plot_swing_com_height(panel.axes["com_height"], swing_history, legend=False)
        plot_swing_energy(panel.axes["energy"], swing_history, legend=False)
        plot_swing_com_path(panel.axes["com_path"], swing_history, legend=False)

        panel.figure.set_size_inches(3.6, 4.8, forward=True)
        panel.draw()

        minimum_size = panel.canvas.minimumSize()
        assert panel.figure.bbox.width >= minimum_size.width() - 1.0
        assert panel.figure.bbox.height >= minimum_size.height() - 1.0
        self._assert_docked_legends_do_not_cover_plots(panel)

    def test_chain_legends_are_docked_outside_data_axes(self, qapp, chain_history) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(["tension", "curvature", "energy", "tip_speed"], rows=2, cols=2)
        plot_chain_tension(panel.axes["tension"], chain_history)
        plot_chain_curvature(panel.axes["curvature"], chain_history)
        plot_chain_energy(panel.axes["energy"], np.linspace(0, 1, _T), np.zeros(_T))
        plot_chain_tip_speed(panel.axes["tip_speed"], np.linspace(0, 1, _T), np.zeros(_T))

        self._assert_docked_legends_do_not_cover_plots(panel)
        assert all(axes.get_legend() is None for axes in panel.axes.values())

    def test_rejects_empty_axis_names(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        with pytest.raises(ValueError, match="non-empty"):
            MotionAnalysisPanel([], rows=1, cols=1)

    def test_rejects_undersized_grid(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        with pytest.raises(ValueError, match=r"rows \* cols"):
            MotionAnalysisPanel(["a", "b", "c"], rows=1, cols=1)

    def test_rejects_nonpositive_dims(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        with pytest.raises(ValueError, match="positive"):
            MotionAnalysisPanel(["a"], rows=0, cols=1)

    def test_has_legends_reflects_axis_state(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(["a", "b"], rows=1, cols=2)
        assert panel.has_legends() is False
        panel.axes["a"].plot([0, 1], [0, 1], label="series")
        panel.axes["a"].legend()
        assert panel.has_legends() is True

    def test_set_legends_visible_toggles_only_legend_bearing_axes(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(["a", "b"], rows=1, cols=2)
        panel.axes["a"].plot([0, 1], [0, 1], label="series")
        legend = panel.axes["a"].legend()

        panel.set_legends_visible(False)
        assert legend.get_visible() is False
        panel.set_legends_visible(True)
        assert legend.get_visible() is True
        # The legend-free axis is simply skipped (no error).
        assert panel.axes["b"].get_legend() is None

    def test_set_legends_visible_controls_docked_legends(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(["a"], rows=1, cols=1)
        panel.axes["a"].plot([0, 1], [0, 1], label="series")
        panel.axes["a"].legend()

        panel.set_legends_visible(False)
        panel.draw()
        assert panel.axes["a"].get_legend() is None
        assert panel.legend_axes["a"].get_legend() is None

        panel.set_legends_visible(True)
        panel.draw()
        assert panel.axes["a"].get_legend() is None
        assert panel.legend_axes["a"].get_legend() is not None
