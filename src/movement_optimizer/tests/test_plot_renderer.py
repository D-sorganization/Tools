# Copyright (c) 2026 D-Sorganization. All rights reserved.
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.collections import LineCollection

from movement_optimizer.gui.plot_renderer import (
    plot_angles,
    plot_com_balance,
    plot_com_path,
    plot_power,
    plot_spine_loads,
    plot_torques,
)
from movement_optimizer.models import BodyModel
from movement_optimizer.trajectory import OptimizationResult


@pytest.fixture
def mock_ax():
    return MagicMock()


@pytest.fixture
def dummy_result():
    t = np.linspace(0, 1, 10)
    q = np.zeros((10, 3))
    qd = np.zeros((10, 3))
    qdd = np.zeros((10, 3))
    torques = np.zeros((10, 3))
    power = np.zeros((10, 3))
    com = np.zeros((10, 2))
    bar = np.zeros((10, 2))
    return OptimizationResult(
        t=t,
        q=q,
        qd=qd,
        qdd=qdd,
        torques=torques,
        power=power,
        com=com,
        bar=bar,
        success=True,
        cost=0.0,
        com_horizontal_range_cm=0.0,
        elapsed_s=0.1,
        n_evals=100,
        n_joint_limit_violations=0,
    )


@pytest.fixture
def body():
    return BodyModel(body_mass=80.0, height=1.8)


class TestPlotRenderer:
    def test_plot_angles(self, mock_ax, dummy_result):
        plot_angles(mock_ax, dummy_result)
        assert mock_ax.plot.call_count == 3
        mock_ax.set_title.assert_called_once_with(
            "Joint Angles", color=mock_ax.set_title.call_args[1].get("color"), fontsize=10
        )

    def test_plot_torques(self, mock_ax, dummy_result):
        plot_torques(mock_ax, dummy_result)
        assert mock_ax.plot.call_count == 3
        mock_ax.axhline.assert_called_once()
        mock_ax.set_title.assert_called_once()

    def test_plot_power(self, mock_ax, dummy_result):
        plot_power(mock_ax, dummy_result)
        # 3 joints + 1 total
        assert mock_ax.plot.call_count == 4
        mock_ax.axhline.assert_called_once()
        mock_ax.set_title.assert_called_once()

    def test_plot_com_path(self, mock_ax, dummy_result, body):
        plot_com_path(mock_ax, dummy_result, body)
        mock_ax.add_collection.assert_called_once()
        collection = mock_ax.add_collection.call_args.args[0]
        assert isinstance(collection, LineCollection)
        assert len(collection.get_segments()) == len(dummy_result.com) - 1
        # Bar path + COM straight + Start + End. The colour-graded COM path is
        # one collection, not one Line2D artist per sample.
        assert mock_ax.plot.call_count == 4
        mock_ax.autoscale_view.assert_called_once()
        assert mock_ax.axvline.call_count == 4
        mock_ax.set_title.assert_called_once()

    def test_plot_com_path_rejects_degenerate_trace(self, mock_ax, dummy_result, body):
        one_sample_result = replace(dummy_result, com=np.zeros((1, 2)))

        with pytest.raises(ValueError, match="at least two samples"):
            plot_com_path(mock_ax, one_sample_result, body)

    def test_plot_com_balance(self, mock_ax, dummy_result, body):
        plot_com_balance(mock_ax, dummy_result, body)
        mock_ax.plot.assert_called_once()
        assert mock_ax.axhline.call_count == 5
        assert mock_ax.fill_between.call_count == 2
        mock_ax.set_title.assert_called_once()

    def test_plot_spine_loads(self, mock_ax, dummy_result, body):
        ax_comp = MagicMock()
        ax_shear = MagicMock()
        plot_spine_loads(ax_comp, ax_shear, dummy_result, body, bar_mass=20.0, name="squat")

        ax_comp.plot.assert_called_once()
        ax_comp.axhline.assert_called_once()
        ax_comp.fill_between.assert_called_once()
        ax_comp.set_title.assert_called_once()

        ax_shear.plot.assert_called_once()
        ax_shear.axhline.assert_called_once()
        ax_shear.set_title.assert_called_once()


class TestPlotSpineLoadsExerciseAlias:
    def test_bottoms_up_squat_is_aliased_to_squat(self, dummy_result, body):
        from unittest.mock import MagicMock

        from movement_optimizer.gui.plot_renderer import plot_spine_loads

        ax_comp = MagicMock()
        ax_shear = MagicMock()
        plot_spine_loads(ax_comp, ax_shear, dummy_result, body, 60.0, "Bottoms Up Squat")
        assert ax_comp.plot.called
        assert ax_shear.plot.called
