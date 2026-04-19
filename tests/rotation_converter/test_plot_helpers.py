"""Tests for rotation-converter plotting helper validation."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from rotation_converter.ui.pyqt6.plot_helpers import fmt_mat, style_figure


def test_fmt_mat_rejects_none_matrix() -> None:
    with pytest.raises(ValueError, match="M must be provided"):
        fmt_mat(None)


def test_fmt_mat_formats_numpy_matrix() -> None:
    assert fmt_mat(np.eye(2), decimals=1) == " 1.0   0.0\n 0.0   1.0"


def test_style_figure_rejects_none_figure() -> None:
    with pytest.raises(ValueError, match="fig must be provided"):
        style_figure(None)


def test_style_figure_accepts_figure_without_axes() -> None:
    fig = Figure()
    style_figure(fig)
