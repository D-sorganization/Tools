"""Tests for clipboard utility functions."""

from __future__ import annotations
import numpy as np
from double_pendulum_golf.gui.clipboard_utils import (
    matrix_to_tsv,
    series_to_tsv,
    scalar_dict_to_text,
)


class TestMatrixToTsv:
    def test_2x2_matrix(self) -> None:
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = matrix_to_tsv(m)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "1" in lines[0] and "2" in lines[0]

    def test_single_row(self) -> None:
        m = np.array([[1.0, 2.0, 3.0]])
        result = matrix_to_tsv(m)
        assert "\n" not in result
        assert "\t" in result


class TestSeriesToTsv:
    def test_basic(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        result = series_to_tsv(x, y, "time", "angle")
        lines = result.split("\n")
        assert lines[0] == "time\tangle"
        assert len(lines) == 4  # header + 3 data

    def test_custom_labels(self) -> None:
        x = np.array([0.0])
        y = np.array([1.0])
        result = series_to_tsv(x, y, "t", "θ")
        assert result.startswith("t\tθ")


class TestScalarDictToText:
    def test_with_title(self) -> None:
        d = {"kinetic": 1.5, "potential": -2.3}
        result = scalar_dict_to_text(d, "Energy")
        assert "Energy" in result
        assert "kinetic" in result

    def test_without_title(self) -> None:
        d = {"x": 1.0}
        result = scalar_dict_to_text(d)
        assert "x: 1" in result
