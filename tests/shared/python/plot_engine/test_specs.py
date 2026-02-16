"""Tests for plot_engine.specs Pydantic models.

Covers:
- Default construction of all spec models
- Field validation (ranges, enums)
- Serialization (model_dump)
- Inheritance hierarchy (SurfacePlotSpec extends PlotSpec)
- SeriesData construction with styling and trendline
"""

from __future__ import annotations

import pytest
from plot_engine.specs import (
    AxisSpec,
    ContourPlotSpec,
    FilterComparisonSpec,
    HeatmapSpec,
    HistogramSpec,
    LegendSpec,
    PlotSpec,
    SeriesData,
    SeriesStyle,
    SurfacePlotSpec,
    TrendlineSpec,
)
from pydantic import ValidationError

# ── SeriesStyle ──────────────────────────────────────────────────────────


class TestSeriesStyle:
    """Test SeriesStyle Pydantic model."""

    def test_default_construction(self) -> None:
        style = SeriesStyle()
        assert style.line_style == "solid"
        assert style.line_width == 1.5
        assert style.marker == "none"
        assert style.opacity == 1.0
        assert style.display_mode == "line"

    def test_custom_style(self) -> None:
        style = SeriesStyle(
            color="#ff0000",
            line_style="dashed",
            line_width=2.5,
            marker="circle",
            opacity=0.8,
        )
        assert style.color == "#ff0000"
        assert style.line_style == "dashed"
        assert style.line_width == 2.5

    def test_opacity_range_validation(self) -> None:
        with pytest.raises(ValidationError):
            SeriesStyle(opacity=1.5)

    def test_opacity_negative_invalid(self) -> None:
        with pytest.raises(ValidationError):
            SeriesStyle(opacity=-0.1)

    def test_invalid_line_style(self) -> None:
        with pytest.raises(ValidationError):
            SeriesStyle(line_style="wavy")

    def test_invalid_marker(self) -> None:
        with pytest.raises(ValidationError):
            SeriesStyle(marker="heart")

    def test_display_mode_options(self) -> None:
        for mode in ("line", "scatter", "line+scatter"):
            style = SeriesStyle(display_mode=mode)
            assert style.display_mode == mode


# ── TrendlineSpec ────────────────────────────────────────────────────────


class TestTrendlineSpec:
    """Test TrendlineSpec Pydantic model."""

    def test_linear_trendline(self) -> None:
        ts = TrendlineSpec(type="linear")
        assert ts.type == "linear"
        assert ts.show_equation is True
        assert ts.show_r_squared is True

    def test_polynomial_with_degree(self) -> None:
        ts = TrendlineSpec(type="polynomial", degree=3)
        assert ts.degree == 3

    def test_degree_range_validation(self) -> None:
        with pytest.raises(ValidationError):
            TrendlineSpec(type="polynomial", degree=0)

    def test_degree_max_validation(self) -> None:
        with pytest.raises(ValidationError):
            TrendlineSpec(type="polynomial", degree=11)

    def test_invalid_type(self) -> None:
        with pytest.raises(ValidationError):
            TrendlineSpec(type="logarithmic")


# ── AxisSpec ─────────────────────────────────────────────────────────────


class TestAxisSpec:
    """Test AxisSpec Pydantic model."""

    def test_defaults(self) -> None:
        axis = AxisSpec()
        assert axis.label == ""
        assert axis.min is None
        assert axis.max is None
        assert axis.log_scale is False
        assert axis.grid is True

    def test_custom_axis(self) -> None:
        axis = AxisSpec(label="Time (s)", min=0, max=100, log_scale=True)
        assert axis.label == "Time (s)"
        assert axis.min == 0
        assert axis.max == 100
        assert axis.log_scale is True


# ── LegendSpec ───────────────────────────────────────────────────────────


class TestLegendSpec:
    """Test LegendSpec Pydantic model."""

    def test_defaults(self) -> None:
        legend = LegendSpec()
        assert legend.visible is True
        assert legend.position == "right"

    def test_hidden_legend(self) -> None:
        legend = LegendSpec(visible=False, position="none")
        assert legend.visible is False

    def test_invalid_position(self) -> None:
        with pytest.raises(ValidationError):
            LegendSpec(position="center")


# ── SeriesData ───────────────────────────────────────────────────────────


class TestSeriesData:
    """Test SeriesData Pydantic model."""

    def test_construction(self) -> None:
        sd = SeriesData(name="Signal A", x=[0, 1, 2], y=[0, 1, 4])
        assert sd.name == "Signal A"
        assert sd.x == [0, 1, 2]
        assert sd.y == [0, 1, 4]

    def test_default_style(self) -> None:
        sd = SeriesData(name="test", x=[0], y=[0])
        assert isinstance(sd.style, SeriesStyle)

    def test_with_trendline(self) -> None:
        sd = SeriesData(
            name="test",
            x=[0, 1, 2],
            y=[0, 1, 4],
            trendline=TrendlineSpec(type="linear"),
        )
        assert sd.trendline is not None
        assert sd.trendline.type == "linear"


# ── PlotSpec ─────────────────────────────────────────────────────────────


class TestPlotSpec:
    """Test PlotSpec Pydantic model."""

    def test_defaults(self) -> None:
        ps = PlotSpec()
        assert ps.title == ""
        assert ps.series == []
        assert ps.width == 800
        assert ps.height == 600

    def test_with_series(self) -> None:
        ps = PlotSpec(
            title="My Plot",
            series=[SeriesData(name="A", x=[1, 2], y=[3, 4])],
        )
        assert len(ps.series) == 1
        assert ps.series[0].name == "A"

    def test_width_min_validation(self) -> None:
        with pytest.raises(ValidationError):
            PlotSpec(width=50)

    def test_width_max_validation(self) -> None:
        with pytest.raises(ValidationError):
            PlotSpec(width=5000)

    def test_serialization(self) -> None:
        ps = PlotSpec(title="Test")
        data = ps.model_dump()
        assert isinstance(data, dict)
        assert data["title"] == "Test"


# ── SurfacePlotSpec ──────────────────────────────────────────────────────


class TestSurfacePlotSpec:
    """Test SurfacePlotSpec inherits from PlotSpec."""

    def test_construction(self) -> None:
        sp = SurfacePlotSpec(
            title="Surface",
            z_data=[[1, 2], [3, 4]],
            x_grid=[0, 1],
            y_grid=[0, 1],
        )
        assert sp.title == "Surface"
        assert sp.colormap == "viridis"

    def test_inherits_plot_spec(self) -> None:
        sp = SurfacePlotSpec(
            z_data=[[1]],
            x_grid=[0],
            y_grid=[0],
            width=1200,
        )
        assert sp.width == 1200
        assert isinstance(sp, PlotSpec)


# ── ContourPlotSpec ──────────────────────────────────────────────────────


class TestContourPlotSpec:
    """Test ContourPlotSpec."""

    def test_defaults(self) -> None:
        cp = ContourPlotSpec(
            z_data=[[1, 2], [3, 4]],
            x_grid=[0, 1],
            y_grid=[0, 1],
        )
        assert cp.levels == 20
        assert cp.filled is True
        assert cp.show_colorbar is True

    def test_levels_validation(self) -> None:
        with pytest.raises(ValidationError):
            ContourPlotSpec(z_data=[[1]], x_grid=[0], y_grid=[0], levels=1)


# ── HeatmapSpec ──────────────────────────────────────────────────────────


class TestHeatmapSpec:
    """Test HeatmapSpec."""

    def test_defaults(self) -> None:
        hm = HeatmapSpec(z_data=[[1, 2], [3, 4]])
        assert hm.colormap == "YlGnBu"
        assert hm.annotate is False

    def test_with_labels(self) -> None:
        hm = HeatmapSpec(
            z_data=[[1, 2], [3, 4]],
            x_labels=["A", "B"],
            y_labels=["X", "Y"],
        )
        assert hm.x_labels == ["A", "B"]


# ── HistogramSpec ────────────────────────────────────────────────────────


class TestHistogramSpec:
    """Test HistogramSpec."""

    def test_defaults(self) -> None:
        hs = HistogramSpec()
        assert hs.bins == 30
        assert hs.density is False
        assert hs.cumulative is False

    def test_bins_validation(self) -> None:
        with pytest.raises(ValidationError):
            HistogramSpec(bins=0)


# ── FilterComparisonSpec ─────────────────────────────────────────────────


class TestFilterComparisonSpec:
    """Test FilterComparisonSpec."""

    def test_defaults(self) -> None:
        fc = FilterComparisonSpec()
        assert fc.show_difference is False
        assert fc.difference_color == "#ff6b6b"
        assert isinstance(fc, PlotSpec)
