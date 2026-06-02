from __future__ import annotations

from types import SimpleNamespace

import plot_engine.plotly_converter as converter_module
import pytest
from plot_engine.plotly_converter import PlotlyConverter
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


def make_series(
    name: str = "signal",
    *,
    style: SeriesStyle | None = None,
    trendline: TrendlineSpec | None = None,
    x: list[float] | None = None,
    y: list[float] | None = None,
) -> SeriesData:
    return SeriesData(
        name=name,
        x=[0.0, 1.0, 2.0] if x is None else x,
        y=[1.0, 3.0, 5.0] if y is None else y,
        style=SeriesStyle() if style is None else style,
        trendline=trendline,
    )


def test_line_scatter_converts_styled_series_and_trendline() -> None:
    converter = PlotlyConverter()
    series = make_series(
        style=SeriesStyle(
            color="#123456",
            line_style="dashdot",
            line_width=2.5,
            marker="star",
            marker_size=9.0,
            opacity=0.5,
            display_mode="line+scatter",
        ),
        trendline=TrendlineSpec(
            type="linear",
            color="#654321",
            line_style="dotted",
        ),
    )
    spec = PlotSpec(
        title="Signal",
        series=[series],
        x_axis=AxisSpec(
            label="Time",
            min=0.0,
            max=2.0,
            log_scale=True,
            grid=False,
        ),
        y_axis=AxisSpec(label="Value", min=1.0, max=5.0),
        legend=LegendSpec(position="top"),
        width=500,
        height=300,
    )

    result = converter.convert(spec)

    data_trace, trend_trace = result["data"]
    assert data_trace == {
        "type": "scatter",
        "x": [0.0, 1.0, 2.0],
        "y": [1.0, 3.0, 5.0],
        "name": "signal",
        "mode": "lines+markers",
        "line": {"width": 2.5, "dash": "dashdot", "color": "#123456"},
        "marker": {"size": 9.0, "symbol": "star", "color": "#123456"},
        "opacity": 0.5,
    }
    assert trend_trace["type"] == "scatter"
    assert trend_trace["mode"] == "lines"
    assert trend_trace["line"] == {
        "dash": "dot",
        "width": 1.5,
        "color": "#654321",
    }
    assert "y =" in trend_trace["name"]
    assert "R\u00b2=" in trend_trace["name"]
    assert len(trend_trace["x"]) == 200
    assert len(trend_trace["y"]) == 200
    assert result["layout"] == {
        "title": {"text": "Signal"},
        "width": 500,
        "height": 300,
        "xaxis": {
            "title": {"text": "Time"},
            "range": [0.0, 2.0],
            "type": "log",
            "showgrid": False,
        },
        "yaxis": {
            "title": {"text": "Value"},
            "range": [1.0, 5.0],
            "showgrid": True,
        },
        "showlegend": True,
        "legend": {
            "x": 0.5,
            "y": 1.1,
            "xanchor": "center",
            "orientation": "h",
        },
    }


def test_line_scatter_skips_failed_trendline(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_trendline(*_args: object, **_kwargs: object) -> object:
        raise ValueError("bad fit")

    monkeypatch.setattr(converter_module, "compute_trendline", fail_trendline)
    converter = PlotlyConverter()
    series = make_series("failing", trendline=TrendlineSpec(type="linear"))

    result = converter.convert(PlotSpec(series=[series]))

    assert len(result["data"]) == 1
    assert result["data"][0]["name"] == "failing"
    assert "Trendline failed for failing: bad fit" in caplog.text


def test_trendline_trace_uses_default_name_when_labels_are_hidden() -> None:
    converter = PlotlyConverter()
    series = make_series(
        trendline=TrendlineSpec(
            type="linear",
            show_equation=False,
            show_r_squared=False,
        )
    )

    assert converter._trendline_trace(make_series()) is None

    trace = converter._trendline_trace(series)

    assert trace is not None
    assert trace["name"] == "Trend: signal"
    assert trace["line"] == {"dash": "dash", "width": 1.5}


def test_surface_converter_adds_wireframe_scatter_and_scene_axes() -> None:
    spec = SurfacePlotSpec(
        title="Surface",
        z_data=[[1.0, 2.0], [3.0, 4.0]],
        x_grid=[0.0, 1.0],
        y_grid=[10.0, 20.0],
        z_axis=AxisSpec(label="Height"),
        colormap="Plasma",
        opacity=0.7,
        show_wireframe=True,
        show_scatter=True,
    )

    result = PlotlyConverter().convert(spec)

    surface, scatter = result["data"]
    assert surface["type"] == "surface"
    assert surface["z"] == [[1.0, 2.0], [3.0, 4.0]]
    assert surface["x"] == [0.0, 1.0]
    assert surface["y"] == [10.0, 20.0]
    assert surface["colorscale"] == "Plasma"
    assert surface["opacity"] == 0.7
    assert surface["contours"]["x"] == {"show": True, "color": "gray", "width": 1}
    assert scatter["type"] == "scatter3d"
    assert scatter["x"] == [0.0, 1.0, 0.0, 1.0]
    assert scatter["y"] == [10.0, 10.0, 20.0, 20.0]
    assert scatter["z"] == [1.0, 2.0, 3.0, 4.0]
    assert result["layout"]["scene"]["zaxis"] == {
        "title": {"text": "Height"},
        "showgrid": True,
    }


def test_contour_converter_handles_filled_labels_and_colorbar() -> None:
    spec = ContourPlotSpec(
        z_data=[[1.0, 2.0], [3.0, 4.0]],
        x_grid=[0.0, 1.0],
        y_grid=[0.0, 1.0],
        levels=8,
        filled=True,
        colormap="Viridis",
        show_colorbar=False,
        show_labels=True,
    )

    result = PlotlyConverter().convert(spec)

    trace = result["data"][0]
    assert trace == {
        "type": "contour",
        "z": [[1.0, 2.0], [3.0, 4.0]],
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "colorscale": "Viridis",
        "ncontours": 8,
        "contours": {"coloring": "heatmap", "showlabels": True},
        "showscale": False,
    }


def test_heatmap_converter_adds_labels_and_annotations() -> None:
    spec = HeatmapSpec(
        z_data=[[1.234, 2.0], [3.5, 4.25]],
        x_labels=["a", "b"],
        y_labels=["c", "d"],
        annotate=True,
        show_colorbar=False,
    )

    result = PlotlyConverter().convert(spec)

    trace = result["data"][0]
    assert trace["type"] == "heatmap"
    assert trace["z"] == [[1.234, 2.0], [3.5, 4.25]]
    assert trace["x"] == ["a", "b"]
    assert trace["y"] == ["c", "d"]
    assert trace["showscale"] is False
    assert trace["text"] == [["1.23", "2.00"], ["3.50", "4.25"]]
    assert trace["texttemplate"] == "%{text}"
    assert trace["hoverinfo"] == "z"


def test_histogram_converter_handles_styling_and_distribution_modes() -> None:
    spec = HistogramSpec(
        series=[make_series(style=SeriesStyle(color="#abcdef"))],
        bins=12,
        density=True,
        cumulative=True,
        stacked=True,
    )

    result = PlotlyConverter().convert(spec)

    trace = result["data"][0]
    assert trace == {
        "type": "histogram",
        "x": [1.0, 3.0, 5.0],
        "name": "signal",
        "nbinsx": 12,
        "marker": {"color": "#abcdef"},
        "histnorm": "probability density",
        "cumulative": {"enabled": True},
    }
    assert result["layout"]["barmode"] == "stack"


def test_filter_comparison_adds_prefixed_and_difference_traces() -> None:
    original = make_series("raw", x=[0.0, 1.0, 2.0], y=[2.0, 4.0, 6.0])
    filtered = make_series(
        "raw",
        x=[0.0, 1.0, 2.0],
        y=[1.0, 3.0, 4.0],
        style=SeriesStyle(display_mode="scatter"),
    )
    spec = FilterComparisonSpec(
        original_series=[original],
        filtered_series=[filtered],
        show_difference=True,
        difference_color="#ff0000",
    )

    result = PlotlyConverter().convert(spec)

    original_trace, filtered_trace, diff_trace = result["data"]
    assert original_trace["name"] == "Original: raw"
    assert filtered_trace["name"] == "Filtered: raw"
    assert filtered_trace["line"]["dash"] == "dash"
    assert diff_trace == {
        "type": "scatter",
        "x": [0.0, 1.0, 2.0],
        "y": [1.0, 1.0, 2.0],
        "name": "Diff: raw",
        "mode": "lines",
        "line": {"color": "#ff0000", "width": 1},
        "yaxis": "y2",
    }
    assert result["layout"]["yaxis2"] == {
        "title": "Difference",
        "overlaying": "y",
        "side": "right",
    }


@pytest.mark.parametrize(
    ("method_name", "message"),
    [
        ("_line_scatter", "spec must be provided"),
        ("_surface", "spec must be provided"),
        ("_contour", "spec must be provided"),
        ("_heatmap", "spec must be provided"),
        ("_histogram", "spec must be provided"),
        ("_filter_comparison", "spec must be provided"),
        ("_series_trace", "series must be provided"),
        ("_trendline_trace", "series must be provided"),
        ("_build_layout", "spec must be provided"),
    ],
)
def test_converter_helpers_validate_required_inputs(
    method_name: str,
    message: str,
) -> None:
    converter = PlotlyConverter()
    method = getattr(converter, method_name)

    with pytest.raises(ValueError, match=message):
        method(None)


def test_axis_legend_and_display_mode_helpers_cover_defaults() -> None:
    converter = PlotlyConverter()

    assert converter._axis_dict(AxisSpec()) == {"showgrid": True}
    assert converter._axis_dict(
        AxisSpec(label="X", min=1.0, max=9.0, log_scale=True, grid=False)
    ) == {
        "title": {"text": "X"},
        "range": [1.0, 9.0],
        "type": "log",
        "showgrid": False,
    }
    assert converter._legend_dict(LegendSpec(position="right")) == {
        "x": 1.02,
        "y": 1,
        "xanchor": "left",
    }
    assert converter._legend_dict(LegendSpec(position="left")) == {
        "x": -0.15,
        "y": 1,
        "xanchor": "right",
    }
    assert converter._legend_dict(LegendSpec(position="bottom")) == {
        "x": 0.5,
        "y": -0.15,
        "xanchor": "center",
        "orientation": "h",
    }
    assert converter._legend_dict(SimpleNamespace(position="unknown")) == {}
    assert converter._display_mode_to_plotly("line") == "lines"
    assert converter._display_mode_to_plotly("scatter") == "markers"
    assert converter._display_mode_to_plotly("line+scatter") == "lines+markers"
    assert converter._display_mode_to_plotly("unknown") == "lines"


def test_build_layout_hides_absent_or_disabled_legend() -> None:
    converter = PlotlyConverter()

    none_position = converter._build_layout(
        PlotSpec(legend=LegendSpec(position="none"))
    )
    hidden = converter._build_layout(PlotSpec(legend=LegendSpec(visible=False)))

    assert none_position["showlegend"] is False
    assert hidden["showlegend"] is False
