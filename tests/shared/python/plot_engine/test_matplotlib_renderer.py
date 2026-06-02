from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import plot_engine.matplotlib_renderer as renderer_module
import pytest
from plot_engine.matplotlib_renderer import MatplotlibRenderer
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


class FakeThemeManager:
    def __init__(self) -> None:
        self.applied_figures: list[Any] = []

    def get_colors(self) -> dict[str, Any]:
        return {
            "primary_colors": ["#111111", "#222222"],
            "secondary_colors": ["#333333"],
            "contour_cmap": "plasma",
            "heatmap_cmap": "magma",
        }

    def apply_to_figure(self, fig: Any) -> None:
        self.applied_figures.append(fig)


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def make_series(
    name: str = "signal",
    *,
    x: list[float] | None = None,
    y: list[float] | None = None,
    style: SeriesStyle | None = None,
    trendline: TrendlineSpec | None = None,
) -> SeriesData:
    return SeriesData(
        name=name,
        x=[1.0, 2.0, 3.0] if x is None else x,
        y=[2.0, 4.0, 6.0] if y is None else y,
        style=SeriesStyle() if style is None else style,
        trendline=trendline,
    )


def test_render_line_plot_applies_styles_trendline_theme_and_axes() -> None:
    theme = FakeThemeManager()
    renderer = MatplotlibRenderer(theme)
    spec = PlotSpec(
        title="Line Contract",
        series=[
            make_series(
                "line",
                style=SeriesStyle(
                    line_style="dashdot",
                    line_width=2.5,
                    display_mode="line",
                ),
                trendline=TrendlineSpec(
                    type="linear",
                    color="#abcdef",
                    line_style="dotted",
                ),
            ),
            make_series(
                "scatter",
                y=[1.0, 3.0, 9.0],
                style=SeriesStyle(
                    color="#123456",
                    marker="diamond",
                    marker_size=4.0,
                    opacity=0.4,
                    display_mode="scatter",
                ),
            ),
            make_series(
                "combo",
                y=[3.0, 5.0, 7.0],
                style=SeriesStyle(
                    marker="star",
                    marker_size=5.0,
                    display_mode="line+scatter",
                ),
            ),
        ],
        x_axis=AxisSpec(label="Time", min=1.0, max=3.0, log_scale=True),
        y_axis=AxisSpec(label="Value", min=1.0, max=10.0, log_scale=True),
        legend=LegendSpec(
            position="left",
            labels={"line": "Line renamed", "scatter": "Scatter renamed"},
        ),
        width=500,
        height=300,
    )

    fig = renderer.render(spec)
    ax = fig.axes[0]

    assert fig in theme.applied_figures
    assert fig.get_size_inches().tolist() == [5.0, 3.0]
    assert ax.get_title() == "Line Contract"
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Value"
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert ax.get_xlim() == pytest.approx((1.0, 3.0))
    assert ax.get_ylim() == pytest.approx((1.0, 10.0))

    line, trendline, combo = ax.lines
    assert line.get_label() == "line"
    assert line.get_color() == "#111111"
    assert line.get_linestyle() == "-."
    assert line.get_linewidth() == 2.5
    assert trendline.get_color() == "#abcdef"
    assert trendline.get_linestyle() == ":"
    assert combo.get_marker() == "*"
    assert combo.get_markersize() == 5.0

    scatter = ax.collections[0]
    assert len(scatter.get_offsets()) == 3
    assert len(ax.texts) == 1
    assert "R\u00b2 =" in ax.texts[0].get_text()
    assert ax.get_legend() is not None
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Line renamed",
        "Scatter renamed",
        "combo",
    ]


def test_render_line_plot_skips_failed_and_empty_trendline(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_trendline(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("bad fit")

    monkeypatch.setattr(renderer_module, "compute_trendline", fail_trendline)
    renderer = MatplotlibRenderer()
    spec = PlotSpec(
        series=[
            make_series("no-trend"),
            make_series("bad-trend", trendline=TrendlineSpec(type="linear")),
        ],
        legend=LegendSpec(visible=False),
    )

    fig = renderer.render_line_plot(spec)

    assert len(fig.axes[0].lines) == 2
    assert fig.axes[0].get_legend() is None
    assert "Trendline computation failed for bad-trend: bad fit" in caplog.text
    renderer._render_trendline(fig.axes[0], make_series(), "#111111")


def test_render_surface_uses_theme_colormap_wireframe_and_scatter() -> None:
    renderer = MatplotlibRenderer(FakeThemeManager())
    spec = SurfacePlotSpec(
        title="Surface",
        z_data=[[1.0, 2.0], [3.0, 4.0]],
        x_grid=[0.0, 1.0],
        y_grid=[10.0, 20.0],
        x_axis=AxisSpec(label="X axis"),
        y_axis=AxisSpec(label="Y axis"),
        z_axis=AxisSpec(label="Z axis"),
        colormap="viridis",
        opacity=0.6,
        show_wireframe=True,
        show_scatter=True,
    )

    fig = renderer.render(spec)
    ax = fig.axes[0]

    assert ax.name == "3d"
    assert ax.get_title() == "Surface"
    assert ax.get_xlabel() == "X axis"
    assert ax.get_ylabel() == "Y axis"
    assert ax.get_zlabel() == "Z axis"
    assert len(ax.collections) == 2


def test_render_surface_reuses_figure_and_defaults_axes_without_scatter() -> None:
    renderer = MatplotlibRenderer()
    fig = plt.figure()
    spec = SurfacePlotSpec(
        z_data=[[1.0, 2.0], [3.0, 4.0]],
        x_grid=[0.0, 1.0],
        y_grid=[0.0, 1.0],
        show_wireframe=False,
        show_scatter=False,
    )

    result = renderer.render_surface(spec, fig=fig)

    assert result is fig
    ax = result.axes[0]
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"
    assert ax.get_zlabel() == "Z"
    assert len(ax.collections) == 1


def test_render_contour_covers_line_labels_and_colorbar() -> None:
    renderer = MatplotlibRenderer(FakeThemeManager())
    spec = ContourPlotSpec(
        title="Contour",
        z_data=[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
        x_grid=[0.0, 1.0, 2.0],
        y_grid=[0.0, 1.0, 2.0],
        x_axis=AxisSpec(label="x"),
        y_axis=AxisSpec(label="y"),
        levels=3,
        filled=False,
        show_labels=True,
        show_colorbar=True,
    )

    fig = renderer.render(spec)

    assert len(fig.axes) == 2
    ax = fig.axes[0]
    assert ax.get_title() == "Contour"
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"


def test_render_contour_covers_filled_without_labels_or_colorbar() -> None:
    spec = ContourPlotSpec(
        z_data=[[0.0, 1.0], [1.0, 2.0]],
        x_grid=[0.0, 1.0],
        y_grid=[0.0, 1.0],
        filled=True,
        show_labels=True,
        show_colorbar=False,
    )

    fig = MatplotlibRenderer().render_contour(spec)

    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == ""


def test_render_heatmap_adds_labels_annotations_and_colorbar() -> None:
    renderer = MatplotlibRenderer(FakeThemeManager())
    spec = HeatmapSpec(
        title="Heatmap",
        z_data=[[1.234, 2.0], [3.5, 4.25]],
        x_labels=["a", "b"],
        y_labels=["c", "d"],
        annotate=True,
        show_colorbar=True,
    )

    fig = renderer.render(spec)
    ax = fig.axes[0]

    assert len(fig.axes) == 2
    assert ax.get_title() == "Heatmap"
    assert [label.get_text() for label in ax.get_xticklabels()] == ["a", "b"]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["c", "d"]
    assert [text.get_text() for text in ax.texts] == [
        "1.23",
        "2.00",
        "3.50",
        "4.25",
    ]


def test_render_heatmap_covers_minimal_options() -> None:
    spec = HeatmapSpec(z_data=[[1.0]], annotate=False, show_colorbar=False)

    fig = MatplotlibRenderer().render_heatmap(spec)

    assert len(fig.axes) == 1
    assert not fig.axes[0].texts


def test_render_histogram_handles_data_styles_and_empty_input() -> None:
    renderer = MatplotlibRenderer()
    spec = HistogramSpec(
        title="Histogram",
        series=[
            make_series("styled", style=SeriesStyle(color="#123456")),
            make_series("cycled", y=[2.0, 2.5, 3.0]),
        ],
        x_axis=AxisSpec(label="bins"),
        y_axis=AxisSpec(label="density"),
        legend=LegendSpec(position="bottom"),
        bins=3,
        density=True,
        cumulative=True,
        stacked=True,
    )

    fig = renderer.render(spec)
    ax = fig.axes[0]

    assert ax.get_title() == "Histogram"
    assert ax.get_xlabel() == "bins"
    assert ax.get_ylabel() == "density"
    assert ax.get_legend() is not None
    assert len(ax.patches) > 0

    empty_fig = renderer.render_histogram(HistogramSpec(series=[]))
    assert len(empty_fig.axes[0].patches) == 0


def test_render_filter_comparison_adds_difference_subplot() -> None:
    renderer = MatplotlibRenderer()
    original = make_series("raw", y=[2.0, 4.0, 6.0])
    filtered = make_series("raw", y=[1.0, 3.0, 4.0])
    spec = FilterComparisonSpec(
        title="Filter",
        original_series=[original],
        filtered_series=[filtered],
        x_axis=AxisSpec(label="time"),
        show_difference=True,
        difference_color="#ff0000",
    )

    with pytest.warns(UserWarning, match="tight_layout"):
        fig = renderer.render(spec)

    main_ax, diff_ax = fig.axes
    assert [line.get_label() for line in main_ax.lines] == [
        "Original: raw",
        "Filtered: raw",
    ]
    assert main_ax.lines[1].get_linestyle() == "--"
    assert diff_ax.get_xlabel() == "time"
    assert diff_ax.get_ylabel() == "Difference"
    assert diff_ax.lines[0].get_ydata().tolist() == [1.0, 1.0, 2.0]
    assert diff_ax.lines[0].get_color() == "#ff0000"
    assert diff_ax.lines[1].get_linestyle() == ":"


def test_render_filter_comparison_reuses_figure_and_suppresses_tight_layout() -> None:
    renderer = MatplotlibRenderer()
    fig = plt.figure()

    def fail_tight_layout() -> None:
        raise ValueError("layout failed")

    fig.tight_layout = fail_tight_layout  # type: ignore[method-assign]
    spec = FilterComparisonSpec(
        original_series=[make_series("raw")],
        filtered_series=[
            make_series(
                "raw",
                style=SeriesStyle(line_style="dotted"),
            )
        ],
        show_difference=False,
    )

    result = renderer.render_filter_comparison(spec, fig=fig)

    assert result is fig
    assert len(result.axes) == 1
    assert result.axes[0].lines[1].get_linestyle() == ":"


def test_to_image_returns_png_bytes_and_closes_figure() -> None:
    renderer = MatplotlibRenderer()
    spec = PlotSpec(series=[make_series()])

    data = renderer.to_image(spec, fmt="png", dpi=80)

    assert data.startswith(b"\x89PNG")
    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    ("method_name", "message"),
    [
        ("render", "spec must be provided"),
        ("render_line_plot", "spec must be provided"),
        ("render_surface", "spec must be provided"),
        ("render_contour", "spec must be provided"),
        ("render_heatmap", "spec must be provided"),
        ("render_histogram", "spec must be provided"),
        ("render_filter_comparison", "spec must be provided"),
        ("to_image", "spec must be provided"),
    ],
)
def test_public_methods_validate_required_specs(
    method_name: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        getattr(MatplotlibRenderer(), method_name)(None)


def test_private_helpers_validate_inputs_and_cover_defaults() -> None:
    renderer = MatplotlibRenderer()
    fig, ax = plt.subplots()
    spec = PlotSpec(series=[make_series()], legend=LegendSpec(position="none"))

    with pytest.raises(ValueError, match="spec must be provided"):
        renderer._ensure_fig_ax(None, None, None)
    with pytest.raises(ValueError, match="colors must be provided"):
        renderer._cycle_color(None, 0)
    with pytest.raises(ValueError, match="ax must be provided"):
        renderer._plot_series(None, make_series(), "#111111")
    with pytest.raises(ValueError, match="ax must be provided"):
        renderer._render_trendline(None, make_series(), "#111111")
    with pytest.raises(ValueError, match="ax must be provided"):
        renderer._apply_axis_spec(None, spec)
    with pytest.raises(ValueError, match="ax must be provided"):
        renderer._apply_legend(None, spec)

    renderer._plot_series(
        ax,
        make_series(style=SeriesStyle.model_construct(display_mode="unknown")),
        "#111111",
    )
    renderer._render_trendline(
        ax,
        make_series(
            trendline=TrendlineSpec(
                type="linear",
                show_equation=False,
                show_r_squared=False,
            )
        ),
        "#111111",
    )

    assert renderer._cycle_color([], 10) == "#1f77b4"
    assert renderer._cycle_color(["#111111", "#222222"], 3) == "#222222"
    assert renderer._get_theme_colors()[0] == "#1f77b4"

    ensured_fig, ensured_ax = renderer._ensure_fig_ax(fig, None, spec)
    assert ensured_fig is fig
    assert ensured_ax in fig.axes
    same_fig, same_ax = renderer._ensure_fig_ax(fig, ax, spec)
    assert same_fig is fig
    assert same_ax is ax

    renderer._apply_legend(ax, spec)
    assert ax.get_legend() is None


def test_apply_legend_covers_no_handles_custom_labels_and_unknown_position() -> None:
    renderer = MatplotlibRenderer()
    fig, ax = plt.subplots()
    renderer._apply_legend(
        ax,
        SimpleNamespace(
            legend=SimpleNamespace(visible=True, position="right", labels={})
        ),
    )
    assert ax.get_legend() is None

    ax.plot([1.0, 2.0], [2.0, 3.0], label="raw")
    renderer._apply_legend(
        ax,
        SimpleNamespace(
            legend=SimpleNamespace(
                visible=True,
                position="unexpected",
                labels={"raw": "renamed"},
            )
        ),
    )

    assert ax.get_legend() is not None
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["renamed"]
