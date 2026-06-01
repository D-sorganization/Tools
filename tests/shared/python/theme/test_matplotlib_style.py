"""Focused coverage for shared matplotlib theme helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt

from src.shared.python.theme import matplotlib_style
from src.shared.python.theme.colors import CHART_COLORS


def test_apply_plot_theme_styles_axes_legend_and_canvas(monkeypatch):
    fig, ax = plt.subplots()
    draw_calls: list[bool] = []
    monkeypatch.setattr(fig.canvas, "draw_idle", lambda: draw_calls.append(True))

    ax.plot([0, 1], [1, 0], label="trend")
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_title("Result")
    legend = ax.legend()

    theme = {
        "bg": "#111111",
        "group_bg": "#222222",
        "border": "#333333",
        "text": "#444444",
        "text_secondary": "#555555",
    }

    matplotlib_style.apply_plot_theme(fig, theme)

    assert fig.get_facecolor() == mpl.colors.to_rgba("#111111")
    assert ax.get_facecolor() == mpl.colors.to_rgba("#222222")
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.spines["left"].get_edgecolor() == mpl.colors.to_rgba("#333333")
    assert ax.spines["bottom"].get_linewidth() == 1.0
    assert ax.xaxis.label.get_color() == "#444444"
    assert ax.yaxis.label.get_color() == "#444444"
    assert ax.title.get_color() == "#444444"
    assert legend.get_frame().get_facecolor() == mpl.colors.to_rgba("#222222", 0.9)
    assert legend.get_texts()[0].get_color() == "#444444"
    assert draw_calls == [True]
    plt.close(fig)


def test_apply_plot_theme_uses_defaults_for_missing_theme_keys(monkeypatch):
    fig, ax = plt.subplots()
    monkeypatch.setattr(fig.canvas, "draw_idle", lambda: None)

    matplotlib_style.apply_plot_theme(fig, {})

    assert fig.get_facecolor() == mpl.colors.to_rgba("#ffffff")
    assert ax.get_facecolor() == mpl.colors.to_rgba("#f8f9fa")
    assert ax.spines["left"].get_edgecolor() == mpl.colors.to_rgba("#ced4da")
    assert ax.xaxis.label.get_color() == "#212529"
    plt.close(fig)


def test_apply_plot_theme_handles_figures_without_axes(monkeypatch):
    fig = plt.figure()
    draw_calls: list[bool] = []
    monkeypatch.setattr(fig.canvas, "draw_idle", lambda: draw_calls.append(True))

    matplotlib_style.apply_plot_theme(fig, {"bg": "#abcdef"})

    assert fig.get_facecolor() == mpl.colors.to_rgba("#abcdef")
    assert draw_calls == [True]
    plt.close(fig)


def test_global_style_chart_color_and_styled_figure():
    matplotlib_style.apply_global_style()

    assert mpl.rcParams["axes.grid"] is True
    assert mpl.rcParams["lines.linewidth"] == 1.5
    assert matplotlib_style.get_chart_color(0) == CHART_COLORS[0]
    assert matplotlib_style.get_chart_color(len(CHART_COLORS)) == CHART_COLORS[0]
    assert matplotlib_style.get_chart_color(-1) == CHART_COLORS[-1]

    fig, axes = matplotlib_style.create_styled_figure(nrows=2, ncols=1)

    assert tuple(fig.get_size_inches()) == (10.0, 6.0)
    assert len(axes) == 2
    plt.close(fig)
