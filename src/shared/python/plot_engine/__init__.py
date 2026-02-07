"""Shared plot engine for cross-platform visualization.

This module provides a DRY plotting system that renders the same PlotSpec
contracts to both matplotlib (PyQt6) and Plotly.js JSON (React/Tauri).

Usage:
    from shared.python.plot_engine.specs import PlotSpec, SeriesData
    from shared.python.plot_engine.matplotlib_renderer import MatplotlibRenderer
    from shared.python.plot_engine.plotly_converter import PlotlyConverter

    spec = PlotSpec(title="My Plot", series=[...])

    # PyQt6 path
    fig = MatplotlibRenderer().render(spec)

    # React/Tauri path
    plotly_json = PlotlyConverter().convert(spec)
"""

from .specs import (
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

__all__ = [
    "AxisSpec",
    "ContourPlotSpec",
    "FilterComparisonSpec",
    "HeatmapSpec",
    "HistogramSpec",
    "LegendSpec",
    "PlotSpec",
    "SeriesData",
    "SeriesStyle",
    "SurfacePlotSpec",
    "TrendlineSpec",
]
