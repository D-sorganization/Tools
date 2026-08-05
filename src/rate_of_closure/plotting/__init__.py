"""Investigative plotting suite (epic #4120, phase V1).

A data catalog describing every plottable variable of a
:class:`~rate_of_closure.simulation.session.SimulationRun`, frozen
JSON-round-trippable :class:`PlotSpec` definitions, built-in advanced
plots (sweeps, time series, flight profiles), and one compute/render
pipeline shared by the PyQt6 Plots tab and the web clone.
"""

from rate_of_closure.plotting.builtins import BUILTIN_PLOTS, builtin_spec
from rate_of_closure.plotting.catalog import (
    CATALOG,
    CATEGORIES,
    VariableSpec,
    catalog_keys,
    extract,
    variables_by_category,
)
from rate_of_closure.plotting.putting_catalog import (
    PUTTING_CATALOG,
    PuttingVariableSpec,
    extract_putting,
    putting_catalog_keys,
)
from rate_of_closure.plotting.render import (
    PlotData,
    compute_plot_data,
    plot_data_rows,
    render_plot,
    write_plot_csv,
    write_plot_json,
)
from rate_of_closure.plotting.spec import (
    PLOT_KINDS,
    PlotSpec,
    spec_from_json,
    spec_to_json,
)

__all__ = [
    "BUILTIN_PLOTS",
    "CATALOG",
    "CATEGORIES",
    "PLOT_KINDS",
    "PUTTING_CATALOG",
    "PlotData",
    "PlotSpec",
    "PuttingVariableSpec",
    "VariableSpec",
    "builtin_spec",
    "catalog_keys",
    "compute_plot_data",
    "extract",
    "extract_putting",
    "putting_catalog_keys",
    "plot_data_rows",
    "render_plot",
    "spec_from_json",
    "spec_to_json",
    "variables_by_category",
    "write_plot_csv",
    "write_plot_json",
]
