"""The one plot pipeline: PlotSpec + SimulationRun -> data -> figure.

``compute_plot_data`` turns a definition and a reference run into a
:class:`PlotData` (X array plus labelled Y series). Line, scatter, and
histogram kinds read per-sample series straight off the run; the
``sweep`` kind rebuilds the simulation config with the swept Input
variable at each grid value and re-runs the full swing → impact →
flight pipeline, extracting the scalar Y outputs per point.

``render_plot`` draws a PlotData onto a matplotlib figure using the
shared theme palette (``get_chart_color``) — no hard-coded colors.
``write_plot_csv`` / ``write_plot_json`` export exactly what was
plotted, for reproducible investigations.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from rate_of_closure._contracts import ensure, require
from rate_of_closure.plot_point_inspector import (
    MAX_ABS_PLOT_VALUE,
    MAX_PLOT_SAMPLES,
    MAX_PLOT_SERIES,
    MAX_PLOT_VERTICES,
    plan_plot_inspection,
)
from rate_of_closure.plotting.catalog import CATALOG, DISTANCE_KEYS, extract
from rate_of_closure.plotting.spec import PlotSpec
from rate_of_closure.simulation.session import (
    SimulationConfig,
    SimulationRun,
    run_simulation,
)
from rate_of_closure.units import DISTANCE_UNITS, display_distance_unit

# ── Theme integration (optional — graceful fallback) ───────────────
try:
    from shared.python.theme.matplotlib_style import get_chart_color
except ImportError:  # standalone / vendored use
    _FALLBACK_COLORS = ("#0A84FF", "#FF9F0A", "#30D158", "#FF375F", "#BF5AF2")

    def get_chart_color(index: int) -> str:
        """Theme-neutral palette stand-in matching the shared hues."""
        return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


__all__ = [
    "PlotData",
    "compute_plot_data",
    "get_chart_color",
    "plot_data_rows",
    "render_plot",
    "write_plot_csv",
    "write_plot_json",
]

#: Sweepable Input key -> how it lands in a new SimulationConfig.
_SCENARIO_FIELDS = {
    "input.clubhead_speed_mph": "clubhead_speed_mph",
    "input.omega_plane_dps": "omega_plane_dps",
    "input.omega_shaft_dps": "omega_shaft_dps",
    "input.lie_angle_deg": "lie_angle_deg",
    "input.com_to_face_mm": "com_to_face_mm",
    "input.impact_offset_toe_mm": "impact_offset_toe_mm",
    "input.impact_offset_high_mm": "impact_offset_high_mm",
    "input.contact_duration_us": "contact_duration_us",
}
_PLANE_FIELDS = {
    "input.plane_yaw_deg": "yaw_deg",
    "input.plane_side_tilt_deg": "side_tilt_deg",
    "input.plane_forward_tilt_deg": "forward_tilt_deg",
}


@dataclass(frozen=True)
class PlotData:
    """The numbers behind one rendered plot.

    Attributes:
        spec: The definition that produced this data.
        x: (N,) X values.
        series: Y-series label -> (N,) values, in legend order.
        x_label: X axis label with unit.
        y_label: Y axis label with unit (shared unit or generic).
    """

    spec: PlotSpec
    x: np.ndarray
    series: Mapping[str, np.ndarray]
    x_label: str
    y_label: str

    def __post_init__(self) -> None:
        x_values = np.asarray(self.x, dtype=np.float64)
        require(
            x_values.ndim == 1 and 1 <= x_values.size <= MAX_PLOT_SAMPLES,
            f"plot evidence must contain 1..{MAX_PLOT_SAMPLES} samples",
        )
        require(
            np.all(np.isfinite(x_values))
            and np.all(np.abs(x_values) <= MAX_ABS_PLOT_VALUE),
            "plot x values must be finite and bounded",
        )
        require(
            len(self.series) <= MAX_PLOT_SERIES,
            f"plot supports at most {MAX_PLOT_SERIES} series",
        )
        require(
            x_values.size * len(self.series) <= MAX_PLOT_VERTICES,
            f"plot exceeds {MAX_PLOT_VERTICES} vertices",
        )
        checked: dict[str, np.ndarray] = {}
        for label, raw_values in self.series.items():
            require(
                isinstance(label, str) and 1 <= len(label) <= 512,
                "plot series label must contain 1..512 characters",
            )
            values = np.asarray(raw_values, dtype=np.float64)
            require(values.shape == x_values.shape, f"series {label!r} must match x")
            finite = np.isfinite(values)
            require(
                np.all(np.isnan(values) | finite)
                and np.all(np.abs(values[finite]) <= MAX_ABS_PLOT_VALUE),
                f"series {label!r} must contain bounded values or NaN gaps",
            )
            checked[label] = np.frombuffer(values.tobytes(), dtype=np.float64)
        x = np.frombuffer(x_values.tobytes(), dtype=np.float64)
        series = MappingProxyType(checked)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "series", series)


def _config_with(config: SimulationConfig, key: str, value: float) -> SimulationConfig:
    """A copy of ``config`` with one Input variable replaced."""
    if key in _SCENARIO_FIELDS:
        scenario = dataclasses.replace(
            config.scenario, **{_SCENARIO_FIELDS[key]: float(value)}
        )
        return dataclasses.replace(config, scenario=scenario)
    if key in _PLANE_FIELDS:
        plane = dataclasses.replace(config.plane, **{_PLANE_FIELDS[key]: float(value)})
        return dataclasses.replace(config, plane=plane)
    require(key == "input.impact_time_s", f"cannot sweep {key!r}", key)
    return dataclasses.replace(config, impact_time_s=float(value))


def _display_factor(key: str) -> float:
    """Canonical -> display divide factor for one catalog variable.

    Ball-flight distance variables (#4125 H6) follow the session's
    Distance display unit (yards default); everything else is 1.
    """
    if key in DISTANCE_KEYS:
        return float(DISTANCE_UNITS[display_distance_unit()])
    return 1.0


def _display_unit(key: str) -> str:
    """The display unit string for one catalog variable."""
    if key in DISTANCE_KEYS:
        return str(display_distance_unit())
    return str(CATALOG[key].unit)


def _axis_label(key: str) -> str:
    """Axis label in the display unit (#4125 H6)."""
    unit = _display_unit(key)
    label = str(CATALOG[key].label)
    return f"{label} [{unit}]" if unit else label


def _shared_y_label(y_keys: tuple[str, ...]) -> str:
    units = {_display_unit(key) for key in y_keys}
    if len(y_keys) == 1:
        return _axis_label(y_keys[0])
    if len(units) == 1:
        return f"Value [{units.pop()}]" if "" not in units else "Value"
    return "Value (Mixed Units)"


def _sweep_data(spec: PlotSpec, run: SimulationRun) -> PlotData:
    assert spec.x_start is not None and spec.x_stop is not None
    grid = np.linspace(spec.x_start, spec.x_stop, spec.x_count)
    columns: dict[str, list[float]] = {key: [] for key in spec.y_keys}
    kept: list[float] = []
    for value in grid:
        try:
            point = run_simulation(_config_with(run.config, spec.x_key, float(value)))
        except Exception:  # noqa: BLE001 — skip infeasible sweep points
            continue
        kept.append(float(value))
        for key in spec.y_keys:
            columns[key].append(float(extract(point, key)))
    ensure(len(kept) >= 2, "sweep produced fewer than 2 feasible points")
    return PlotData(
        spec=spec,
        x=np.asarray(kept, dtype=float),
        series={
            CATALOG[key].label: (
                np.asarray(columns[key], dtype=float) / _display_factor(key)
            )
            for key in spec.y_keys
        },
        x_label=_axis_label(spec.x_key),
        y_label=_shared_y_label(spec.y_keys),
    )


def compute_plot_data(spec: PlotSpec, run: SimulationRun) -> PlotData:
    """Evaluate a plot definition against a reference run.

    Args:
        spec: The plot definition.
        run: The reference simulation run (sweeps rebuild its config).

    Returns:
        The plottable / exportable data.
    """
    require(isinstance(spec, PlotSpec), "spec must be a PlotSpec", spec)
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    if spec.kind == "sweep":
        return _sweep_data(spec, run)
    x = np.asarray(extract(run, spec.x_key), dtype=float) / _display_factor(spec.x_key)
    if spec.kind == "histogram":
        return PlotData(
            spec=spec,
            x=x,
            series={},
            x_label=_axis_label(spec.x_key),
            y_label="Count",
        )
    series = {
        CATALOG[key].label: (
            np.asarray(extract(run, key), dtype=float) / _display_factor(key)
        )
        for key in spec.y_keys
    }
    return PlotData(
        spec=spec,
        x=x,
        series=series,
        x_label=_axis_label(spec.x_key),
        y_label=_shared_y_label(spec.y_keys),
    )


def render_plot(data: PlotData, figure: Figure) -> None:
    """Draw the data onto a (cleared) matplotlib figure, themed.

    Args:
        data: Output of :func:`compute_plot_data`.
        figure: Destination figure; existing axes are removed.
    """
    require(isinstance(data, PlotData), "data must be a PlotData", data)
    figure.clear()
    axes = figure.add_subplot(111)
    spec = data.spec
    if spec.kind == "histogram":
        plan = plan_plot_inspection("histogram", data.x, [])
        edges = [plan.bins[0].lower, *(item.upper for item in plan.bins)]
        axes.hist(
            data.x,
            bins=edges,
            color=get_chart_color(0),
            alpha=0.85,
        )
    else:
        for index, (label, values) in enumerate(data.series.items()):
            color = get_chart_color(index)
            if spec.kind == "scatter":
                axes.scatter(data.x, values, s=14, color=color, label=label)
            else:  # line and sweep
                axes.plot(data.x, values, lw=1.8, color=color, label=label)
        if len(data.series) >= 1:
            axes.legend(loc="best", fontsize=8)
    if spec.x_log:
        axes.set_xscale("log")
    if spec.y_log:
        axes.set_yscale("log")
    axes.set_xlabel(data.x_label)
    axes.set_ylabel(data.y_label)
    axes.set_title(spec.title or CATALOG[spec.x_key].label)
    axes.grid(alpha=0.25)


def plot_data_rows(data: PlotData) -> tuple[list[str], list[list[float]]]:
    """Header + rows of the plotted numbers (CSV/JSON payload)."""
    header = [data.x_label, *data.series]
    columns = [data.x, *data.series.values()]
    rows = [[float(column[i]) for column in columns] for i in range(data.x.size)]
    return header, rows


def write_plot_csv(data: PlotData, path: str | Path) -> None:
    """Write the plotted data as CSV.

    Args:
        data: Output of :func:`compute_plot_data`.
        path: Destination file path.
    """
    header, rows = plot_data_rows(data)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_plot_json(data: PlotData, path: str | Path) -> None:
    """Write the plotted data + its definition as a JSON document.

    Args:
        data: Output of :func:`compute_plot_data`.
        path: Destination file path.
    """
    header, rows = plot_data_rows(data)

    def _clean(value: float) -> float | None:
        return value if math.isfinite(value) else None

    payload: dict[str, Any] = {
        "format": "rate_of_closure.plot_data/1",
        "spec": data.spec.to_json_dict(),
        "columns": header,
        "rows": [[_clean(v) for v in row] for row in rows],
    }
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
