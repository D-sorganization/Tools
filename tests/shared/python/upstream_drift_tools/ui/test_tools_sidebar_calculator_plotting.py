"""Tests for Sidekick calculator plot request contracts."""

from __future__ import annotations

import sys

import pytest
from upstream_drift_tools.ui.tools_sidebar import WorkspaceRegistry
from upstream_drift_tools.ui.tools_sidebar.calculator_plotting import (
    CalculatorPlotRequest,
    CalculatorPlotSource,
    CalculatorPlotTabConfig,
    build_calculator_plot_spec,
)


def test_workspace_xy_plot_request_validates_lengths_and_uses_plot_spec() -> None:
    registry = WorkspaceRegistry({"x_values": [0, 1, 2], "y_values": [0, 1]})
    request = CalculatorPlotRequest(
        source=CalculatorPlotSource.WORKSPACE_XY,
        x_ref="x_values",
        y_ref="y_values",
    )

    with pytest.raises(ValueError, match="same length"):
        build_calculator_plot_spec(request, registry)

    registry.set("y_values", [0, 1, 4])
    spec = build_calculator_plot_spec(request, registry)

    assert spec.series[0].x == [0.0, 1.0, 2.0]
    assert spec.series[0].y == [0.0, 1.0, 4.0]
    assert spec.legend.visible is True


def test_expression_range_plot_does_not_mutate_workspace() -> None:
    registry = WorkspaceRegistry({"x": [100], "calculator_result": 9})
    request = CalculatorPlotRequest(
        source=CalculatorPlotSource.EXPRESSION_RANGE,
        expression="x**2 + 1",
        x_min=0,
        x_max=2,
        points=3,
    )

    before = registry.to_dict()
    spec = build_calculator_plot_spec(request, registry)

    assert registry.to_dict() == before
    assert spec.series[0].x == [0.0, 1.0, 2.0]
    assert spec.series[0].y == [1.0, 2.0, 5.0]
    assert spec.title == "x**2 + 1"


def test_large_workspace_arrays_are_referenced_and_sampled() -> None:
    registry = WorkspaceRegistry({"series": list(range(20))})
    request = CalculatorPlotRequest(
        source=CalculatorPlotSource.WORKSPACE_RESULT,
        y_ref="series",
        config=CalculatorPlotTabConfig(max_points=5),
    )

    spec = build_calculator_plot_spec(request, registry)

    assert request.serializable_config()["data_refs"] == {"y": "series"}
    assert spec.series[0].x == [0.0, 4.0, 8.0, 12.0, 16.0]
    assert spec.series[0].y == [0.0, 4.0, 8.0, 12.0, 16.0]


def test_plot_theme_defaults_to_sidekick_inheritance() -> None:
    request = CalculatorPlotRequest(source=CalculatorPlotSource.WORKSPACE_RESULT)

    assert request.config.theme == "inherit"
    assert request.serializable_config()["theme"] == "inherit"


def test_calculator_plot_contract_imports_without_qt_or_matplotlib_backend() -> None:
    loaded = {
        name.partition(".")[0]
        for name in sys.modules
        if name.partition(".")[0] in {"PyQt6", "PySide6", "PyQt5", "PySide2"}
    }

    from upstream_drift_tools.ui.tools_sidebar import calculator_plotting

    assert calculator_plotting.CalculatorPlotRequest is CalculatorPlotRequest
    assert loaded == {
        name.partition(".")[0]
        for name in sys.modules
        if name.partition(".")[0] in {"PyQt6", "PySide6", "PyQt5", "PySide2"}
    }
