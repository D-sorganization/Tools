"""Unit tests for ``tools_sidebar.calculator_plotting``.

Qt-free module that converts validated calculator/workspace plot requests into
the shared ``plot_engine`` PlotSpec contract. Tests cover the three request
sources end-to-end, the restricted-namespace expression evaluator, numeric
series coercion, downsampling, and every validation guard.
"""

from __future__ import annotations

import pytest
from sidekick.ui.tools_sidebar.calculator_plotting import (
    CalculatorPlotRequest,
    CalculatorPlotSource,
    CalculatorPlotTabConfig,
    build_calculator_plot_spec,
)
from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

# ---------------------------------------------------------------------------
# CalculatorPlotTabConfig
# ---------------------------------------------------------------------------


def test_config_defaults_and_to_dict() -> None:
    config = CalculatorPlotTabConfig()
    assert config.to_dict() == {"theme": "inherit", "max_points": 1000}


def test_config_blank_theme_raises() -> None:
    with pytest.raises(ValueError, match="theme must be non-empty"):
        CalculatorPlotTabConfig(theme="   ")


@pytest.mark.parametrize("bad", [1, 100_001])
def test_config_max_points_out_of_range_raises(bad: int) -> None:
    with pytest.raises(ValueError, match="max_points must be between"):
        CalculatorPlotTabConfig(max_points=bad)


def test_request_serializable_config_has_refs_and_range() -> None:
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.WORKSPACE_XY,
        x_ref="xs",
        y_ref="ys",
        x_min=0.0,
        x_max=1.0,
        points=10,
    )
    payload = req.serializable_config()
    assert payload["source"] == "workspace_xy"
    assert payload["data_refs"] == {"x": "xs", "y": "ys"}
    assert payload["range"] == {"x_min": 0.0, "x_max": 1.0, "points": 10}


# ---------------------------------------------------------------------------
# build_calculator_plot_spec — happy paths
# ---------------------------------------------------------------------------


def test_workspace_xy_builds_spec() -> None:
    registry = WorkspaceRegistry()
    registry.set("xs", [0.0, 1.0, 2.0])
    registry.set("ys", [0.0, 1.0, 4.0])
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.WORKSPACE_XY, x_ref="xs", y_ref="ys"
    )
    spec = build_calculator_plot_spec(req, registry)
    assert spec.series[0].x == [0.0, 1.0, 2.0]
    assert spec.series[0].y == [0.0, 1.0, 4.0]


def test_workspace_result_uses_index_as_x() -> None:
    registry = WorkspaceRegistry()
    registry.set("calculator_result", [10.0, 20.0, 30.0])
    req = CalculatorPlotRequest(source=CalculatorPlotSource.WORKSPACE_RESULT)
    spec = build_calculator_plot_spec(req, registry)
    assert spec.series[0].x == [0.0, 1.0, 2.0]
    assert spec.series[0].y == [10.0, 20.0, 30.0]


def test_expression_range_builds_spec() -> None:
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.EXPRESSION_RANGE,
        expression="x**2",
        x_min=0.0,
        x_max=2.0,
        points=3,
    )
    spec = build_calculator_plot_spec(req, WorkspaceRegistry())
    assert spec.series[0].y == pytest.approx([0.0, 1.0, 4.0])


def test_expression_uses_safe_math_names() -> None:
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.EXPRESSION_RANGE,
        expression="sin(x)",
        x_min=0.0,
        x_max=3.14159265,
        points=3,
    )
    spec = build_calculator_plot_spec(req, WorkspaceRegistry())
    # sin(0)=0, sin(pi/2)≈1, sin(pi)≈0
    assert spec.series[0].y[0] == pytest.approx(0.0, abs=1e-6)
    assert spec.series[0].y[1] == pytest.approx(1.0, abs=1e-6)


def test_mismatched_lengths_rejected() -> None:
    registry = WorkspaceRegistry()
    registry.set("xs", [1.0, 2.0])
    registry.set("scalar", 5.0)
    # y is scalar -> single element; mismatched length triggers the guard.
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.WORKSPACE_XY, x_ref="xs", y_ref="scalar"
    )
    with pytest.raises(ValueError, match="same length"):
        build_calculator_plot_spec(req, registry)


# ---------------------------------------------------------------------------
# build_calculator_plot_spec — validation guards
# ---------------------------------------------------------------------------


def test_none_request_raises() -> None:
    with pytest.raises(ValueError, match="request must be provided"):
        build_calculator_plot_spec(None, WorkspaceRegistry())  # type: ignore[arg-type]


def test_none_registry_raises() -> None:
    req = CalculatorPlotRequest(source=CalculatorPlotSource.EXPRESSION_RANGE)
    with pytest.raises(ValueError, match="registry must be provided"):
        build_calculator_plot_spec(req, None)  # type: ignore[arg-type]


def test_missing_workspace_variable_raises() -> None:
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.WORKSPACE_XY, x_ref="absent", y_ref="ys"
    )
    with pytest.raises(ValueError, match="Workspace variable not found"):
        build_calculator_plot_spec(req, WorkspaceRegistry())


def test_string_workspace_value_rejected() -> None:
    registry = WorkspaceRegistry()
    registry.set("xs", [0.0, 1.0])
    registry.set("text", "not-numbers")
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.WORKSPACE_XY, x_ref="xs", y_ref="text"
    )
    with pytest.raises(ValueError, match="numeric sequence data"):
        build_calculator_plot_spec(req, registry)


def test_non_numeric_item_rejected() -> None:
    registry = WorkspaceRegistry()
    registry.set("xs", [0.0, 1.0])
    registry.set("ys", [1.0, "two"])
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.WORKSPACE_XY, x_ref="xs", y_ref="ys"
    )
    with pytest.raises(ValueError, match="non-numeric plot data"):
        build_calculator_plot_spec(req, registry)


def test_too_few_points_rejected() -> None:
    registry = WorkspaceRegistry()
    registry.set("xs", [0.0])
    registry.set("ys", [1.0])
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.WORKSPACE_XY, x_ref="xs", y_ref="ys"
    )
    with pytest.raises(ValueError, match="at least two points"):
        build_calculator_plot_spec(req, registry)


def test_expression_empty_rejected() -> None:
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.EXPRESSION_RANGE, expression="   "
    )
    with pytest.raises(ValueError, match="expression must be provided"):
        build_calculator_plot_spec(req, WorkspaceRegistry())


def test_expression_bad_range_rejected() -> None:
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.EXPRESSION_RANGE,
        expression="x",
        x_min=1.0,
        x_max=0.0,
    )
    with pytest.raises(ValueError, match="x_max must be greater"):
        build_calculator_plot_spec(req, WorkspaceRegistry())


def test_expression_invalid_syntax_rejected() -> None:
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.EXPRESSION_RANGE,
        expression="x +* 2",
        x_min=0.0,
        x_max=1.0,
        points=3,
    )
    with pytest.raises(ValueError, match="Invalid plot expression"):
        build_calculator_plot_spec(req, WorkspaceRegistry())


def test_expression_forbids_builtins() -> None:
    # __import__ is not in the safe namespace and builtins are stripped.
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.EXPRESSION_RANGE,
        expression="__import__('os').getpid()",
        x_min=0.0,
        x_max=1.0,
        points=3,
    )
    with pytest.raises(ValueError, match="Invalid plot expression"):
        build_calculator_plot_spec(req, WorkspaceRegistry())


def test_downsampling_respects_max_points() -> None:
    registry = WorkspaceRegistry()
    n = 5000
    registry.set("xs", [float(i) for i in range(n)])
    registry.set("ys", [float(i) for i in range(n)])
    req = CalculatorPlotRequest(
        source=CalculatorPlotSource.WORKSPACE_XY,
        x_ref="xs",
        y_ref="ys",
        config=CalculatorPlotTabConfig(max_points=100),
    )
    spec = build_calculator_plot_spec(req, registry)
    assert len(spec.series[0].x) <= 100
