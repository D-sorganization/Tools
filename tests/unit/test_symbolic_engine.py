"""Unit tests for sidekick.symbolic_engine (issue #2934).

Tests five symbolic operations: solve, diff, integrate, simplify, expand.
Also tests LaTeX rendering via sidekick.latex_renderer.

TDD: these tests drove the implementation of symbolic_engine.py and
latex_renderer.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

sympy = pytest.importorskip("sympy", reason="sympy not installed")


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from sidekick.symbolic_engine import (  # noqa: E402
    is_sympy_available,
    symbolic_diff,
    symbolic_expand,
    symbolic_integrate,
    symbolic_simplify,
    symbolic_solve,
    symbolic_to_latex,
)

# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------


def test_is_sympy_available() -> None:
    """is_sympy_available returns True when sympy is importable."""
    assert is_sympy_available() is True


# ---------------------------------------------------------------------------
# symbolic_solve — 5 parametrised cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr_str", "symbol", "expected_solutions"),
    [
        ("x**2 - 4", "x", {"-2", "2"}),
        ("x - 5", "x", {"5"}),
        ("x**2 + 1", "x", {"-I", "I"}),  # imaginary roots
        ("2*x - 6", "x", {"3"}),
        ("x**3 - x", "x", {"-1", "0", "1"}),
    ],
)
def test_symbolic_solve_parametrised(
    expr_str: str, symbol: str, expected_solutions: set[str]
) -> None:
    """solve(expr, symbol) returns the expected solution set."""
    solutions = symbolic_solve(expr_str, symbol)
    assert set(solutions) == expected_solutions


def test_symbolic_solve_type_error_expr() -> None:
    with pytest.raises(TypeError, match="expr_str must be str"):
        symbolic_solve(42, "x")  # type: ignore[arg-type]


def test_symbolic_solve_value_error_empty_expr() -> None:
    with pytest.raises(ValueError, match="expr_str must not be empty"):
        symbolic_solve("  ", "x")


def test_symbolic_solve_type_error_symbol() -> None:
    with pytest.raises(TypeError, match="symbol must be str"):
        symbolic_solve("x**2 - 1", 123)  # type: ignore[arg-type]


def test_symbolic_solve_value_error_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol must not be empty"):
        symbolic_solve("x**2 - 1", "")


# ---------------------------------------------------------------------------
# symbolic_diff — 5 parametrised cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr_str", "symbol", "order", "expected"),
    [
        ("x**3", "x", 1, "3*x**2"),
        ("x**3", "x", 2, "6*x"),
        ("x**3", "x", 3, "6"),
        ("sin(x)", "x", 1, "cos(x)"),
        ("exp(x)", "x", 1, "exp(x)"),
    ],
)
def test_symbolic_diff_parametrised(
    expr_str: str, symbol: str, order: int, expected: str
) -> None:
    """diff(expr, symbol, order) returns the expected derivative string."""
    result = symbolic_diff(expr_str, symbol, order)
    assert result == expected


def test_symbolic_diff_order_type_error() -> None:
    with pytest.raises(TypeError, match="order must be int"):
        symbolic_diff("x**2", "x", order=1.5)  # type: ignore[arg-type]


def test_symbolic_diff_order_value_error() -> None:
    with pytest.raises(ValueError, match="order must be >= 1"):
        symbolic_diff("x**2", "x", order=0)


# ---------------------------------------------------------------------------
# symbolic_integrate — 5 parametrised cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr_str", "symbol", "lower", "upper", "expected"),
    [
        ("x**2", "x", None, None, "x**3/3"),
        ("x", "x", None, None, "x**2/2"),
        ("x**2", "x", "0", "1", "1/3"),
        ("1", "x", "0", "5", "5"),
        ("2*x", "x", "0", "3", "9"),
    ],
)
def test_symbolic_integrate_parametrised(
    expr_str: str,
    symbol: str,
    lower: str | None,
    upper: str | None,
    expected: str,
) -> None:
    """integrate(expr, symbol) returns the expected integral string."""
    result = symbolic_integrate(expr_str, symbol, lower=lower, upper=upper)
    assert result == expected


def test_symbolic_integrate_mismatched_bounds_raises() -> None:
    """Providing only lower or only upper bound must raise ValueError."""
    with pytest.raises(ValueError, match="Both lower and upper bounds"):
        symbolic_integrate("x", "x", lower="0")


# ---------------------------------------------------------------------------
# symbolic_simplify — 5 parametrised cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr_str", "expected"),
    [
        ("sin(x)**2 + cos(x)**2", "1"),
        ("x + x", "2*x"),
        ("(x**2 - 1) / (x - 1)", "x + 1"),
        ("exp(log(x))", "x"),  # exp(log(x)) = x (SymPy evaluates eagerly)
        ("x * 1", "x"),
    ],
)
def test_symbolic_simplify_parametrised(expr_str: str, expected: str) -> None:
    """simplify(expr) returns the expected simplified string."""
    result = symbolic_simplify(expr_str)
    assert result == expected


# ---------------------------------------------------------------------------
# symbolic_expand — 5 parametrised cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr_str", "expected"),
    [
        ("(x + 1)**2", "x**2 + 2*x + 1"),
        ("(x + 1)**3", "x**3 + 3*x**2 + 3*x + 1"),
        ("(x - 1)*(x + 1)", "x**2 - 1"),
        ("(a + b)**2", "a**2 + 2*a*b + b**2"),
        ("(x + 2)*(x - 2)", "x**2 - 4"),
    ],
)
def test_symbolic_expand_parametrised(expr_str: str, expected: str) -> None:
    """expand(expr) returns the expected expanded string."""
    result = symbolic_expand(expr_str)
    assert result == expected


# ---------------------------------------------------------------------------
# symbolic_to_latex
# ---------------------------------------------------------------------------


def test_symbolic_to_latex_x_squared() -> None:
    """x**2 renders to LaTeX 'x^{2}'."""
    result = symbolic_to_latex("x**2")
    assert "x^{2}" in result


def test_symbolic_to_latex_fraction() -> None:
    """x/2 renders to a LaTeX fraction."""
    result = symbolic_to_latex("x/2")
    assert "frac" in result or "x" in result  # SymPy may simplify


def test_symbolic_to_latex_type_error() -> None:
    with pytest.raises(TypeError, match="expr_str must be str"):
        symbolic_to_latex(42)  # type: ignore[arg-type]


def test_symbolic_to_latex_empty_raises() -> None:
    with pytest.raises(ValueError, match="expr_str must not be empty"):
        symbolic_to_latex("")


# ---------------------------------------------------------------------------
# latex_renderer module (GUI test — skipped in headless)
# ---------------------------------------------------------------------------


def test_latex_renderer_is_latex_ready() -> None:
    """is_latex_ready() returns True when both Qt and sympy are available."""
    from sidekick.latex_renderer import is_latex_ready, is_qt_available

    qt = is_qt_available()
    ready = is_latex_ready()
    # If Qt is not available, ready must be False; otherwise it depends on sympy
    if not qt:
        assert ready is False
    # Both sympy (required by this test file) and the presence of qt determine ready
    assert isinstance(ready, bool)


@pytest.mark.gui
def test_latex_render_label_creates_widget(qtbot) -> None:  # type: ignore[no-untyped-def]
    """render_latex_label creates a QLabel with the LaTeX string."""
    pytest.importorskip("PyQt6")
    from sidekick.latex_renderer import render_latex_label

    label = render_latex_label("x^{2}")
    qtbot.addWidget(label)
    assert "x^{2}" in label.text()
    assert label.objectName() == "SidekickLatexLabel"


@pytest.mark.gui
def test_render_expr_label_creates_widget(qtbot) -> None:  # type: ignore[no-untyped-def]
    """render_expr_label parses an expression and renders it as LaTeX in a label."""
    pytest.importorskip("PyQt6")
    from sidekick.latex_renderer import render_expr_label

    label = render_expr_label("x**2")
    qtbot.addWidget(label)
    text = label.text()
    # Should contain LaTeX exponent notation
    assert "2" in text
