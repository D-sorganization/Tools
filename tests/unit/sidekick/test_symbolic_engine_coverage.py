"""Unit tests for sidekick.symbolic_engine (issues #3032, #2934).

Achieves >= 70% line coverage on symbolic_engine.py without requiring Qt.
SymPy is importskipped wherever it is unavailable; the remaining branches
(SymPy-unavailable path) are tested via monkeypatching.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helper: import module under test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eng():  # type: ignore[no-untyped-def]
    """Import and return the symbolic_engine module."""
    import importlib
    import sys
    from pathlib import Path

    shared = Path(__file__).resolve().parents[4] / "src" / "shared" / "python"
    if str(shared) not in sys.path:
        sys.path.insert(0, str(shared))

    return importlib.import_module("sidekick.symbolic_engine")


# ---------------------------------------------------------------------------
# is_sympy_available
# ---------------------------------------------------------------------------


def test_is_sympy_available_returns_bool(eng) -> None:  # type: ignore[no-untyped-def]
    """is_sympy_available returns a bool."""
    result = eng.is_sympy_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# SymbolicEngineError
# ---------------------------------------------------------------------------


def test_symbolic_engine_error_is_runtime_error(eng) -> None:  # type: ignore[no-untyped-def]
    """SymbolicEngineError is a subclass of RuntimeError."""
    assert issubclass(eng.SymbolicEngineError, RuntimeError)


# ---------------------------------------------------------------------------
# Tests when SymPy IS available (skipped if not installed)
# ---------------------------------------------------------------------------


def test_symbolic_solve_quadratic(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_solve returns two solutions for x**2 - 4."""
    pytest.importorskip("sympy")
    solutions = eng.symbolic_solve("x**2 - 4", "x")
    assert isinstance(solutions, list)
    assert len(solutions) == 2
    assert "-2" in solutions
    assert "2" in solutions


def test_symbolic_solve_linear(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_solve solves a simple linear equation."""
    pytest.importorskip("sympy")
    solutions = eng.symbolic_solve("x - 7", "x")
    assert "7" in solutions


def test_symbolic_diff_basic(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_diff differentiates x**3 to 3*x**2."""
    pytest.importorskip("sympy")
    result = eng.symbolic_diff("x**3", "x")
    assert "3" in result
    assert "x" in result


def test_symbolic_diff_second_order(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_diff with order=2 computes the second derivative."""
    pytest.importorskip("sympy")
    result = eng.symbolic_diff("x**3", "x", order=2)
    # Second derivative of x**3 is 6x
    assert "6" in result


def test_symbolic_integrate_indefinite(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_integrate returns indefinite integral of x**2."""
    pytest.importorskip("sympy")
    result = eng.symbolic_integrate("x**2", "x")
    assert "x" in result


def test_symbolic_integrate_definite(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_integrate computes definite integral of x**2 from 0 to 1."""
    pytest.importorskip("sympy")
    result = eng.symbolic_integrate("x**2", "x", lower="0", upper="1")
    assert "1" in result and "3" in result  # result is 1/3


def test_symbolic_simplify_trig_identity(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_simplify reduces sin^2+cos^2 to 1."""
    pytest.importorskip("sympy")
    result = eng.symbolic_simplify("sin(x)**2 + cos(x)**2")
    assert result == "1"


def test_symbolic_expand_binomial(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_expand expands (x + 1)**2."""
    pytest.importorskip("sympy")
    result = eng.symbolic_expand("(x + 1)**2")
    assert "x**2" in result
    assert "2*x" in result


def test_symbolic_to_latex_power(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_to_latex renders x**2 as x^{2}."""
    pytest.importorskip("sympy")
    result = eng.symbolic_to_latex("x**2")
    assert "x^{2}" in result


# ---------------------------------------------------------------------------
# Input validation (do NOT need SymPy for TypeError / ValueError paths)
# ---------------------------------------------------------------------------


def test_symbolic_diff_bad_order_type(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_diff raises TypeError if order is not an int."""
    pytest.importorskip("sympy")
    with pytest.raises(TypeError, match="order must be int"):
        eng.symbolic_diff("x**2", "x", order=1.5)  # type: ignore[arg-type]


def test_symbolic_diff_bad_order_value(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_diff raises ValueError if order < 1."""
    pytest.importorskip("sympy")
    with pytest.raises(ValueError, match="order must be >= 1"):
        eng.symbolic_diff("x**2", "x", order=0)


def test_symbolic_integrate_mismatched_bounds(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_integrate raises ValueError if only one bound is given."""
    pytest.importorskip("sympy")
    with pytest.raises(ValueError, match="Both lower and upper"):
        eng.symbolic_integrate("x", "x", lower="0")


def test_symbolic_solve_empty_expr_raises(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_solve raises ValueError for an empty expression."""
    pytest.importorskip("sympy")
    with pytest.raises(ValueError, match="must not be empty"):
        eng.symbolic_solve("", "x")


def test_symbolic_solve_bad_type_raises(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_solve raises TypeError if expr_str is not a string."""
    pytest.importorskip("sympy")
    with pytest.raises(TypeError, match="expr_str must be str"):
        eng.symbolic_solve(42, "x")  # type: ignore[arg-type]


def test_symbolic_diff_empty_symbol_raises(eng) -> None:  # type: ignore[no-untyped-def]
    """symbolic_diff raises ValueError for an empty symbol."""
    pytest.importorskip("sympy")
    with pytest.raises(ValueError, match="symbol must not be empty"):
        eng.symbolic_diff("x**2", "")


# ---------------------------------------------------------------------------
# Tests when SymPy is NOT available (monkeypatched)
# ---------------------------------------------------------------------------


def test_symbolic_solve_no_sympy_raises(eng, monkeypatch: pytest.MonkeyPatch) -> None:
    """symbolic_solve raises SymbolicEngineError when SymPy is unavailable."""
    monkeypatch.setattr(eng, "_SYMPY_AVAILABLE", False)
    with pytest.raises(eng.SymbolicEngineError, match="sympy is not installed"):
        eng.symbolic_solve("x**2 - 4", "x")


def test_symbolic_diff_no_sympy_raises(eng, monkeypatch: pytest.MonkeyPatch) -> None:
    """symbolic_diff raises SymbolicEngineError when SymPy is unavailable."""
    monkeypatch.setattr(eng, "_SYMPY_AVAILABLE", False)
    with pytest.raises(eng.SymbolicEngineError, match="sympy is not installed"):
        eng.symbolic_diff("x**2")


def test_symbolic_integrate_no_sympy_raises(
    eng, monkeypatch: pytest.MonkeyPatch
) -> None:
    """symbolic_integrate raises SymbolicEngineError when SymPy is unavailable."""
    monkeypatch.setattr(eng, "_SYMPY_AVAILABLE", False)
    with pytest.raises(eng.SymbolicEngineError, match="sympy is not installed"):
        eng.symbolic_integrate("x")


def test_symbolic_simplify_no_sympy_raises(
    eng, monkeypatch: pytest.MonkeyPatch
) -> None:
    """symbolic_simplify raises SymbolicEngineError when SymPy is unavailable."""
    monkeypatch.setattr(eng, "_SYMPY_AVAILABLE", False)
    with pytest.raises(eng.SymbolicEngineError, match="sympy is not installed"):
        eng.symbolic_simplify("x")


def test_symbolic_expand_no_sympy_raises(eng, monkeypatch: pytest.MonkeyPatch) -> None:
    """symbolic_expand raises SymbolicEngineError when SymPy is unavailable."""
    monkeypatch.setattr(eng, "_SYMPY_AVAILABLE", False)
    with pytest.raises(eng.SymbolicEngineError, match="sympy is not installed"):
        eng.symbolic_expand("x")


def test_symbolic_to_latex_no_sympy_raises(
    eng, monkeypatch: pytest.MonkeyPatch
) -> None:
    """symbolic_to_latex raises SymbolicEngineError when SymPy is unavailable."""
    monkeypatch.setattr(eng, "_SYMPY_AVAILABLE", False)
    with pytest.raises(eng.SymbolicEngineError, match="sympy is not installed"):
        eng.symbolic_to_latex("x**2")


def test_is_sympy_available_false_when_patched(
    eng, monkeypatch: pytest.MonkeyPatch
) -> None:
    """is_sympy_available returns False when _SYMPY_AVAILABLE is False."""
    monkeypatch.setattr(eng, "_SYMPY_AVAILABLE", False)
    assert eng.is_sympy_available() is False
