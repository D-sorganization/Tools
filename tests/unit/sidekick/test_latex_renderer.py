"""Unit tests for sidekick.latex_renderer (issues #3032, #2934).

Achieves >= 70% line coverage on latex_renderer.py.
Qt-based tests are skipped when PyQt6 is unavailable.
SymPy/Qt unavailability paths are tested via monkeypatching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SHARED = Path(__file__).resolve().parents[4] / "src" / "shared" / "python"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def renderer():  # type: ignore[no-untyped-def]
    """Import and return the latex_renderer module."""
    import importlib

    return importlib.import_module("sidekick.latex_renderer")


# ---------------------------------------------------------------------------
# LatexRenderError
# ---------------------------------------------------------------------------


def test_latex_render_error_is_runtime_error(renderer) -> None:  # type: ignore[no-untyped-def]
    """LatexRenderError is a subclass of RuntimeError."""
    assert issubclass(renderer.LatexRenderError, RuntimeError)


# ---------------------------------------------------------------------------
# is_qt_available / is_latex_ready
# ---------------------------------------------------------------------------


def test_is_qt_available_returns_bool(renderer) -> None:  # type: ignore[no-untyped-def]
    """is_qt_available returns a bool."""
    result = renderer.is_qt_available()
    assert isinstance(result, bool)


def test_is_latex_ready_returns_bool(renderer) -> None:  # type: ignore[no-untyped-def]
    """is_latex_ready returns a bool."""
    result = renderer.is_latex_ready()
    assert isinstance(result, bool)


def test_is_latex_ready_false_when_qt_missing(
    renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """is_latex_ready is False when Qt is absent."""
    monkeypatch.setattr(renderer, "_QT_AVAILABLE", False)
    assert renderer.is_latex_ready() is False


def test_is_latex_ready_false_when_sympy_missing(
    renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """is_latex_ready is False when SymPy is absent."""
    monkeypatch.setattr(renderer, "_SYMPY_AVAILABLE", False)
    assert renderer.is_latex_ready() is False


# ---------------------------------------------------------------------------
# Input validation for render_latex_label
# ---------------------------------------------------------------------------


def test_render_latex_label_type_error(renderer) -> None:  # type: ignore[no-untyped-def]
    """render_latex_label raises TypeError for non-string input."""
    with pytest.raises(TypeError, match="latex_str must be str"):
        renderer.render_latex_label(123)  # type: ignore[arg-type]


def test_render_latex_label_value_error_empty(renderer) -> None:  # type: ignore[no-untyped-def]
    """render_latex_label raises ValueError for empty string."""
    with pytest.raises(ValueError, match="must not be empty"):
        renderer.render_latex_label("   ")


# ---------------------------------------------------------------------------
# Input validation for render_expr_label
# ---------------------------------------------------------------------------


def test_render_expr_label_type_error(renderer) -> None:  # type: ignore[no-untyped-def]
    """render_expr_label raises TypeError for non-string input."""
    with pytest.raises(TypeError, match="expr_str must be str"):
        renderer.render_expr_label(42)  # type: ignore[arg-type]


def test_render_expr_label_value_error_empty(renderer) -> None:  # type: ignore[no-untyped-def]
    """render_expr_label raises ValueError for empty string."""
    with pytest.raises(ValueError, match="must not be empty"):
        renderer.render_expr_label("")


# ---------------------------------------------------------------------------
# render_latex_label without Qt (monkeypatched)
# ---------------------------------------------------------------------------


def test_render_latex_label_no_qt_raises(
    renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """render_latex_label raises LatexRenderError when Qt is unavailable."""
    monkeypatch.setattr(renderer, "_QT_AVAILABLE", False)
    with pytest.raises(renderer.LatexRenderError, match="PyQt6 is not installed"):
        renderer.render_latex_label("x^{2}")


# ---------------------------------------------------------------------------
# _expr_to_latex — validates directly without Qt
# ---------------------------------------------------------------------------


def test_expr_to_latex_type_error(renderer) -> None:  # type: ignore[no-untyped-def]
    """_expr_to_latex raises TypeError if expr_str is not a string."""
    with pytest.raises(TypeError, match="expr_str must be str"):
        renderer._expr_to_latex(99)  # type: ignore[arg-type]


def test_expr_to_latex_value_error_empty(renderer) -> None:  # type: ignore[no-untyped-def]
    """_expr_to_latex raises ValueError if expr_str is empty."""
    with pytest.raises(ValueError, match="must not be empty"):
        renderer._expr_to_latex("")


def test_expr_to_latex_no_sympy_raises(
    renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_expr_to_latex raises LatexRenderError if SymPy is unavailable."""
    monkeypatch.setattr(renderer, "_SYMPY_AVAILABLE", False)
    with pytest.raises(renderer.LatexRenderError, match="sympy is not installed"):
        renderer._expr_to_latex("x**2")


def test_expr_to_latex_success(renderer) -> None:  # type: ignore[no-untyped-def]
    """_expr_to_latex returns a LaTeX string for a valid expression."""
    pytest.importorskip("sympy")
    result = renderer._expr_to_latex("x**2")
    assert "x" in result


# ---------------------------------------------------------------------------
# render_expr_label — no Qt (monkeypatched)
# ---------------------------------------------------------------------------


def test_render_expr_label_no_sympy_raises(
    renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """render_expr_label raises LatexRenderError when SymPy is unavailable."""
    monkeypatch.setattr(renderer, "_SYMPY_AVAILABLE", False)
    with pytest.raises(renderer.LatexRenderError, match="sympy is not installed"):
        renderer.render_expr_label("x**2")


# ---------------------------------------------------------------------------
# Qt-backed tests (skipped when PyQt6 not installed)
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_render_latex_label_returns_widget(renderer, qtbot) -> None:  # type: ignore[no-untyped-def]
    """render_latex_label returns a QLabel widget."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QLabel

    label = renderer.render_latex_label(r"x^{2}")
    qtbot.addWidget(label)
    assert isinstance(label, QLabel)


@pytest.mark.gui
def test_render_latex_label_object_name(renderer, qtbot) -> None:  # type: ignore[no-untyped-def]
    """render_latex_label sets the expected object name on the QLabel."""
    pytest.importorskip("PyQt6")

    label = renderer.render_latex_label(r"x^{2}")
    qtbot.addWidget(label)
    assert label.objectName() == "SidekickLatexLabel"


@pytest.mark.gui
def test_render_latex_label_tooltip(renderer, qtbot) -> None:  # type: ignore[no-untyped-def]
    """render_latex_label sets the tooltip to the LaTeX source."""
    pytest.importorskip("PyQt6")

    latex = r"x^{2}"
    label = renderer.render_latex_label(latex)
    qtbot.addWidget(label)
    assert latex in label.toolTip()


@pytest.mark.gui
def test_render_expr_label_with_sympy_and_qt(renderer, qtbot) -> None:  # type: ignore[no-untyped-def]
    """render_expr_label converts expression to LaTeX label widget."""
    pytest.importorskip("sympy")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QLabel

    label = renderer.render_expr_label("x**2")
    qtbot.addWidget(label)
    assert isinstance(label, QLabel)
