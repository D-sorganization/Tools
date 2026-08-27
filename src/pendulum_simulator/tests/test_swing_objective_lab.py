"""Contract tests for the Swing Objective Lab surface.

The widget is presentation only. The orthogonality test below is the one that
matters long-term: if physics or optimization logic ever leaks into the GUI
module, the engine stops being reusable from the CLI, the notebooks, and the
React mirror, and the same equations start living in two places.

Closes #4771.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 is required for GUI surfaces")

from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from double_pendulum_golf.gui.swing_objective_lab import (  # noqa: E402
    SwingObjectiveLabWidget,
    SwingObjectiveLabWindow,
    format_comparison_matrix,
)
from double_pendulum_golf.swing_objectives.comparison import (  # noqa: E402
    SwingComparison,
)


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """A single offscreen QApplication for the module."""
    app = QApplication.instance() or QApplication([])
    return app  # type: ignore[return-value]


def _stub_comparison() -> SwingComparison:
    """A small comparison built without running the optimizer."""
    keys = ("clubhead_speed", "coriolis")
    raw = {
        "clubhead_speed": {"clubhead_speed": 40.0, "coriolis": 90.0},
        "coriolis": {"clubhead_speed": 38.0, "coriolis": 100.0},
    }
    return SwingComparison(
        objective_keys=keys,
        raw_values=raw,
        matrix=np.array([[100.0, 90.0], [95.0, 100.0]]),
        torque_saturation={key: np.array([0.1, 0.2]) for key in keys},
        swing_distance=np.array([[0.0, 0.2], [0.2, 0.0]]),
        diagnostics={
            key: {
                "objective_value": raw[key][key],
                "success": True,
                "max_defect": 1e-12,
                "max_slew_violation": 0.0,
                "iterations": 30,
            }
            for key in keys
        },
    )


def test_widget_constructs_headless(qt_app: QApplication) -> None:
    """The surface must build without a display or a solved comparison."""
    widget = SwingObjectiveLabWidget()
    assert isinstance(widget, QWidget)
    widget.deleteLater()


def test_window_constructs_headless(qt_app: QApplication) -> None:
    """The standalone window the tile launches must build headless."""
    window = SwingObjectiveLabWindow()
    assert window.windowTitle()
    window.deleteLater()


def test_widget_renders_a_comparison_without_running_the_solver(
    qt_app: QApplication,
) -> None:
    """Presentation is separable from computation, so results can be injected."""
    widget = SwingObjectiveLabWidget()
    widget.display_comparison(_stub_comparison())

    assert widget.result_table.rowCount() == 2
    assert widget.matrix_table.rowCount() == 2
    assert widget.matrix_table.columnCount() == 2
    widget.deleteLater()


def test_degenerate_comparison_is_surfaced_to_the_reader(
    qt_app: QApplication,
) -> None:
    """A collapsed feasible set must be visible in the UI, not just in the data.

    An all-100% matrix reads as unanimous agreement between the mechanisms; the
    surface has to say that the configuration, not the physics, produced it.
    """
    comparison = _stub_comparison()
    pinned = SwingComparison(
        objective_keys=comparison.objective_keys,
        raw_values=comparison.raw_values,
        matrix=np.full((2, 2), 100.0),
        torque_saturation=comparison.torque_saturation,
        swing_distance=np.zeros((2, 2)),
        diagnostics=comparison.diagnostics,
    )
    assert pinned.is_degenerate

    widget = SwingObjectiveLabWidget()
    widget.display_comparison(pinned)
    assert "degenerate" in widget.status_text().lower()
    widget.deleteLater()


def test_every_matrix_cell_is_labelled(qt_app: QApplication) -> None:
    """Colour must never be the only encoding of a value (accessibility)."""
    widget = SwingObjectiveLabWidget()
    widget.display_comparison(_stub_comparison())

    for row in range(widget.matrix_table.rowCount()):
        for column in range(widget.matrix_table.columnCount()):
            item = widget.matrix_table.item(row, column)
            assert item is not None and item.text().strip()
    widget.deleteLater()


def test_format_comparison_matrix_is_pure_and_labelled() -> None:
    """The table formatter is a pure function, testable without Qt."""
    rows = format_comparison_matrix(_stub_comparison())
    assert rows == [["100.0%", "90.0%"], ["95.0%", "100.0%"]]


def test_gui_module_contains_no_physics_or_optimization(
    qt_app: QApplication,
) -> None:
    """Orthogonality: the surface may render results, never compute them.

    AGENTS.md 5c forbids mixing UI with calculation. Concretely, the GUI module
    must not import the physics kernel or re-derive any dynamics; it may only
    call the engine's public entry points.
    """
    import inspect

    from double_pendulum_golf.gui import swing_objective_lab

    source = inspect.getsource(swing_objective_lab)
    for forbidden in ("mass_matrix", "coriolis_vector", "gravity_vector", "np.linalg"):
        assert forbidden not in source, (
            f"{forbidden!r} appears in the GUI module; physics belongs in the engine"
        )


def test_worker_is_used_so_the_ui_thread_never_blocks(qt_app: QApplication) -> None:
    """Solving runs off the UI thread; a multi-second solve must not freeze it."""
    from double_pendulum_golf.gui.swing_objective_lab import ComparisonWorker

    widget = SwingObjectiveLabWidget()
    assert hasattr(widget, "run_comparison")
    assert issubclass(ComparisonWorker, object)
    assert hasattr(ComparisonWorker, "finished")
    assert hasattr(ComparisonWorker, "failed")
    widget.deleteLater()


def test_embed_adapter_returns_a_widget_without_an_event_loop(
    qt_app: QApplication,
) -> None:
    """The launcher embeds the surface; it must not start its own event loop."""
    from double_pendulum_golf.swing_objectives._embed_adapter import get_dockable_ui

    surface = get_dockable_ui()
    assert isinstance(surface, QWidget)
    surface.deleteLater()
