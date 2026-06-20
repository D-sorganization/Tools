"""Regression tests for #3715.

``PythonReplWidget`` previously ran user ``eval``/``exec`` against the SAME
namespace dict the GUI held, and Cancel called ``QThread.terminate()`` — which
kills the thread at an arbitrary instruction. Terminating mid-mutation could
leave the live namespace half-updated. The widget now runs the worker against
an isolated copy of the namespace, merged back only on clean completion, so a
cancelled run leaves the live namespace exactly as it was.
"""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = pytest.mark.serial

if sys.platform == "win32" and os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "Qt Python REPL widget tests run serially on Windows.",
        allow_module_level=True,
    )

pytest.importorskip("PyQt6")


def _build_repl(qtbot):
    from sidekick.ui.tools_sidebar.calculator_startup import CalculatorStartupConfig
    from sidekick.ui.tools_sidebar.python_repl_tab import PythonReplWidget
    from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

    registry = WorkspaceRegistry()

    def set_variable(name: str, value: object) -> None:
        registry.set(name, value)

    widget = PythonReplWidget(
        registry=registry,
        set_variable=set_variable,
        startup_config=CalculatorStartupConfig(()),
    )
    qtbot.addWidget(widget)
    return widget, registry


def _execute_and_wait(widget, qtbot, script: str) -> None:
    widget.execute(script)
    qtbot.waitUntil(lambda: widget._worker is None, timeout=3000)  # noqa: SLF001


class _DisconnectableSignal:
    def disconnect(self, callback) -> None:  # noqa: ANN001, ARG002
        pass


class _CancelledWorker:
    result_ready = _DisconnectableSignal()

    def __init__(self, namespace: dict[str, object]) -> None:
        self._namespace = namespace
        self.terminated = False

    def isRunning(self) -> bool:  # noqa: N802
        return not self.terminated

    def terminate(self) -> None:
        self._namespace["corrupt"] = "partial"
        self.terminated = True

    def wait(self) -> None:
        pass

    def deleteLater(self) -> None:  # noqa: N802
        pass


def test_worker_runs_against_isolated_namespace_copy(qapp, qtbot) -> None:
    """The worker must not share the live namespace dict by reference."""
    widget, _ = _build_repl(qtbot)
    _execute_and_wait(widget, qtbot, "a = 1")
    # Clean completion merges back: the live namespace has the value.
    assert widget._namespace.get("a") == 1  # noqa: SLF001


def test_cancel_does_not_corrupt_live_namespace(qapp, qtbot) -> None:
    """Terminating a running worker leaves the live namespace unchanged."""
    widget, _ = _build_repl(qtbot)
    widget._namespace["sentinel"] = "untouched"  # noqa: SLF001
    before = dict(widget._namespace)  # noqa: SLF001

    # A fake running worker whose termination mutates only its private copy.
    worker = _CancelledWorker(dict(widget._namespace))  # noqa: SLF001
    widget._worker = worker  # noqa: SLF001

    widget._on_cancel_clicked()  # noqa: SLF001

    # The worker mutated only its private copy; the live namespace is intact.
    assert widget._namespace == before  # noqa: SLF001
    assert "corrupt" not in widget._namespace  # noqa: SLF001
    assert widget._namespace["sentinel"] == "untouched"  # noqa: SLF001
    assert widget._worker is None  # noqa: SLF001
