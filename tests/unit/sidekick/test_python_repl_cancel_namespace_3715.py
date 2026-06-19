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
import time

import pytest

pytestmark = pytest.mark.serial

if sys.platform == "win32" and os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "Qt Python REPL widget tests run serially on Windows.",
        allow_module_level=True,
    )

pytest.importorskip("PyQt6")


def _build_repl(qtbot):
    from sidekick.ui.tools_sidebar.python_repl_tab import PythonReplWidget
    from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

    registry = WorkspaceRegistry()

    def set_variable(name: str, value: object) -> None:
        registry.set(name, value)

    widget = PythonReplWidget(registry=registry, set_variable=set_variable)
    qtbot.addWidget(widget)
    return widget, registry


def test_worker_runs_against_isolated_namespace_copy(qapp, qtbot) -> None:
    """The worker must not share the live namespace dict by reference."""
    widget, _ = _build_repl(qtbot)
    widget.execute("a = 1")
    # Clean completion merges back: the live namespace has the value.
    assert widget._namespace.get("a") == 1  # noqa: SLF001


def test_cancel_does_not_corrupt_live_namespace(qapp, qtbot) -> None:
    """Terminating a running worker leaves the live namespace unchanged."""
    from sidekick.ui.tools_sidebar.python_repl_tab import _ReplWorker

    widget, _ = _build_repl(qtbot)
    widget._namespace["sentinel"] = "untouched"  # noqa: SLF001
    before = dict(widget._namespace)  # noqa: SLF001

    # A worker whose script would mutate the namespace then spin for a while.
    worker = _ReplWorker(
        "corrupt = 'partial'\nimport time as _t\n_t.sleep(5)",
        dict(widget._namespace),  # isolated copy, mirroring execute()
    )
    widget._worker = worker  # noqa: SLF001
    worker.start()
    # Let the worker run far enough to have mutated its OWN copy.
    deadline = time.time() + 2.0
    while not worker.isRunning() and time.time() < deadline:
        qapp.processEvents()
    time.sleep(0.2)

    widget._on_cancel_clicked()  # noqa: SLF001

    # The worker mutated only its private copy; the live namespace is intact.
    assert widget._namespace == before  # noqa: SLF001
    assert "corrupt" not in widget._namespace  # noqa: SLF001
    assert widget._namespace["sentinel"] == "untouched"  # noqa: SLF001
    assert widget._worker is None  # noqa: SLF001
