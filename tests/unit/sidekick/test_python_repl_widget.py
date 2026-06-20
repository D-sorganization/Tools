"""``PythonReplWidget`` REPL behaviour tests (UpstreamDrift #5616).

The widget is the reusable DRY surface shared by the Terminal tab and the
MATLAB-home command window.
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
    from upstream_drift_tools.ui.tools_sidebar.calculator_startup import (
        CalculatorStartupConfig,
    )
    from upstream_drift_tools.ui.tools_sidebar.registry import WorkspaceRegistry
    from upstream_drift_tools.ui.tools_sidebar.runtime_tabs import PythonReplWidget

    registry = WorkspaceRegistry()
    captured: dict[str, object] = {}

    def set_variable(name: str, value: object) -> None:
        captured[name] = value
        registry.set(name, value)

    widget = PythonReplWidget(
        registry=registry,
        set_variable=set_variable,
        startup_config=CalculatorStartupConfig(()),
    )
    qtbot.addWidget(widget)
    return widget, registry, captured


def _execute_and_wait(widget, qtbot, script: str) -> None:
    widget.execute(script)
    qtbot.waitUntil(lambda: widget._worker is None, timeout=3000)  # noqa: SLF001


class _DisconnectableSignal:
    def __init__(self) -> None:
        self.disconnected = False

    def disconnect(self, callback) -> None:  # noqa: ANN001, ARG002
        self.disconnected = True


class _RunningWorker:
    def __init__(self) -> None:
        self.result_ready = _DisconnectableSignal()
        self.terminated = False
        self.waited = False
        self.deleted = False

    def isRunning(self) -> bool:  # noqa: N802
        return not self.terminated

    def terminate(self) -> None:
        self.terminated = True

    def wait(self) -> None:
        self.waited = True

    def deleteLater(self) -> None:  # noqa: N802
        self.deleted = True


def test_evaluates_expression_and_shows_result(qapp, qtbot) -> None:
    widget, _, _ = _build_repl(qtbot)
    _execute_and_wait(widget, qtbot, "1 + 2")
    assert "3" in widget.output_text()


def test_repeated_execution_finishes_before_widget_teardown(qapp, qtbot) -> None:
    widget, registry, _ = _build_repl(qtbot)
    _execute_and_wait(widget, qtbot, "x = 1")
    _execute_and_wait(widget, qtbot, "x = x + 1")

    assert registry.get("x") == 2
    assert widget._worker is None  # noqa: SLF001

    widget.close()
    qapp.processEvents()


def test_assignment_stores_in_workspace_registry(qapp, qtbot) -> None:
    widget, registry, captured = _build_repl(qtbot)
    _execute_and_wait(widget, qtbot, "x = 3")
    assert registry.get("x") == 3
    assert captured.get("x") == 3


def test_history_recorded(qapp, qtbot) -> None:
    widget, _, _ = _build_repl(qtbot)
    _execute_and_wait(widget, qtbot, "a = 1")
    _execute_and_wait(widget, qtbot, "b = 2")
    history = widget.history()
    assert history[-2:] == ("a = 1", "b = 2")


def test_exception_displays_without_crashing(qapp, qtbot) -> None:
    widget, _, _ = _build_repl(qtbot)
    _execute_and_wait(widget, qtbot, "raise ValueError('boom')")
    output = widget.output_text()
    assert "ValueError" in output
    assert "boom" in output


def test_execute_returns_while_worker_is_running(qapp, qtbot) -> None:
    widget, _, _ = _build_repl(qtbot)

    started_at = time.perf_counter()
    widget.execute("import time\ntime.sleep(0.5)\nslow_result = 1")
    elapsed = time.perf_counter() - started_at

    assert elapsed < 0.2
    assert widget._worker is not None  # noqa: SLF001
    assert not widget._run_button.isEnabled()  # noqa: SLF001

    qtbot.waitUntil(lambda: widget._worker is None, timeout=3000)  # noqa: SLF001
    assert widget._run_button.isEnabled()  # noqa: SLF001
    assert widget._namespace["slow_result"] == 1  # noqa: SLF001


def test_reentrant_execute_is_ignored_while_worker_runs(qapp, qtbot) -> None:
    widget, _, _ = _build_repl(qtbot)
    widget.execute("import time\ntime.sleep(0.5)\nfirst = 1")
    first_worker = widget._worker  # noqa: SLF001

    widget.execute("second = 2")

    assert widget._worker is first_worker  # noqa: SLF001
    assert widget.history() == ("import time\ntime.sleep(0.5)\nfirst = 1",)
    qtbot.waitUntil(lambda: widget._worker is None, timeout=3000)  # noqa: SLF001
    assert "second" not in widget._namespace  # noqa: SLF001


def test_cancel_restores_non_running_state_and_records_output(qapp, qtbot) -> None:
    widget, _, _ = _build_repl(qtbot)
    worker = _RunningWorker()
    widget._worker = worker  # noqa: SLF001
    widget._set_running(True)  # noqa: SLF001

    widget._on_cancel_clicked()  # noqa: SLF001

    assert worker.result_ready.disconnected
    assert worker.terminated
    assert worker.waited
    assert worker.deleted
    assert widget._worker is None  # noqa: SLF001
    assert widget._run_button.isEnabled()  # noqa: SLF001
    assert widget._cancel_button.isHidden()  # noqa: SLF001
    assert "[Cancelled]" in widget.output_text()


def test_deleted_variable_is_removed_from_registry(qapp, qtbot) -> None:
    widget, registry, _ = _build_repl(qtbot)
    _execute_and_wait(widget, qtbot, "x = 1")
    assert registry.get("x") == 1

    _execute_and_wait(widget, qtbot, "del x")

    assert "x" not in registry.list_names()


def test_rebinding_exported_name_to_callable_removes_registry_entry(
    qapp, qtbot
) -> None:
    widget, registry, _ = _build_repl(qtbot)
    _execute_and_wait(widget, qtbot, "x = 1")
    assert registry.get("x") == 1

    _execute_and_wait(widget, qtbot, "x = lambda: 1")

    assert "x" not in registry.list_names()


def test_namespace_export_filter_excludes_modules_callables_and_reserved_names(
    qapp, qtbot
) -> None:
    widget, registry, _ = _build_repl(qtbot)

    _execute_and_wait(
        widget,
        qtbot,
        "import math\n"
        "plain_value = 7\n"
        "def user_function():\n"
        "    return 1\n"
        "np = 5\n"
        "_private_value = 8",
    )

    names = set(registry.list_names())
    assert "plain_value" in names
    assert "math" not in names
    assert "user_function" not in names
    assert "np" not in names
    assert "_private_value" not in names


def test_registry_required(qapp) -> None:
    from upstream_drift_tools.ui.tools_sidebar.runtime_tabs import PythonReplWidget

    with pytest.raises((TypeError, ValueError)):
        PythonReplWidget(registry=None, set_variable=lambda *_: None)


def test_set_variable_required(qapp) -> None:
    from upstream_drift_tools.ui.tools_sidebar.registry import WorkspaceRegistry
    from upstream_drift_tools.ui.tools_sidebar.runtime_tabs import PythonReplWidget

    with pytest.raises((TypeError, ValueError)):
        PythonReplWidget(registry=WorkspaceRegistry(), set_variable=None)
