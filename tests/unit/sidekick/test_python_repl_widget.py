"""``PythonReplWidget`` REPL behaviour tests (UpstreamDrift #5616).

The widget is the reusable DRY surface shared by the Terminal tab and the
MATLAB-home command window.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def _build_repl(qt_app):
    from upstream_drift_tools.ui.tools_sidebar.registry import WorkspaceRegistry
    from upstream_drift_tools.ui.tools_sidebar.runtime_tabs import PythonReplWidget

    registry = WorkspaceRegistry()
    captured: dict[str, object] = {}

    def set_variable(name: str, value: object) -> None:
        captured[name] = value
        registry.set(name, value)

    widget = PythonReplWidget(registry=registry, set_variable=set_variable)
    return widget, registry, captured


def test_evaluates_expression_and_shows_result(qt_app) -> None:
    widget, _, _ = _build_repl(qt_app)
    widget.execute("1 + 2")
    assert "3" in widget.output_text()


def test_assignment_stores_in_workspace_registry(qt_app) -> None:
    widget, registry, captured = _build_repl(qt_app)
    widget.execute("x = 3")
    assert registry.get("x") == 3
    assert captured.get("x") == 3


def test_history_recorded(qt_app) -> None:
    widget, _, _ = _build_repl(qt_app)
    widget.execute("a = 1")
    widget.execute("b = 2")
    history = widget.history()
    assert history[-2:] == ("a = 1", "b = 2")


def test_exception_displays_without_crashing(qt_app) -> None:
    widget, _, _ = _build_repl(qt_app)
    widget.execute("raise ValueError('boom')")
    output = widget.output_text()
    assert "ValueError" in output
    assert "boom" in output


def test_registry_required(qt_app) -> None:
    from upstream_drift_tools.ui.tools_sidebar.runtime_tabs import PythonReplWidget

    with pytest.raises((TypeError, ValueError)):
        PythonReplWidget(registry=None, set_variable=lambda *_: None)


def test_set_variable_required(qt_app) -> None:
    from upstream_drift_tools.ui.tools_sidebar.registry import WorkspaceRegistry
    from upstream_drift_tools.ui.tools_sidebar.runtime_tabs import PythonReplWidget

    with pytest.raises((TypeError, ValueError)):
        PythonReplWidget(registry=WorkspaceRegistry(), set_variable=None)
