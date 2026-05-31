# ruff: noqa: E501
"""Headless tests for ``WorkspaceCommandHandler`` (ADR-0022 / issue #6119).

No ``QApplication`` and no widget instantiation — the controller is driven
through a fake ``emit`` sink and fake provider/sink collaborators. This
sidesteps the Sidekick multi-widget Qt segfault while still covering the full
``/ws.read``, ``/ws.write`` and ``/plot`` routing contract.
"""

from __future__ import annotations

import pytest

from chat._workspace_protocol import WorkspaceVariableInfo
from chat.workspace_command_handler import (
    WorkspaceCommandHandler,
    build_workspace_context_block,
)


class _Recorder:
    """Captures emitted assistant-bubble text."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, text: str) -> None:
        self.messages.append(text)


class _FakeProvider:
    def __init__(self, *, store: dict | None = None, variables: list | None = None):
        self.store = store if store is not None else {}
        self._variables = variables or []
        self.writes: list[tuple[str, object]] = []

    def describe(self) -> list:
        return self._variables

    def read(self, name: str):
        return self.store[name]  # raises KeyError when absent

    def write(self, name: str, value) -> None:
        self.writes.append((name, value))
        self.store[name] = value


def test_emit_must_be_callable() -> None:
    with pytest.raises(TypeError):
        WorkspaceCommandHandler(emit="not callable")  # type: ignore[arg-type]


def test_dispatch_rejects_unknown_command() -> None:
    handler = WorkspaceCommandHandler(emit=_Recorder())
    with pytest.raises(ValueError, match="unknown workspace command"):
        handler.dispatch("nope", "")


def test_context_block_empty_without_provider() -> None:
    assert build_workspace_context_block(None) == ""
    handler = WorkspaceCommandHandler(emit=_Recorder(), provider=None)
    assert handler.context_block() == ""


def test_context_block_lists_variables() -> None:
    var = WorkspaceVariableInfo(
        name="x", dtype="float64", shape=(3, 2), preview="[[1,2],...]"
    )
    handler = WorkspaceCommandHandler(
        emit=_Recorder(), provider=_FakeProvider(variables=[var])
    )
    block = handler.context_block()
    assert block.startswith("Available workspace variables:")
    assert "- x: float64, shape (3, 2)" in block


def test_ws_read_usage_when_no_arg() -> None:
    rec = _Recorder()
    WorkspaceCommandHandler(emit=rec, provider=_FakeProvider()).handle_ws_read("  ")
    assert rec.messages == ["Usage: /ws.read NAME"]


def test_ws_read_no_provider() -> None:
    rec = _Recorder()
    WorkspaceCommandHandler(emit=rec, provider=None).handle_ws_read("x")
    assert rec.messages == ["Workspace bridge not available in this chat."]


def test_ws_read_not_found() -> None:
    rec = _Recorder()
    WorkspaceCommandHandler(emit=rec, provider=_FakeProvider()).handle_ws_read("x")
    assert rec.messages == ["Workspace variable not found: x"]


def test_ws_read_success_and_truncation() -> None:
    rec = _Recorder()
    provider = _FakeProvider(store={"short": 42, "long": "y" * 500})
    handler = WorkspaceCommandHandler(emit=rec, provider=provider)
    handler.handle_ws_read("short")
    assert rec.messages[-1] == "short = 42"
    handler.handle_ws_read("long")
    # repr of the long string is truncated to <= 200 chars total and ends "..."
    out = rec.messages[-1]
    assert out.startswith("long = ")
    assert out.endswith("...")
    assert len(out.split(" = ", 1)[1]) == 200


def test_ws_write_usage() -> None:
    rec = _Recorder()
    WorkspaceCommandHandler(emit=rec, provider=_FakeProvider()).handle_ws_write("x")
    assert rec.messages == ["Usage: /ws.write NAME JSON_VALUE"]


def test_ws_write_bad_json() -> None:
    rec = _Recorder()
    WorkspaceCommandHandler(emit=rec, provider=_FakeProvider()).handle_ws_write(
        "x notjson"
    )
    assert rec.messages[-1].startswith("Could not parse JSON value:")


def test_ws_write_success() -> None:
    rec = _Recorder()
    provider = _FakeProvider()
    handler = WorkspaceCommandHandler(emit=rec, provider=provider)
    handler.handle_ws_write("x [1, 2, 3]")
    assert provider.writes == [("x", [1, 2, 3])]
    assert rec.messages[-1] == "Wrote workspace variable: x"


def test_ws_write_no_provider() -> None:
    rec = _Recorder()
    WorkspaceCommandHandler(emit=rec, provider=None).handle_ws_write("x 1")
    assert rec.messages == ["Workspace bridge not available in this chat."]


def test_plot_usage() -> None:
    rec = _Recorder()
    WorkspaceCommandHandler(emit=rec, plot_sink=lambda spec: None).handle_plot("  ")
    assert rec.messages == ["Usage: /plot {json plot spec}"]


def test_plot_no_sink() -> None:
    rec = _Recorder()
    WorkspaceCommandHandler(emit=rec, plot_sink=None).handle_plot('{"a": 1}')
    assert rec.messages == ["Plot tab not available in this chat."]


def test_plot_bad_json() -> None:
    rec = _Recorder()
    WorkspaceCommandHandler(emit=rec, plot_sink=lambda spec: None).handle_plot("{bad")
    assert rec.messages[-1].startswith("Could not parse plot spec JSON:")


def test_plot_success_routes_through_dispatch() -> None:
    rec = _Recorder()
    seen: list[object] = []
    handler = WorkspaceCommandHandler(emit=rec, plot_sink=seen.append)
    handler.dispatch("plot", '{"kind": "line"}')
    assert seen == [{"kind": "line"}]
    assert rec.messages[-1] == "Plot request submitted."


def test_plot_sink_failure_is_reported() -> None:
    rec = _Recorder()

    def boom(_spec: object) -> None:
        raise RuntimeError("sink down")

    WorkspaceCommandHandler(emit=rec, plot_sink=boom).handle_plot('{"a": 1}')
    assert rec.messages[-1] == "Plot request failed: sink down"
