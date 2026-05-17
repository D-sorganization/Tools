"""Tests for ``_gather_session_context`` covering Tools #2747.

The reporting tab's UI label promises that the report aggregates
"workspace context, chat interactions, and terminal history".  The
historical gatherer only returned workspace variables and the project
root.  These tests pin the extended contract: the gatherer must
collect snapshots from the chat, terminal, calculator, and
data_processor sub-tabs when those tabs are present and expose a
``get_context_snapshot`` method.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sidekick.ui.tools_sidebar.reporting_tab import _gather_session_context


class _FakeRegistry:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def variables(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=n) for n in self._names]


class _SnapshotTab:
    """Stand-in for a sub-tab widget that exposes a snapshot method."""

    def __init__(self, snapshot: dict[str, Any]) -> None:
        self._snapshot = snapshot
        self.calls = 0

    def get_context_snapshot(self) -> dict[str, Any]:
        self.calls += 1
        return dict(self._snapshot)


class _BoomTab:
    """Sub-tab whose snapshot method raises — must not break gather."""

    def get_context_snapshot(self) -> dict[str, Any]:
        raise RuntimeError("snapshot failure")


class _OpaqueTab:
    """Sub-tab with no snapshot method — must be skipped silently."""


def _build_sidebar(
    *,
    variables: list[str] | None = None,
    project_root: str = "/tmp/project",
    tab_widgets: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        registry=_FakeRegistry(variables or []),
        project_root=project_root,
        _tab_widgets=tab_widgets or {},
    )


def test_base_context_still_present() -> None:
    """Existing workspace + project_root keys must not regress."""
    sidebar = _build_sidebar(variables=["alpha", "beta"], project_root="/x")
    ctx = _gather_session_context(sidebar)
    assert ctx["workspace_variables"] == ["alpha", "beta"]
    assert ctx["project_root"] == "/x"


@pytest.mark.parametrize(
    "tab_id",
    ["chat", "terminal", "calculator", "data_processor"],
)
def test_snapshot_collected_for_each_known_subtab(tab_id: str) -> None:
    """Each advertised sub-tab's snapshot must appear in the context."""
    snap = {"sample": [1, 2, 3], "tab": tab_id}
    sidebar = _build_sidebar(tab_widgets={tab_id: _SnapshotTab(snap)})
    ctx = _gather_session_context(sidebar)
    assert tab_id in ctx, f"missing {tab_id!r} entry in {sorted(ctx)}"
    assert ctx[tab_id] == snap


def test_all_subtabs_collected_simultaneously() -> None:
    chat = _SnapshotTab({"messages": ["hi", "hello"]})
    term = _SnapshotTab({"history": ["ls", "pwd"]})
    calc = _SnapshotTab({"expressions": ["1+1=2"]})
    data = _SnapshotTab({"dataset": "df.csv"})
    sidebar = _build_sidebar(
        tab_widgets={
            "chat": chat,
            "terminal": term,
            "calculator": calc,
            "data_processor": data,
        },
    )
    ctx = _gather_session_context(sidebar)
    assert ctx["chat"] == {"messages": ["hi", "hello"]}
    assert ctx["terminal"] == {"history": ["ls", "pwd"]}
    assert ctx["calculator"] == {"expressions": ["1+1=2"]}
    assert ctx["data_processor"] == {"dataset": "df.csv"}
    assert chat.calls == 1


def test_missing_subtab_is_skipped() -> None:
    """Absent sub-tabs must not appear in the context (no None pollution)."""
    sidebar = _build_sidebar(tab_widgets={"chat": _SnapshotTab({"m": []})})
    ctx = _gather_session_context(sidebar)
    assert "chat" in ctx
    assert "terminal" not in ctx
    assert "calculator" not in ctx
    assert "data_processor" not in ctx


def test_subtab_without_snapshot_method_is_skipped() -> None:
    sidebar = _build_sidebar(tab_widgets={"chat": _OpaqueTab()})
    ctx = _gather_session_context(sidebar)
    assert "chat" not in ctx
    # Base keys still survive
    assert "workspace_variables" in ctx
    assert "project_root" in ctx


def test_snapshot_exception_is_swallowed() -> None:
    """A failing snapshot must not break the whole gather."""
    sidebar = _build_sidebar(
        tab_widgets={
            "chat": _BoomTab(),
            "terminal": _SnapshotTab({"history": ["ok"]}),
        },
    )
    ctx = _gather_session_context(sidebar)
    # chat snapshot failed silently and does not poison the dict
    assert "chat" not in ctx
    # terminal snapshot succeeded
    assert ctx["terminal"] == {"history": ["ok"]}


def test_sidebar_without_tab_widgets_attribute() -> None:
    """Sidebars predating the registry must not raise."""
    sidebar = SimpleNamespace(
        registry=_FakeRegistry([]),
        project_root="/p",
    )
    ctx = _gather_session_context(sidebar)
    assert ctx["workspace_variables"] == []
    assert ctx["project_root"] == "/p"


def test_output_shape_matches_ui_label_promise() -> None:
    """The UI label promises workspace + chat + terminal at minimum."""
    sidebar = _build_sidebar(
        variables=["x"],
        tab_widgets={
            "chat": _SnapshotTab({"messages": []}),
            "terminal": _SnapshotTab({"history": []}),
        },
    )
    ctx = _gather_session_context(sidebar)
    expected = {"workspace_variables", "project_root", "chat", "terminal"}
    assert expected.issubset(ctx.keys())
