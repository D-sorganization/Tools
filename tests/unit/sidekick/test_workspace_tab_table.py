"""Workspace tab table-model tests (UpstreamDrift #5616).

The MATLAB-style workspace tab uses ``QTableView`` with the columns
Name / Type / Size / Preview and updates automatically from the
``WorkspaceRegistry``.
"""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = pytest.mark.serial

if sys.platform == "win32" and os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "Qt workspace tab table tests run serially on Windows.",
        allow_module_level=True,
    )

pytest.importorskip("PyQt6")


def _build_table(qt_app, qtbot):
    from upstream_drift_tools.ui.tools_sidebar.default_tabs import (
        WorkspaceTableWidget,
    )
    from upstream_drift_tools.ui.tools_sidebar.registry import WorkspaceRegistry

    registry = WorkspaceRegistry()
    widget = WorkspaceTableWidget(registry=registry)
    qtbot.addWidget(widget)
    return widget, registry


def test_columns_are_name_type_size_preview(qt_app, qtbot) -> None:
    widget, _ = _build_table(qt_app, qtbot)
    headers = widget.column_headers()
    assert headers == ("Name", "Type", "Size", "Preview")


def test_table_updates_when_registry_changes(qt_app, qtbot) -> None:
    widget, registry = _build_table(qt_app, qtbot)

    registry.set("alpha", 1)
    registry.set("beta", [1, 2, 3])

    rows = widget.row_data()
    names = [row[0] for row in rows]
    assert "alpha" in names
    assert "beta" in names


def test_table_sortable_by_name(qt_app, qtbot) -> None:
    widget, registry = _build_table(qt_app, qtbot)
    registry.set("zebra", 1)
    registry.set("apple", 2)

    widget.sort_by_column(0, ascending=True)
    rows = widget.row_data()
    names = [row[0] for row in rows]
    assert names == sorted(names)


def test_double_click_emits_inspect_request(qt_app, qtbot) -> None:
    widget, registry = _build_table(qt_app, qtbot)
    registry.set("x", [1, 2, 3])
    captured: list[str] = []
    widget.inspect_requested.connect(captured.append)

    widget.trigger_inspect("x")

    assert captured == ["x"]


def test_table_removes_row_when_variable_removed(qt_app, qtbot) -> None:
    widget, registry = _build_table(qt_app, qtbot)
    registry.set("a", 1)
    registry.set("b", 2)
    registry.remove("a")

    names = [row[0] for row in widget.row_data()]
    assert "a" not in names
    assert "b" in names


def test_registry_required(qt_app) -> None:
    from upstream_drift_tools.ui.tools_sidebar.default_tabs import (
        WorkspaceTableWidget,
    )

    with pytest.raises((TypeError, ValueError)):
        WorkspaceTableWidget(registry=None)  # type: ignore[arg-type]


def test_source_qualified_registry_alias_is_accepted(qt_app, qtbot) -> None:
    from sidekick.ui.tools_sidebar.workspace_tab import WorkspaceTableWidget

    from src.shared.python.sidekick.ui.tools_sidebar.registry import (
        WorkspaceRegistry,
    )

    registry = WorkspaceRegistry()
    widget = WorkspaceTableWidget(registry=registry)
    qtbot.addWidget(widget)

    registry.set("alias_ok", 1)

    assert ("alias_ok", "int", "", "1") in widget.row_data()
