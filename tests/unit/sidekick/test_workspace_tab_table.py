"""Workspace tab table-model tests (UpstreamDrift #5616).

The MATLAB-style workspace tab uses ``QTableView`` with the columns
Name / Type / Size / Preview and updates automatically from the
``WorkspaceRegistry``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def _build_table(qt_app):
    from upstream_drift_tools.ui.tools_sidebar.default_tabs import (
        WorkspaceTableWidget,
    )
    from upstream_drift_tools.ui.tools_sidebar.registry import WorkspaceRegistry

    registry = WorkspaceRegistry()
    widget = WorkspaceTableWidget(registry=registry)
    return widget, registry


def test_columns_are_name_type_size_preview(qt_app) -> None:
    widget, _ = _build_table(qt_app)
    headers = widget.column_headers()
    assert headers == ("Name", "Type", "Size", "Preview")


def test_table_updates_when_registry_changes(qt_app) -> None:
    widget, registry = _build_table(qt_app)

    registry.set("alpha", 1)
    registry.set("beta", [1, 2, 3])

    rows = widget.row_data()
    names = [row[0] for row in rows]
    assert "alpha" in names
    assert "beta" in names


def test_table_sortable_by_name(qt_app) -> None:
    widget, registry = _build_table(qt_app)
    registry.set("zebra", 1)
    registry.set("apple", 2)

    widget.sort_by_column(0, ascending=True)
    rows = widget.row_data()
    names = [row[0] for row in rows]
    assert names == sorted(names)


def test_double_click_emits_inspect_request(qt_app) -> None:
    widget, registry = _build_table(qt_app)
    registry.set("x", [1, 2, 3])
    captured: list[str] = []
    widget.inspect_requested.connect(captured.append)

    widget.trigger_inspect("x")

    assert captured == ["x"]


def test_table_removes_row_when_variable_removed(qt_app) -> None:
    widget, registry = _build_table(qt_app)
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
