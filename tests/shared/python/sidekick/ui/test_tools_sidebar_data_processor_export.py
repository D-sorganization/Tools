"""Unit tests for the Qt-free export helpers in ``data_processor_tab``.

``export_data_processor_frame`` and its private helpers convert a tabular
frame (pandas DataFrame) into shared-workspace values without any Qt. These
tests exercise single/multi-column export, cell normalization, column
selection, and the structured ``DataProcessorTabError`` guards.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

from sidekick.ui.tools_sidebar import data_processor_tab as dpt  # noqa: E402
from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry  # noqa: E402


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


# ---------------------------------------------------------------------------
# export_data_processor_frame
# ---------------------------------------------------------------------------


def test_export_single_column_produces_list() -> None:
    registry = WorkspaceRegistry()
    var = dpt.export_data_processor_frame(
        _frame(), registry, "col_a", selected_columns=["a"]
    )
    assert var.name == "col_a"
    assert registry.get("col_a") == [1, 2, 3]


def test_export_multi_column_produces_records() -> None:
    registry = WorkspaceRegistry()
    dpt.export_data_processor_frame(_frame(), registry, "table")
    value = registry.get("table")
    assert value == [
        {"a": 1, "b": 4.0},
        {"a": 2, "b": 5.0},
        {"a": 3, "b": 6.0},
    ]


def test_export_blank_name_raises() -> None:
    with pytest.raises(dpt.DataProcessorTabError, match="non-empty"):
        dpt.export_data_processor_frame(_frame(), WorkspaceRegistry(), "   ")


def test_export_unknown_selected_column_raises() -> None:
    with pytest.raises(dpt.DataProcessorTabError, match="not available"):
        dpt.export_data_processor_frame(
            _frame(), WorkspaceRegistry(), "x", selected_columns=["nope"]
        )


# ---------------------------------------------------------------------------
# _current_frame
# ---------------------------------------------------------------------------


def test_current_frame_returns_engine_data() -> None:
    widget = SimpleNamespace(engine=SimpleNamespace(data=_frame()))
    frame = dpt._current_frame(widget)
    assert list(frame.columns) == ["a", "b"]


def test_current_frame_missing_data_raises() -> None:
    widget = SimpleNamespace(engine=SimpleNamespace(data=None))
    with pytest.raises(dpt.DataProcessorTabError, match="Load data"):
        dpt._current_frame(widget)


def test_current_frame_empty_columns_raises() -> None:
    widget = SimpleNamespace(engine=SimpleNamespace(data=pd.DataFrame()))
    with pytest.raises(dpt.DataProcessorTabError, match="Load data"):
        dpt._current_frame(widget)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_frame_columns_non_tabular_raises() -> None:
    with pytest.raises(dpt.DataProcessorTabError, match="not tabular"):
        dpt._frame_columns(SimpleNamespace(columns=None))


def test_resolve_selected_columns_defaults_to_all() -> None:
    assert dpt._resolve_selected_columns(["a", "b"], None) == ["a", "b"]


def test_resolve_selected_columns_filters_blanks() -> None:
    assert dpt._resolve_selected_columns(["a", "b"], ["a", "  "]) == ["a"]


def test_resolve_selected_columns_missing_raises() -> None:
    with pytest.raises(dpt.DataProcessorTabError, match="not available"):
        dpt._resolve_selected_columns(["a"], ["b"])


def test_frame_records_round_trip() -> None:
    records = dpt._frame_records(_frame(), ["a"])
    assert records == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_normalize_cell_unwraps_numpy_scalar() -> None:
    import numpy as np

    assert dpt._normalize_cell(np.int64(5)) == 5
    assert isinstance(dpt._normalize_cell(np.int64(5)), int)


def test_normalize_cell_passes_through_plain_values() -> None:
    assert dpt._normalize_cell("hello") == "hello"
    assert dpt._normalize_cell(3) == 3
