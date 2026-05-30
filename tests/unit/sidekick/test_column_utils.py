"""Tests for the shared _resolve_columns helper (issue #3104 F11 — DRY)."""

from __future__ import annotations

import pytest
from sidekick.ui.tools_sidebar._column_utils import _resolve_columns


class _DomainError(ValueError):
    """Stand-in domain error for tests."""


def _make_err(missing: list[str]) -> _DomainError:
    return _DomainError(f"Missing: {missing}")


@pytest.mark.unit
def test_returns_all_available_when_none_selected() -> None:
    result = _resolve_columns(["a", "b", "c"], None, _make_err)
    assert result == ["a", "b", "c"]


@pytest.mark.unit
def test_returns_all_available_when_empty_list() -> None:
    result = _resolve_columns(["x", "y"], [], _make_err)
    assert result == ["x", "y"]


@pytest.mark.unit
def test_returns_normalized_subset() -> None:
    result = _resolve_columns(["alpha", "beta", "gamma"], ["beta", "gamma"], _make_err)
    assert result == ["beta", "gamma"]


@pytest.mark.unit
def test_strips_whitespace_from_selected_columns() -> None:
    result = _resolve_columns(["col"], ["  col  "], _make_err)
    assert result == ["col"]


@pytest.mark.unit
def test_raises_domain_error_for_missing_column() -> None:
    with pytest.raises(_DomainError) as exc_info:
        _resolve_columns(["a", "b"], ["a", "z"], _make_err)
    assert "z" in str(exc_info.value)


@pytest.mark.unit
def test_raises_for_noncallable_make_error() -> None:
    with pytest.raises(TypeError):
        _resolve_columns(["a"], ["a"], None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_raises_for_non_list_available() -> None:
    with pytest.raises(TypeError):
        _resolve_columns("a,b", ["a"], _make_err)  # type: ignore[arg-type]


@pytest.mark.unit
def test_skips_blank_entries_in_selected() -> None:
    result = _resolve_columns(["a", "b"], ["a", "  ", ""], _make_err)
    assert result == ["a"]


@pytest.mark.unit
def test_data_explorer_service_uses_shared_helper() -> None:
    """_resolve_selected_columns in data_explorer_service delegates to shared helper."""
    from sidekick.ui.tools_sidebar import data_explorer_service as svc
    from sidekick.ui.tools_sidebar.data_explorer_service import (
        DataExplorerColumnSummary,
        DataExplorerError,
        DataExplorerPreview,
    )

    col = DataExplorerColumnSummary(name="val", dtype="float64", missing_count=0)
    preview = DataExplorerPreview(
        source_path="fake.csv",
        format="csv",
        columns=(col,),
        preview_rows=({"val": 1.0},),
        total_rows=1,
        total_columns=1,
        truncated=False,
        load_mode="full",
    )

    with pytest.raises(DataExplorerError):
        svc._resolve_selected_columns(preview, ["nonexistent"])  # noqa: SLF001


@pytest.mark.unit
def test_data_processor_tab_uses_shared_helper() -> None:
    """_resolve_selected_columns in data_processor_tab delegates to _resolve_columns."""
    try:
        from sidekick.ui.tools_sidebar import data_processor_tab as tab
    except ImportError:
        pytest.skip("Qt not available; skipping data_processor_tab integration test")

    with pytest.raises(tab.DataProcessorTabError):
        tab._resolve_selected_columns(["a", "b"], ["z"])  # noqa: SLF001
