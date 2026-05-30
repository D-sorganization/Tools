"""Unit tests for the Sidekick Data Processor tab export logic.

Focus on the pure (non-Qt) helpers, especially ``_extract_frame`` which must
read the current data frame from *either* widget shape:

* the full Data Processor application widget (``processed_df`` / ``current_df``)
* the lightweight built-in widget (``engine.data``)

Related: D-sorganization/Tools#3111.
"""

from __future__ import annotations

import types

import pandas as pd
import pytest
from sidekick.ui.tools_sidebar import data_processor_tab as tab


class _FakeRegistry:
    """Minimal stand-in for the workspace registry used by the export path."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def set(self, name: str, value: object) -> types.SimpleNamespace:
        self.values[name] = value
        return types.SimpleNamespace(name=name, value=value)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"time": [0.0, 0.1], "signal": [1.0, 2.0]})


def test_extract_frame_prefers_processed_df() -> None:
    processed = _frame()
    widget = types.SimpleNamespace(processed_df=processed, current_df=_frame())
    assert tab._extract_frame(widget) is processed


def test_extract_frame_falls_back_to_current_df() -> None:
    current = _frame()
    widget = types.SimpleNamespace(processed_df=None, current_df=current)
    assert tab._extract_frame(widget) is current


def test_extract_frame_supports_engine_data_shape() -> None:
    data = _frame()
    widget = types.SimpleNamespace(engine=types.SimpleNamespace(data=data))
    assert tab._extract_frame(widget) is data


def test_extract_frame_returns_none_when_empty() -> None:
    assert tab._extract_frame(types.SimpleNamespace()) is None


def test_current_frame_raises_when_no_data() -> None:
    with pytest.raises(tab.DataProcessorTabError):
        tab._current_frame(types.SimpleNamespace())


def test_export_all_columns_round_trips() -> None:
    registry = _FakeRegistry()
    variable = tab.export_data_processor_frame(_frame(), registry, "result")
    assert variable.name == "result"
    assert registry.values["result"] == [
        {"time": 0.0, "signal": 1.0},
        {"time": 0.1, "signal": 2.0},
    ]


def test_export_single_column_yields_flat_list() -> None:
    registry = _FakeRegistry()
    tab.export_data_processor_frame(
        _frame(), registry, "sig", selected_columns=["signal"]
    )
    assert registry.values["sig"] == [1.0, 2.0]


def test_export_rejects_empty_variable_name() -> None:
    with pytest.raises(tab.DataProcessorTabError):
        tab.export_data_processor_frame(_frame(), _FakeRegistry(), "   ")


def test_export_rejects_unknown_column() -> None:
    with pytest.raises(tab.DataProcessorTabError):
        tab.export_data_processor_frame(
            _frame(), _FakeRegistry(), "x", selected_columns=["missing"]
        )
