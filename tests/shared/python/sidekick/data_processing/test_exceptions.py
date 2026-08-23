from __future__ import annotations

import pandas as pd
import pytest

from shared.python.sidekick.data_processing.core import DataProcessorEngine
from shared.python.sidekick.data_processing.exceptions import FilterError


@pytest.fixture
def engine_with_data() -> DataProcessorEngine:
    engine = DataProcessorEngine()
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "category": ["x", "y", "x", "y", "z"],
        }
    )
    engine.load_dataframe(df)
    return engine


def test_filter_data_valid_condition(engine_with_data: DataProcessorEngine) -> None:
    result = engine_with_data.filter_data("A", ">", 2)
    assert result.success is True
    assert len(engine_with_data.data) == 3
    assert list(engine_with_data.data["A"]) == [3, 4, 5]


def test_filter_data_rejects_unsafe_expression(
    engine_with_data: DataProcessorEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged = []

    def fake_log_rejected(expr, err):
        logged.append((expr, err))

    monkeypatch.setattr(
        "shared.python.sidekick.data_processing.core.log_formula_rejected",
        fake_log_rejected,
    )

    with pytest.raises(FilterError):
        engine_with_data.filter_data("A", "__import__", "os")


def test_query_valid_expression(engine_with_data: DataProcessorEngine) -> None:
    result = engine_with_data.query("A >= 3 and B <= 40")
    assert result.success is True
    assert len(engine_with_data.data) == 2
    assert list(engine_with_data.data["A"]) == [3, 4]


def test_query_rejects_malicious_expression(
    engine_with_data: DataProcessorEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged = []

    def fake_log_rejected(expr, err):
        logged.append((expr, err))

    monkeypatch.setattr(
        "shared.python.sidekick.data_processing.core.log_formula_rejected",
        fake_log_rejected,
    )

    with pytest.raises(FilterError):
        engine_with_data.query("__import__('os').system('echo pwned')")

    assert len(logged) > 0
    assert len(engine_with_data.data) == 5


def test_query_rejects_disallowed_columns(
    engine_with_data: DataProcessorEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged = []

    def fake_log_rejected(expr, err):
        logged.append((expr, err))

    monkeypatch.setattr(
        "shared.python.sidekick.data_processing.core.log_formula_rejected",
        fake_log_rejected,
    )

    with pytest.raises(FilterError):
        engine_with_data.query("non_existent_column > 10")

    assert len(logged) > 0
