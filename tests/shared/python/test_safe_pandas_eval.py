"""Tests for safe pandas formula validation and DataProcessor integration."""

from __future__ import annotations

import logging

import pandas as pd
import pytest
from data_processing.processor import DataProcessor
from safe_pandas_eval import log_formula_rejected, validate_pandas_formula


def test_validate_pandas_formula_accepts_numeric_boolean_expression() -> None:
    validate_pandas_formula(
        "(temperature + pressure / 2) > 30 and enabled",
        allowed_columns={"temperature", "pressure", "enabled"},
    )


@pytest.mark.parametrize(
    ("expression", "match"),
    [
        ("", "non-empty"),
        ("value + missing", "Unknown formula column: missing"),
        ("value.__class__", "forbidden pattern"),
        ("abs(value)", "Unsupported formula syntax: Call"),
        ("value[0]", "Unsupported formula syntax: Subscript"),
        ("value ** scale", "Formula exponent must be a numeric constant"),
        ("value ** 7", "Formula exponent is too large"),
        ("label == 'ready'", "Formula constants must be numeric or boolean"),
    ],
)
def test_validate_pandas_formula_rejects_unsafe_or_unsupported_syntax(
    expression: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_pandas_formula(
            expression,
            allowed_columns={"value", "scale", "label"},
        )


def test_validate_pandas_formula_rejects_overly_complex_expression() -> None:
    expression = " + ".join(["v"] * 45)

    with pytest.raises(ValueError, match="too complex"):
        validate_pandas_formula(expression, allowed_columns={"v"})


def test_log_formula_rejected_omits_expression_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="safe_pandas_eval"):
        log_formula_rejected("secret_column + 1", ValueError("Unknown formula column"))

    record = caplog.records[0]
    assert record.message == "Rejected pandas formula expression"
    assert record.formula_length == len("secret_column + 1")
    assert record.reason == "Unknown formula column"
    assert "secret_column" not in caplog.text


def test_data_processor_apply_formula_uses_validated_expression() -> None:
    processor = DataProcessor().load_dataframe(
        pd.DataFrame(
            {
                "distance": [10.0, 22.5, 40.0],
                "time": [2.0, 3.0, 5.0],
            }
        )
    )

    result = processor.apply_formula("speed", "distance / time")

    assert result is processor
    assert processor.dataframe["speed"].tolist() == [5.0, 7.5, 8.0]
    assert processor.history[-1] == "Created column 'speed' = distance / time"


def test_data_processor_apply_formula_rejects_unknown_columns() -> None:
    processor = DataProcessor().load_dataframe(pd.DataFrame({"value": [1, 2, 3]}))

    with pytest.raises(ValueError, match="Unknown formula column: missing"):
        processor.apply_formula("bad", "value + missing")

    assert "bad" not in processor.dataframe.columns
