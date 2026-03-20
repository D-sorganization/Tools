"""Minimum test-contract coverage for shared data_processing package."""

from __future__ import annotations

import pandas as pd
import pytest
from data_processing.processor import DataProcessor


def test_data_processing_unknown_filter_contract() -> None:
    """DbC: unknown filter types are rejected at API boundary."""
    processor = DataProcessor()
    processor.load_dataframe(pd.DataFrame({"signal": [1.0, 2.0, 3.0]}))

    with pytest.raises(ValueError, match="Unknown filter type"):
        processor.apply_filter("not-a-filter", columns=["signal"])
