"""TDD / DbC tests for DataLoader — contract enforcement on public API.

Issue #929: decompose long functions and adopt DbC checks in core pipeline.

These tests follow the Test-Driven Development pattern used across the fleet:
  - Each pre-condition has at least one passing and one failing test.
  - Contracts are verified via PreconditionError (or ValueError as fallback).
  - No mocking of internal logic; only the public interface is exercised.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

# ── Imports ──────────────────────────────────────────────────────────────────
# conftest.py ensures data_processor is on sys.path via utils.path_helpers.
from data_processor.contracts import PreconditionError
from data_processor.core.data_loader import DataLoader, load_csv_files

# ── Helpers ───────────────────────────────────────────────────────────────────


def _dummy_csv(tmp_path, filename: str = "test.csv") -> str:
    """Write a minimal CSV file and return its path."""
    p = tmp_path / filename
    p.write_text("time,A,B\n1,1.0,2.0\n2,3.0,4.0\n")
    return str(p)


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestDataLoaderInit:
    def test_default_uses_high_performance(self) -> None:
        loader = DataLoader()
        assert loader.use_high_performance is True
        assert loader.hp_loader is not None

    def test_low_performance_mode(self) -> None:
        loader = DataLoader(use_high_performance=False)
        assert loader.use_high_performance is False
        assert loader.hp_loader is None


# ── load_csv_file pre-conditions ──────────────────────────────────────────────


class TestLoadCsvFileContracts:
    """DbC: pre-conditions on load_csv_file."""

    def test_rejects_empty_string(self) -> None:
        loader = DataLoader(use_high_performance=False)
        with pytest.raises((PreconditionError, ValueError)):
            loader.load_csv_file("", validate_security=False)

    def test_rejects_whitespace_only(self) -> None:
        loader = DataLoader(use_high_performance=False)
        with pytest.raises((PreconditionError, ValueError)):
            loader.load_csv_file("   ", validate_security=False)

    def test_rejects_non_csv_extension(self) -> None:
        loader = DataLoader(use_high_performance=False)
        with pytest.raises((PreconditionError, ValueError)):
            loader.load_csv_file("bad_file.exe", validate_security=False)

    def test_accepts_csv_extension(self, tmp_path) -> None:
        loader = DataLoader(use_high_performance=False)
        path = _dummy_csv(tmp_path)
        result = loader.load_csv_file(path, validate_security=False)
        assert result is not None
        assert len(result) == 2

    def test_accepts_txt_extension(self, tmp_path) -> None:
        loader = DataLoader(use_high_performance=False)
        path = _dummy_csv(tmp_path, filename="data.txt")
        result = loader.load_csv_file(path, validate_security=False)
        assert result is not None

    def test_returns_none_for_missing_file(self, tmp_path) -> None:
        loader = DataLoader(use_high_performance=False)
        result = loader.load_csv_file(
            str(tmp_path / "missing.csv"), validate_security=False
        )
        assert result is None


# ── load_multiple_files pre-conditions ───────────────────────────────────────


class TestLoadMultipleFilesContracts:
    """DbC: pre-conditions on load_multiple_files."""

    def test_rejects_empty_list(self) -> None:
        loader = DataLoader(use_high_performance=False)
        with pytest.raises((PreconditionError, ValueError)):
            loader.load_multiple_files([])

    def test_rejects_list_with_empty_string(self) -> None:
        loader = DataLoader(use_high_performance=False)
        with pytest.raises((PreconditionError, ValueError)):
            loader.load_multiple_files([""])

    def test_accepts_valid_paths(self, tmp_path) -> None:
        loader = DataLoader(use_high_performance=False)
        path = _dummy_csv(tmp_path)
        result = loader.load_multiple_files([path])
        assert isinstance(result, dict)

    def test_combines_to_dataframe_when_requested(self, tmp_path) -> None:
        loader = DataLoader(use_high_performance=False)
        path = _dummy_csv(tmp_path)
        result = loader.load_multiple_files([path], combine=True)
        assert isinstance(result, pd.DataFrame)


# ── detect_signals pre-conditions ────────────────────────────────────────────


class TestDetectSignalsContracts:
    def test_rejects_empty_list(self) -> None:
        loader = DataLoader(use_high_performance=False)
        with pytest.raises((PreconditionError, ValueError)):
            loader.detect_signals([])

    def test_accepts_valid_paths(self, tmp_path) -> None:
        loader = DataLoader(use_high_performance=False)
        path = _dummy_csv(tmp_path)
        signals = loader.detect_signals([path])
        assert isinstance(signals, set)


# ── combine_dataframes pre-conditions ────────────────────────────────────────


class TestCombineDataFramesContracts:
    def test_rejects_invalid_how(self) -> None:
        loader = DataLoader(use_high_performance=False)
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"y": [3, 4]})
        with pytest.raises((PreconditionError, ValueError)):
            loader.combine_dataframes([df1, df2], how="invalid_join")

    def test_accepts_valid_how_values(self) -> None:
        loader = DataLoader(use_high_performance=False)
        df1 = pd.DataFrame({"x": [1, 2]}, index=[0, 1])
        df2 = pd.DataFrame({"y": [3, 4]}, index=[0, 1])
        for how in ("inner", "outer", "left", "right"):
            result = loader.combine_dataframes([df1, df2], how=how)
            assert isinstance(result, pd.DataFrame), f"how='{how}' failed"

    def test_returns_empty_df_for_empty_input(self) -> None:
        loader = DataLoader(use_high_performance=False)
        result = loader.combine_dataframes([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ── save_dataframe pre-conditions ────────────────────────────────────────────


class TestSaveDataFrameContracts:
    def test_rejects_empty_dataframe(self, tmp_path) -> None:
        loader = DataLoader(use_high_performance=False)
        with pytest.raises((PreconditionError, ValueError)):
            loader.save_dataframe(pd.DataFrame(), str(tmp_path / "out.csv"))

    def test_rejects_empty_output_path(self) -> None:
        loader = DataLoader(use_high_performance=False)
        df = pd.DataFrame({"x": [1]})
        with pytest.raises((PreconditionError, ValueError)):
            loader.save_dataframe(df, "")

    def test_rejects_unknown_format(self, tmp_path) -> None:
        loader = DataLoader(use_high_performance=False)
        df = pd.DataFrame({"x": [1]})
        with pytest.raises((PreconditionError, ValueError)):
            loader.save_dataframe(df, str(tmp_path / "out.json"), format_type="json")

    def test_saves_csv_successfully(self, tmp_path) -> None:
        loader = DataLoader(use_high_performance=False)
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        path = str(tmp_path / "output.csv")
        success = loader.save_dataframe(df, path, format_type="csv")
        assert success is True
        assert (tmp_path / "output.csv").exists()


# ── Convenience functions ─────────────────────────────────────────────────────


class TestConvenienceFunctions:
    def test_load_csv_files_returns_dict(self, tmp_path) -> None:
        path = _dummy_csv(tmp_path)
        result = load_csv_files([path])
        assert isinstance(result, dict)

    def test_detect_signals_from_files_returns_set(self, tmp_path) -> None:
        from data_processor.core.data_loader import detect_signals_from_files

        path = _dummy_csv(tmp_path)
        signals = detect_signals_from_files([path])
        assert isinstance(signals, set)


# ── detect_time_column ────────────────────────────────────────────────────────


class TestDetectTimeColumn:
    def test_detects_time_column_by_keyword(self) -> None:
        loader = DataLoader(use_high_performance=False)
        df = pd.DataFrame({"timestamp": [1, 2], "value": [3.0, 4.0]})
        col = loader.detect_time_column(df)
        assert col == "timestamp"

    def test_returns_none_for_no_time_column(self) -> None:
        loader = DataLoader(use_high_performance=False)
        df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.0]})
        col = loader.detect_time_column(df)
        assert col is None
