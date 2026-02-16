"""Tests for python.src.utils.csv_utils module.

Covers:
- safe_read_csv (missing files, valid files)
- safe_write_csv (create parents, round-trip)
- read_csv_with_validation (required columns)
- merge_csv_files
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from utils.csv_utils import (
    merge_csv_files,
    read_csv_with_validation,
    safe_read_csv,
    safe_write_csv,
)


class TestSafeReadCsv:
    """Tests for safe_read_csv function."""

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = safe_read_csv(tmp_path / "nonexistent.csv")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_missing_file_returns_default(self, tmp_path: Path) -> None:
        default = pd.DataFrame({"a": [1]})
        result = safe_read_csv(tmp_path / "missing.csv", default=default)
        pd.testing.assert_frame_equal(result, default)

    def test_valid_file(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df.to_csv(csv_path, index=False)
        result = safe_read_csv(csv_path)
        assert list(result.columns) == ["x", "y"]
        assert len(result) == 3


class TestSafeWriteCsv:
    """Tests for safe_write_csv function."""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "output.csv"
        df = pd.DataFrame({"a": [1, 2]})
        assert safe_write_csv(df, csv_path, index=False)
        assert csv_path.exists()

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sub" / "dir" / "output.csv"
        df = pd.DataFrame({"a": [1]})
        assert safe_write_csv(df, csv_path, create_parents=True, index=False)
        assert csv_path.exists()

    def test_round_trip(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "round.csv"
        original = pd.DataFrame({"col1": [10, 20], "col2": ["a", "b"]})
        safe_write_csv(original, csv_path, index=False)
        loaded = safe_read_csv(csv_path)
        pd.testing.assert_frame_equal(loaded, original)


class TestReadCsvWithValidation:
    """Tests for read_csv_with_validation."""

    def test_valid_columns(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "valid.csv"
        df = pd.DataFrame({"name": ["a"], "value": [1]})
        df.to_csv(csv_path, index=False)
        result = read_csv_with_validation(csv_path, required_columns=["name", "value"])
        assert result is not None
        assert len(result) == 1

    def test_missing_column_returns_none(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "partial.csv"
        df = pd.DataFrame({"name": ["a"]})
        df.to_csv(csv_path, index=False)
        result = read_csv_with_validation(csv_path, required_columns=["name", "value"])
        assert result is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        result = read_csv_with_validation(csv_path)
        assert result is None

    def test_no_required_columns(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "any.csv"
        pd.DataFrame({"x": [1]}).to_csv(csv_path, index=False)
        result = read_csv_with_validation(csv_path)
        assert result is not None


class TestMergeCsvFiles:
    """Tests for merge_csv_files."""

    def test_merge_two_files(self, tmp_path: Path) -> None:
        f1 = tmp_path / "f1.csv"
        f2 = tmp_path / "f2.csv"
        out = tmp_path / "merged.csv"

        pd.DataFrame({"x": [1]}).to_csv(f1, index=False)
        pd.DataFrame({"x": [2]}).to_csv(f2, index=False)

        assert merge_csv_files([f1, f2], out)
        merged = pd.read_csv(out)
        assert len(merged) == 2

    def test_merge_empty_list_returns_false(self, tmp_path: Path) -> None:
        out = tmp_path / "empty_merge.csv"
        assert merge_csv_files([], out) is False

    def test_merge_nonexistent_files(self, tmp_path: Path) -> None:
        out = tmp_path / "bad_merge.csv"
        result = merge_csv_files(
            [tmp_path / "a.csv", tmp_path / "b.csv"], out
        )
        assert result is False
