"""Comprehensive tests for data_processing.io module.

Covers FileFormatDetector, DataReader, and DataWriter with round-trip
tests for CSV, TSV, JSON, and format detection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from upstream_drift_tools.data_processing.io import (
    DataReader,
    DataWriter,
    FileFormatDetector,
)

# ─── FileFormatDetector ──────────────────────────────────────


class TestFileFormatDetector:
    """Test file format detection from extensions."""

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("data.csv", "csv"),
            ("data.tsv", "tsv"),
            ("data.txt", "tsv"),
            ("data.xlsx", "excel"),
            ("data.xls", "excel"),
            ("data.json", "json"),
            ("data.npy", "numpy"),
            ("data.mat", "matlab"),
            ("data.db", "sqlite"),
            ("data.sqlite", "sqlite"),
            ("data.parquet", "parquet"),
            ("data.pq", "parquet"),
            ("data.h5", "hdf5"),
            ("data.hdf5", "hdf5"),
            ("data.feather", "feather"),
        ],
    )
    def test_known_extensions(self, filename: str, expected: str) -> None:
        assert FileFormatDetector.detect_format(filename) == expected

    def test_unknown_extension_returns_none(self) -> None:
        assert FileFormatDetector.detect_format("data.xyz") is None

    def test_case_insensitive(self) -> None:
        assert FileFormatDetector.detect_format("data.CSV") == "csv"

    def test_path_object(self) -> None:
        assert FileFormatDetector.detect_format(Path("dir/data.json")) == "json"

    def test_get_supported_extensions(self) -> None:
        exts = FileFormatDetector.get_supported_extensions()
        assert isinstance(exts, list)
        assert ".csv" in exts
        assert ".json" in exts
        assert len(exts) > 10


# ─── DataReader / DataWriter CSV Round-Trip ──────────────────


class TestCSVRoundTrip:
    def test_write_read_csv(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        path = tmp_path / "test.csv"
        DataWriter.write_file(df, path)
        result = DataReader.read_file(path)
        pd.testing.assert_frame_equal(df, result)

    def test_csv_explicit_format(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"x": [10]})
        path = tmp_path / "data.dat"
        DataWriter.write_file(df, path, format_type="csv")
        result = DataReader.read_file(path, format_type="csv")
        pd.testing.assert_frame_equal(df, result)


# ─── DataReader / DataWriter TSV Round-Trip ──────────────────


class TestTSVRoundTrip:
    def test_write_read_tsv(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"col1": ["a", "b"], "col2": [1, 2]})
        path = tmp_path / "test.tsv"
        DataWriter.write_file(df, path)
        result = DataReader.read_file(path)
        pd.testing.assert_frame_equal(df, result)


# ─── DataReader / DataWriter JSON Round-Trip ─────────────────


class TestJSONRoundTrip:
    def test_write_read_json(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        path = tmp_path / "test.json"
        DataWriter.write_file(df, path)
        result = DataReader.read_file(path)
        pd.testing.assert_frame_equal(df, result)


# ─── DataReader / DataWriter NumPy Round-Trip ────────────────


class TestNumpyRoundTrip:
    def test_write_read_numpy(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        path = tmp_path / "test.npy"
        DataWriter.write_file(df, path)
        result = DataReader.read_file(path)
        # Column names lost in numpy, compare values
        np.testing.assert_array_almost_equal(result.values, df.values)


# ─── DataReader / DataWriter SQLite Round-Trip ───────────────


class TestSQLiteRoundTrip:
    def test_write_read_sqlite(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        path = tmp_path / "test.db"
        DataWriter.write_file(df, path, table_name="mydata")
        result = DataReader.read_file(path, query="SELECT * FROM mydata")
        pd.testing.assert_frame_equal(df, result)


# ─── Error Handling ──────────────────────────────────────────


class TestErrorHandling:
    def test_read_unsupported_format_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "data.xyz"
        path.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            DataReader.read_file(path)

    def test_write_unsupported_format_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1]})
        path = tmp_path / "data.xyz"
        with pytest.raises(ValueError, match="Unsupported"):
            DataWriter.write_file(df, path)

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1]})
        path = tmp_path / "sub" / "dir" / "data.csv"
        DataWriter.write_file(df, path)
        assert path.exists()
