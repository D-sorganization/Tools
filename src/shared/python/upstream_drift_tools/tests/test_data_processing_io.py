"""Tests for data_processing.io - DataReader, DataWriter, FileFormatDetector.

Covers all supported formats using tempfiles:
- CSV, TSV, JSON, Numpy, SQLite, Pickle
- FileFormatDetector extension mapping
- Error paths: unsupported format, parquet/matlab dependency checks
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small DataFrame used by all write/read round-trip tests."""
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


# ---------------------------------------------------------------------------
# DataReader
# ---------------------------------------------------------------------------


class TestDataReader:
    """Test DataReader.read_file() for each supported format."""

    def test_read_csv(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            sample_df.to_csv(f.name, index=False)
            result = DataReader.read_file(f.name)
        assert list(result.columns) == ["x", "y"]
        assert len(result) == 3

    def test_read_csv_with_explicit_format(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".data", delete=False) as f:
            sample_df.to_csv(f.name, index=False)
            result = DataReader.read_file(f.name, format_type="csv")
        assert list(result.columns) == ["x", "y"]

    def test_read_tsv(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
            sample_df.to_csv(f.name, sep="\t", index=False)
            result = DataReader.read_file(f.name)
        assert list(result.columns) == ["x", "y"]

    def test_read_json(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            sample_df.to_json(f.name, orient="records")
            result = DataReader.read_file(f.name)
        assert "x" in result.columns

    def test_read_numpy(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            arr = sample_df.values
            np.save(f.name, arr)
            result = DataReader.read_file(f.name)
        assert result.shape == (3, 2)

    def test_read_pickle(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            sample_df.to_pickle(f.name)
            result = DataReader.read_file(f.name)
        assert list(result.columns) == ["x", "y"]

    def test_read_sqlite(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            conn = sqlite3.connect(f.name)
            sample_df.to_sql("data", conn, if_exists="replace", index=False)
            conn.close()
            result = DataReader.read_file(f.name)
        assert list(result.columns) == ["x", "y"]

    def test_read_unsupported_format_raises(self):
        from upstream_drift_tools.data_processing.io import DataReader

        with pytest.raises(ValueError, match="Unsupported or undetected"):
            DataReader.read_file("/tmp/file.xyz_unknown_ext")

    def test_read_parquet_no_pyarrow_raises(self):
        """If PYARROW_AVAILABLE is False, reading parquet should raise ImportError."""
        import upstream_drift_tools.data_processing.io as io_mod
        from upstream_drift_tools.data_processing.io import DataReader

        original = io_mod.PYARROW_AVAILABLE
        try:
            io_mod.PYARROW_AVAILABLE = False
            with pytest.raises(ImportError, match="PyArrow"):
                DataReader.read_file("/tmp/fake.parquet")
        finally:
            io_mod.PYARROW_AVAILABLE = original


# ---------------------------------------------------------------------------
# DataWriter
# ---------------------------------------------------------------------------


class TestDataWriter:
    """Test DataWriter.write_file() for each supported format."""

    def test_write_csv(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            DataWriter.write_file(sample_df, f.name)
            result = pd.read_csv(f.name)
        assert list(result.columns) == ["x", "y"]

    def test_write_tsv(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
            DataWriter.write_file(sample_df, f.name)
            result = pd.read_csv(f.name, sep="\t")
        assert list(result.columns) == ["x", "y"]

    def test_write_json(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            DataWriter.write_file(sample_df, f.name)
            result = pd.read_json(f.name)
        assert "x" in result.columns

    def test_write_pickle(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            DataWriter.write_file(sample_df, f.name)
            result = pd.read_pickle(f.name)
        assert list(result.columns) == ["x", "y"]

    def test_write_numpy(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            DataWriter.write_file(sample_df, f.name)
            arr = np.load(f.name, allow_pickle=False)
        assert arr.shape == (3, 2)

    def test_write_sqlite(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            DataWriter.write_file(sample_df, f.name)
            conn = sqlite3.connect(f.name)
            result = pd.read_sql("SELECT * FROM data", conn)
            conn.close()
        assert list(result.columns) == ["x", "y"]

    def test_write_unsupported_format_raises(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with pytest.raises(ValueError, match="Unsupported or undetected"):
            DataWriter.write_file(sample_df, "/tmp/output.xyz_unknown_ext")

    def test_write_creates_parent_dirs(self, sample_df: pd.DataFrame, tmp_path: Path):
        from upstream_drift_tools.data_processing.io import DataWriter

        nested = tmp_path / "a" / "b" / "c" / "out.csv"
        DataWriter.write_file(sample_df, nested)
        assert nested.exists()

    def test_write_parquet_no_pyarrow_raises(self, sample_df: pd.DataFrame):
        import upstream_drift_tools.data_processing.io as io_mod
        from upstream_drift_tools.data_processing.io import DataWriter

        original = io_mod.PYARROW_AVAILABLE
        try:
            io_mod.PYARROW_AVAILABLE = False
            with pytest.raises(ImportError, match="PyArrow"):
                DataWriter.write_file(sample_df, "/tmp/fake.parquet")
        finally:
            io_mod.PYARROW_AVAILABLE = original


# ---------------------------------------------------------------------------
# FileFormatDetector
# ---------------------------------------------------------------------------


class TestFileFormatDetector:
    """Cover detect_format and get_supported_extensions."""

    @pytest.mark.parametrize(
        "ext,expected",
        [
            (".csv", "csv"),
            (".tsv", "tsv"),
            (".txt", "tsv"),
            (".xlsx", "excel"),
            (".xls", "excel"),
            (".json", "json"),
            (".pkl", "pickle"),
            (".pickle", "pickle"),
            (".npy", "numpy"),
            (".mat", "matlab"),
            (".db", "sqlite"),
            (".sqlite", "sqlite"),
        ],
    )
    def test_detect_known_extensions(self, ext: str, expected: str):
        from upstream_drift_tools.data_processing.io import FileFormatDetector

        result = FileFormatDetector.detect_format(f"/data/file{ext}")
        assert result == expected

    def test_detect_unknown_extension_returns_none(self):
        from upstream_drift_tools.data_processing.io import FileFormatDetector

        assert FileFormatDetector.detect_format("/data/file.xyz_unknown") is None

    def test_get_supported_extensions_returns_list(self):
        from upstream_drift_tools.data_processing.io import FileFormatDetector

        extensions = FileFormatDetector.get_supported_extensions()
        assert isinstance(extensions, list)
        assert ".csv" in extensions
        assert ".json" in extensions
