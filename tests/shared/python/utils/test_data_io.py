"""Tests for upstream_drift_tools.data_io module.

Covers:
- read_data for CSV files
- read_data with Parquet preference
- write_data to CSV and Parquet
- TSV reading with correct delimiter
- Error handling (missing pandas, unsupported extension, file not found)
- also_csv option for write_data
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from upstream_drift_tools.data_io import read_data, write_data


def _pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False

# ── read_data CSV ────────────────────────────────────────────────────────


class TestReadDataCSV:
    """Test reading CSV files."""

    def test_read_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        df_expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_expected.to_csv(csv_path, index=False)

        df = read_data(csv_path)
        pd.testing.assert_frame_equal(df, df_expected)

    def test_read_csv_string_path(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"x": [10]}).to_csv(csv_path, index=False)
        df = read_data(str(csv_path))
        assert df["x"].iloc[0] == 10

    def test_read_csv_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_data(tmp_path / "nonexistent.csv")

    def test_read_tsv(self, tmp_path: Path) -> None:
        tsv_path = tmp_path / "data.tsv"
        df_expected = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df_expected.to_csv(tsv_path, index=False, sep="\t")

        df = read_data(tsv_path)
        pd.testing.assert_frame_equal(df, df_expected)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "data.xyz"
        bad_path.write_text("hello")
        with pytest.raises((ValueError, KeyError)):
            read_data(bad_path)


# ── read_data Parquet Preference ─────────────────────────────────────────


_skip_no_pyarrow = pytest.mark.skipif(
    not _pyarrow_available(), reason="pyarrow not installed"
)


@_skip_no_pyarrow
class TestReadDataParquetPreference:
    """Test Parquet sibling preference behavior."""

    def test_prefers_parquet_when_available(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        parquet_path = tmp_path / "data.parquet"
        df_csv = pd.DataFrame({"src": ["csv"]})
        df_parquet = pd.DataFrame({"src": ["parquet"]})
        df_csv.to_csv(csv_path, index=False)
        df_parquet.to_parquet(parquet_path, index=False)

        df = read_data(csv_path, prefer_parquet=True)
        assert df["src"].iloc[0] == "parquet"

    def test_falls_back_to_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        df_expected = pd.DataFrame({"src": ["csv"]})
        df_expected.to_csv(csv_path, index=False)

        df = read_data(csv_path, prefer_parquet=True)
        assert df["src"].iloc[0] == "csv"

    def test_prefer_parquet_false_reads_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        parquet_path = tmp_path / "data.parquet"
        pd.DataFrame({"src": ["csv"]}).to_csv(csv_path, index=False)
        pd.DataFrame({"src": ["parquet"]}).to_parquet(parquet_path, index=False)

        df = read_data(csv_path, prefer_parquet=False)
        assert df["src"].iloc[0] == "csv"

    def test_read_parquet_directly(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "data.parquet"
        df_expected = pd.DataFrame({"x": [1, 2, 3]})
        df_expected.to_parquet(parquet_path, index=False)

        df = read_data(parquet_path)
        pd.testing.assert_frame_equal(df, df_expected)

    def test_parquet_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_data(tmp_path / "missing.parquet")


# ── write_data ───────────────────────────────────────────────────────────


class TestWriteData:
    """Test write_data function."""

    def test_write_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "output.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = write_data(df, csv_path)
        assert result == csv_path
        assert csv_path.exists()
        df_read = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(df_read, df)

    @_skip_no_pyarrow
    def test_write_parquet(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "output.parquet"
        df = pd.DataFrame({"x": [10, 20, 30]})
        result = write_data(df, parquet_path)
        assert result == parquet_path
        assert parquet_path.exists()

    @_skip_no_pyarrow
    def test_write_parquet_also_csv(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "output.parquet"
        csv_path = tmp_path / "output.csv"
        df = pd.DataFrame({"x": [1]})
        write_data(df, parquet_path, also_csv=True)
        assert parquet_path.exists()
        assert csv_path.exists()

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested_path = tmp_path / "sub" / "dir" / "data.csv"
        df = pd.DataFrame({"val": [42]})
        write_data(df, nested_path)
        assert nested_path.exists()

    def test_roundtrip_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "roundtrip.csv"
        df = pd.DataFrame({"a": [1.5, 2.5], "b": ["x", "y"]})
        write_data(df, csv_path)
        df_read = read_data(csv_path)
        assert list(df_read.columns) == ["a", "b"]
        assert len(df_read) == 2

    @_skip_no_pyarrow
    def test_roundtrip_parquet(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "roundtrip.parquet"
        df = pd.DataFrame({"num": [100, 200], "text": ["a", "b"]})
        write_data(df, parquet_path)
        df_read = read_data(parquet_path)
        pd.testing.assert_frame_equal(df_read, df)
