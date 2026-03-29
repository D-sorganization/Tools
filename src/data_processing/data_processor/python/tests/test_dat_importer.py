"""Tests for dat_importer -- DAT/DBF file import for industrial data.

Covers: read_dat_file, get_dat_columns, import_dat_with_tags,
        export_dat_to_csv, preview_dat_file, detect_dat_delimiter,
        get_dat_file_info.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_processor.core.dat_importer import (
    detect_dat_delimiter,
    export_dat_to_csv,
    get_dat_columns,
    get_dat_file_info,
    import_dat_with_tags,
    preview_dat_file,
    read_dat_file,
)


@pytest.fixture()
def sample_dat(tmp_path: Path) -> Path:
    """Create a sample tab-separated DAT file."""
    dat_file = tmp_path / "test.dat"
    dat_file.write_text(
        "Time\tTemperature\tPressure\n"
        "0.0\t100.5\t1.01\n"
        "1.0\t101.2\t1.02\n"
        "2.0\t99.8\t1.00\n"
        "3.0\t102.0\t1.03\n",
        encoding="utf-8",
    )
    return dat_file


@pytest.fixture()
def csv_dat(tmp_path: Path) -> Path:
    """Create a comma-separated DAT file."""
    dat_file = tmp_path / "comma.dat"
    dat_file.write_text(
        "Time,Temperature,Pressure\n"
        "0.0,100.5,1.01\n"
        "1.0,101.2,1.02\n",
        encoding="utf-8",
    )
    return dat_file


class TestReadDatFile:
    """Test read_dat_file function."""

    def test_read_basic(self, sample_dat: Path) -> None:
        df = read_dat_file(sample_dat)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "Temperature" in df.columns

    def test_read_string_path(self, sample_dat: Path) -> None:
        df = read_dat_file(str(sample_dat))
        assert len(df) == 4

    def test_read_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            read_dat_file(tmp_path / "nonexistent.dat")

    def test_read_with_custom_delimiter(self, csv_dat: Path) -> None:
        df = read_dat_file(csv_dat, delimiter=",")
        assert len(df) == 2
        assert "Pressure" in df.columns


class TestGetDatColumns:
    """Test get_dat_columns function."""

    def test_get_columns(self, sample_dat: Path) -> None:
        cols = get_dat_columns(sample_dat)
        assert cols == ["Time", "Temperature", "Pressure"]

    def test_get_columns_nrows(self, sample_dat: Path) -> None:
        cols = get_dat_columns(sample_dat, nrows=1)
        assert len(cols) == 3


class TestImportDatWithTags:
    """Test import_dat_with_tags function."""

    def test_import_all(self, sample_dat: Path) -> None:
        df = import_dat_with_tags(sample_dat)
        assert len(df.columns) == 3

    def test_import_selected_tags(self, sample_dat: Path) -> None:
        df = import_dat_with_tags(sample_dat, selected_tags=["Temperature"])
        assert "Temperature" in df.columns
        assert "Time" in df.columns  # Time is always preserved

    def test_import_nonexistent_tags_returns_time_only(self, sample_dat: Path) -> None:
        # Tags that don't exist are silently skipped; Time column is always preserved
        df = import_dat_with_tags(sample_dat, selected_tags=["nonexistent"])
        assert "Time" in df.columns
        assert "nonexistent" not in df.columns


class TestExportDatToCsv:
    """Test export_dat_to_csv function."""

    def test_export(self, sample_dat: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.csv"
        result = export_dat_to_csv(sample_dat, output)
        assert result == output
        assert output.exists()

        # Verify CSV content
        df = pd.read_csv(output)
        assert len(df) == 4
        assert "Temperature" in df.columns


class TestPreviewDatFile:
    """Test preview_dat_file function."""

    def test_preview_default(self, sample_dat: Path) -> None:
        df = preview_dat_file(sample_dat)
        assert len(df) == 4  # File has only 4 rows

    def test_preview_limited(self, sample_dat: Path) -> None:
        df = preview_dat_file(sample_dat, nrows=2)
        assert len(df) == 2


class TestDetectDatDelimiter:
    """Test detect_dat_delimiter function."""

    def test_detect_tab(self, sample_dat: Path) -> None:
        delim = detect_dat_delimiter(sample_dat)
        assert delim == "\t"

    def test_detect_comma(self, csv_dat: Path) -> None:
        delim = detect_dat_delimiter(csv_dat)
        assert delim == ","


class TestGetDatFileInfo:
    """Test get_dat_file_info function."""

    def test_file_info(self, sample_dat: Path) -> None:
        info = get_dat_file_info(sample_dat)
        assert info["column_count"] == 3
        assert info["has_time_column"] is True
        assert info["file_size_bytes"] > 0
        assert "columns" in info
