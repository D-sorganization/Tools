"""Additional behavioural tests for ``upstream_drift_tools.data_processing.io``.

These tests target real edge-cases not covered in :mod:`test_io.py`:

* Pickle CWE-502 rejection (read AND write paths).
* Detection on PosixPath/WindowsPath with mixed-case suffix.
* ``detect_format`` argument validation.
* Empty/None DataFrame in writer.
* Excel round-trip and TSV with explicit format override.
* Reader explicit format override beating mismatched extension.
* Reading an unrecognised extension with explicit override.
* ``get_supported_extensions`` returns a stable list (no duplicates,
  every entry has a leading dot).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
import pandas as pd
from upstream_drift_tools.data_processing.io import (
    DataReader,
    DataWriter,
    FileFormatDetector,
)

# ── FileFormatDetector argument validation ───────────────────────────────


class TestDetectFormatValidation:
    def test_none_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="file_path"):
            FileFormatDetector.detect_format(None)  # type: ignore[arg-type]

    def test_path_with_mixed_case_suffix(self) -> None:
        # Suffix ".CSV", ".CsV" etc. all map to csv.
        assert FileFormatDetector.detect_format("dir/file.CsV") == "csv"
        assert FileFormatDetector.detect_format("dir/file.JsOn") == "json"

    def test_path_without_suffix(self) -> None:
        assert FileFormatDetector.detect_format("README") is None
        assert FileFormatDetector.detect_format("data") is None

    def test_path_with_multiple_dots(self) -> None:
        # Only the final suffix is considered.
        assert FileFormatDetector.detect_format("archive.tar.csv") == "csv"
        assert FileFormatDetector.detect_format("archive.csv.json") == "json"

    def test_supported_extensions_well_formed(self) -> None:
        exts = FileFormatDetector.get_supported_extensions()
        assert len(exts) == len(set(exts)), "duplicate extension entries"
        assert all(e.startswith(".") for e in exts), "extensions must start with a dot"
        assert all(e == e.lower() for e in exts), "extensions must be lower-case"

    def test_detect_handles_pure_path(self) -> None:
        assert FileFormatDetector.detect_format(Path("/tmp/x.json")) == "json"


# ── Pickle is forbidden in both reader and writer ────────────────────────


class TestPickleSecurity:
    """The module documents CWE-502 mitigation; both ends must refuse pickle."""

    def test_reader_rejects_pickle_format(self, tmp_path: Path) -> None:
        path = tmp_path / "data.bin"
        path.write_bytes(b"not really pickle")
        with pytest.raises(ValueError, match="Pickle"):
            DataReader.read_file(path, format_type="pickle")

    def test_writer_rejects_pickle_format(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="Pickle"):
            DataWriter.write_file(df, tmp_path / "out.bin", format_type="pickle")


# ── Writer rejects None DataFrame ────────────────────────────────────────


class TestWriterValidation:
    def test_none_dataframe_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="df"):
            DataWriter.write_file(None, tmp_path / "out.csv")  # type: ignore[arg-type]


# ── Explicit format override ─────────────────────────────────────────────


class TestExplicitFormatOverride:
    def test_explicit_tsv_format_overrides_csv_extension(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "looks_like.csv"
        DataWriter.write_file(df, path, format_type="tsv")
        # File is actually tab-separated despite .csv suffix.
        text = path.read_text()
        assert "\t" in text
        # Reading with the same explicit format must round-trip.
        result = DataReader.read_file(path, format_type="tsv")
        pd.testing.assert_frame_equal(df, result)

    def test_explicit_format_with_unknown_extension(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1]})
        path = tmp_path / "file.weird"
        DataWriter.write_file(df, path, format_type="csv")
        result = DataReader.read_file(path, format_type="csv")
        pd.testing.assert_frame_equal(df, result)

    def test_explicit_unknown_format_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Unsupported"):
            DataWriter.write_file(df, tmp_path / "out.csv", format_type="bogus")


# ── Round-trip extras ────────────────────────────────────────────────────


class TestExcelRoundTrip:
    """Excel round-trip — only when ``openpyxl`` is installed."""

    def test_excel_roundtrip(self, tmp_path: Path) -> None:
        pytest.importorskip("openpyxl")
        df = pd.DataFrame({"col": [1, 2, 3], "name": ["a", "b", "c"]})
        path = tmp_path / "data.xlsx"
        DataWriter.write_file(df, path)
        result = DataReader.read_file(path)
        pd.testing.assert_frame_equal(df, result)


class TestTSVDetected:
    """``.txt`` should be auto-detected as TSV per the format map."""

    def test_txt_treated_as_tsv(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "data.txt"
        DataWriter.write_file(df, path)
        # Confirm real tab separation.
        assert "\t" in path.read_text()
        result = DataReader.read_file(path)
        pd.testing.assert_frame_equal(df, result)


# ── Reader: format auto-detection failure with no override ───────────────


class TestUnknownFormat:
    def test_no_extension_no_override_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "noext"
        path.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="Unsupported"):
            DataReader.read_file(path)
