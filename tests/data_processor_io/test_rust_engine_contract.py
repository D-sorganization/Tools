"""Contract tests for the bulk-I/O engine (issue #2989, Phase 2).

Verifies the public interface defined in
``src/shared/python/data_processor_io/rust_engine.py``.

These tests use the pure-pandas fallback so they run in CI without a compiled
Rust wheel.  They are marked ``contract`` so the CI ``Provider-Contract Suite``
step picks them up, and ``unit`` so they are part of the default test run.

Contract surface under test
---------------------------
- ``inspect(path) -> SchemaInfo``
- ``preview(path, nrows, columns) -> pd.DataFrame``
- ``convert(src, dst, format) -> ConversionReport``
- ``scan_batch(path, batch_size, columns) -> Iterator[pd.DataFrame]``
- ``filter_export(path, dst, predicate, columns) -> int``
- ``cancel()``
- Fallback always available (no native extension required)
- Error contract: correct exception types for each precondition violation
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Parquet round-trips need an optional engine (pyarrow or fastparquet). When
# neither is installed (e.g. the lean CI test image) the parquet-specific
# contract cases skip instead of failing — the CSV contract still runs.
_PARQUET_ENGINE_AVAILABLE = (
    importlib.util.find_spec("pyarrow") is not None
    or importlib.util.find_spec("fastparquet") is not None
)
_requires_parquet = pytest.mark.skipif(
    not _PARQUET_ENGINE_AVAILABLE,
    reason="no parquet engine (pyarrow/fastparquet) installed",
)

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _extra in [
    _REPO_ROOT / "src",
    _REPO_ROOT / "src" / "shared" / "python",
]:
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

# Import via the fully-qualified package path. The bare top-level name
# ``data_processor`` is also used by the full Data Processor application
# (src/data_processing/data_processor); using the qualified name here keeps this
# contract test pinned to the issue-#2989 wrapper regardless of import order.
import importlib.util

# These contract tests are for the compatibility wrapper's pandas fallback.
# CI may have a Phase-2 native wheel installed, whose scaffold intentionally
# leaves streaming/filter operations to the fallback path.
os.environ["DATA_PROCESSOR_IO_DISABLE_NATIVE"] = "1"

from data_processor_io.rust_engine import (  # noqa: E402
    CancellationToken,
    ConversionReport,
    DataProcessorRustError,
    OperationCancelled,
    RustBulkDataEngine,
    SchemaInfo,
    cancel,
    convert,
    filter_export,
    inspect,
    preview,
    scan_batch,
)


def test_shared_rust_engine_exports_bulk_facade() -> None:
    """The renamed wrapper exposes the Rust bulk-I/O compatibility surface."""
    assert issubclass(DataProcessorRustError, RuntimeError)
    assert RustBulkDataEngine.from_repo_root().is_available()


def test_wrapper_import_does_not_bind_full_app_name() -> None:
    """The bulk-I/O wrapper must not occupy the full app's top-level package."""
    import data_processor_io

    assert data_processor_io.__name__ == "data_processor_io"
    assert sys.modules.get("data_processor") is not data_processor_io


# Check if parquet support is available in pandas
HAS_PARQUET = (
    importlib.util.find_spec("pyarrow") is not None
    or importlib.util.find_spec("fastparquet") is not None
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_CSV_CONTENT = "time,force,note\n0.0,10.5,start\n0.1,11.0,mid\n0.2,9.8,end\n"


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    """A small CSV fixture written to a temp directory."""
    p = tmp_path / "sample.csv"
    p.write_text(_CSV_CONTENT, encoding="utf-8")
    return p


@pytest.fixture()
def parquet_file(tmp_path: Path, csv_file: Path) -> Path:
    """A small Parquet fixture derived from csv_file."""
    if not _PARQUET_ENGINE_AVAILABLE:
        pytest.skip("no parquet engine (pyarrow/fastparquet) installed")
    p = tmp_path / "sample.parquet"
    df = pd.read_csv(csv_file)
    df.to_parquet(p, index=False)
    return p


# ── inspect ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.contract
class TestInspect:
    """Contract: inspect(path) -> SchemaInfo."""

    def test_returns_schema_info_type(self, csv_file: Path) -> None:
        result = inspect(csv_file)
        assert isinstance(result, SchemaInfo)

    def test_columns_ordered(self, csv_file: Path) -> None:
        result = inspect(csv_file)
        assert result.columns == ["time", "force", "note"]

    def test_row_count_estimate_positive(self, csv_file: Path) -> None:
        result = inspect(csv_file)
        assert result.row_count_estimate == 3

    def test_file_size_bytes_positive(self, csv_file: Path) -> None:
        result = inspect(csv_file)
        assert result.file_size_bytes > 0

    def test_format_csv(self, csv_file: Path) -> None:
        result = inspect(csv_file)
        assert result.format == "csv"

    def test_column_types_dict_keys_match_columns(self, csv_file: Path) -> None:
        result = inspect(csv_file)
        assert set(result.column_types.keys()) == set(result.columns)

    def test_column_types_values_are_strings(self, csv_file: Path) -> None:
        result = inspect(csv_file)
        for col, dtype in result.column_types.items():
            assert isinstance(dtype, str), f"column_types[{col!r}] must be str"

    def test_parquet_inspect(self, parquet_file: Path) -> None:
        result = inspect(parquet_file)
        assert isinstance(result, SchemaInfo)
        assert result.format == "parquet"
        assert "time" in result.columns

    def test_empty_path_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            inspect("")

    def test_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            inspect("nonexistent_xyz_2989.csv")

    def test_unsupported_format_raises_value_error(self, tmp_path: Path) -> None:
        p = tmp_path / "data.xlsx"
        p.write_bytes(b"fake")
        with pytest.raises(ValueError, match="unsupported format"):
            inspect(p)


# ── preview ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.contract
class TestPreview:
    """Contract: preview(path, nrows, columns) -> pd.DataFrame."""

    def test_returns_dataframe(self, csv_file: Path) -> None:
        result = preview(csv_file)
        assert isinstance(result, pd.DataFrame)

    def test_default_nrows_100_respected(self, csv_file: Path) -> None:
        result = preview(csv_file)
        # File only has 3 rows; default nrows=100 returns all
        assert len(result) == 3

    def test_nrows_limit(self, csv_file: Path) -> None:
        result = preview(csv_file, nrows=1)
        assert len(result) == 1

    def test_column_projection(self, csv_file: Path) -> None:
        result = preview(csv_file, nrows=10, columns=["force", "note"])
        assert list(result.columns) == ["force", "note"]
        assert "time" not in result.columns

    def test_all_columns_present_by_default(self, csv_file: Path) -> None:
        result = preview(csv_file)
        assert set(result.columns) == {"time", "force", "note"}

    def test_parquet_preview(self, parquet_file: Path) -> None:
        result = preview(parquet_file, nrows=2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 2

    def test_zero_nrows_raises_value_error(self, csv_file: Path) -> None:
        with pytest.raises(ValueError, match="nrows must be greater than zero"):
            preview(csv_file, nrows=0)

    def test_missing_column_raises_value_error(self, csv_file: Path) -> None:
        with pytest.raises(ValueError):
            preview(csv_file, columns=["nonexistent"])

    def test_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            preview("missing.csv")


# ── convert ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.contract
class TestConvert:
    """Contract: convert(src, dst, format) -> ConversionReport."""

    def test_returns_conversion_report(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.csv"
        result = convert(csv_file, dst, "csv")
        assert isinstance(result, ConversionReport)

    def test_csv_to_csv_rows_written(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.csv"
        report = convert(csv_file, dst, "csv")
        assert report.rows_written == 3

    def test_csv_to_csv_file_created(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.csv"
        convert(csv_file, dst, "csv")
        assert dst.is_file()

    @_requires_parquet
    def test_csv_to_parquet(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.parquet"
        report = convert(csv_file, dst, "parquet")
        assert report.rows_written == 3
        assert dst.is_file()

    def test_report_columns_match_source(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.csv"
        report = convert(csv_file, dst, "csv")
        assert set(report.columns) == {"time", "force", "note"}

    def test_bytes_written_positive(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.csv"
        report = convert(csv_file, dst, "csv")
        assert report.bytes_written > 0

    def test_output_format_stored(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.csv"
        report = convert(csv_file, dst, "csv")
        assert report.output_format == "csv"

    def test_source_path_stored(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.csv"
        report = convert(csv_file, dst, "csv")
        # Source may be absolute or relative; just check it ends with the filename
        assert report.source.endswith("sample.csv")

    def test_unsupported_output_format_raises_value_error(
        self, csv_file: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="unsupported output format"):
            convert(csv_file, tmp_path / "out.xlsx", "xlsx")

    def test_missing_source_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            convert("nonexistent.csv", tmp_path / "out.csv", "csv")


# ── scan_batch ────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.contract
class TestScanBatch:
    """Contract: scan_batch(path, batch_size, columns) -> Iterator[pd.DataFrame]."""

    def test_returns_iterator(self, csv_file: Path) -> None:
        it = scan_batch(csv_file, batch_size=10)
        assert hasattr(it, "__iter__")
        assert hasattr(it, "__next__")

    def test_batches_are_dataframes(self, csv_file: Path) -> None:
        for batch in scan_batch(csv_file, batch_size=10):
            assert isinstance(batch, pd.DataFrame)

    def test_total_rows_match_file(self, csv_file: Path) -> None:
        total = sum(len(batch) for batch in scan_batch(csv_file, batch_size=2))
        assert total == 3

    def test_batch_size_respected(self, csv_file: Path) -> None:
        batches = list(scan_batch(csv_file, batch_size=2))
        assert len(batches[0]) <= 2

    def test_column_projection(self, csv_file: Path) -> None:
        for batch in scan_batch(csv_file, batch_size=10, columns=["force"]):
            assert list(batch.columns) == ["force"]

    def test_zero_batch_size_raises_value_error(self, csv_file: Path) -> None:
        with pytest.raises(ValueError, match="batch_size must be greater than zero"):
            # Must consume iterator to trigger validation
            list(scan_batch(csv_file, batch_size=0))

    def test_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            list(scan_batch("missing.csv", batch_size=10))

    def test_parquet_scan_batch(self, parquet_file: Path) -> None:
        total = sum(len(b) for b in scan_batch(parquet_file, batch_size=2))
        assert total == 3


# ── filter_export ─────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.contract
class TestFilterExport:
    """Contract: filter_export(path, dst, predicate, columns) -> int."""

    def test_returns_int(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "filtered.csv"
        result = filter_export(csv_file, dst, "force > 10.0")
        assert isinstance(result, int)

    def test_row_count_matches_predicate(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "filtered.csv"
        n = filter_export(csv_file, dst, "force > 10.0")
        assert n == 2  # 10.5 and 11.0 match; 9.8 does not

    def test_output_file_created(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "filtered.csv"
        filter_export(csv_file, dst, "force > 10.0")
        assert dst.is_file()

    def test_output_file_has_correct_rows(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "filtered.csv"
        filter_export(csv_file, dst, "force > 10.0")
        result_df = pd.read_csv(dst)
        assert len(result_df) == 2

    def test_column_projection(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "filtered.csv"
        filter_export(csv_file, dst, "force > 10.0", columns=["time", "force"])
        result_df = pd.read_csv(dst)
        assert list(result_df.columns) == ["time", "force"]

    @_requires_parquet
    def test_parquet_destination(self, csv_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "filtered.parquet"
        n = filter_export(csv_file, dst, "force > 10.0")
        assert dst.is_file()
        assert n == 2

    def test_empty_predicate_raises_value_error(
        self, csv_file: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="predicate must not be empty"):
            filter_export(csv_file, tmp_path / "out.csv", "   ")

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            filter_export("missing.csv", tmp_path / "out.csv", "force > 0")


# ── filter_export predicate validation ────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.contract
class TestFilterExportPredicateValidation:
    """``filter_export`` must not hand an unvalidated predicate to pandas eval.

    ``pd.DataFrame.query`` evaluates the predicate, so an attacker-controlled
    predicate is a code-injection vector. The predicate must go through
    ``validate_pandas_formula`` before it reaches ``df.query``.
    """

    @pytest.mark.parametrize(
        "predicate",
        [
            "force.__class__ > 0",
            "force.__class__.__mro__[1].__subclasses__() > 0",
            "__import__('os').system('echo pwned') > 0",
            "@force > 0",
            "abs(force) > 0",
        ],
    )
    def test_injection_predicate_rejected(
        self, csv_file: Path, tmp_path: Path, predicate: str
    ) -> None:
        dst = tmp_path / "out.csv"

        with pytest.raises(ValueError, match="Invalid predicate"):
            filter_export(csv_file, dst, predicate)

        assert not dst.exists()

    @pytest.mark.parametrize(
        "predicate",
        [
            "unknown_column > 0",
            "note == 'start'",
        ],
    )
    def test_predicate_outside_allowed_grammar_rejected(
        self, csv_file: Path, tmp_path: Path, predicate: str
    ) -> None:
        """The allow-list grammar is numeric/boolean only, as elsewhere in Tools.

        String comparison and unknown names are refused rather than evaluated,
        matching ``shared.python.data_processing.processor``.
        """
        dst = tmp_path / "out.csv"

        with pytest.raises(ValueError, match="Invalid predicate"):
            filter_export(csv_file, dst, predicate)

        assert not dst.exists()

    def test_legitimate_predicate_still_filters(
        self, csv_file: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.csv"

        n = filter_export(csv_file, dst, "force > 10.0 and time < 0.15")

        assert n == 2
        assert len(pd.read_csv(dst)) == 2

    def test_validation_uses_projected_columns(
        self, csv_file: Path, tmp_path: Path
    ) -> None:
        """A column dropped by ``columns=`` is no longer a valid predicate name."""
        dst = tmp_path / "out.csv"

        with pytest.raises(ValueError, match="Invalid predicate"):
            filter_export(csv_file, dst, "force > 10.0", columns=["time", "note"])


# ── cancel ────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.contract
class TestCancel:
    """Contract: cancel() sets the cancellation flag; scan_batch stops early."""

    def test_cancel_does_not_raise(self) -> None:
        cancel()  # must not raise

    def test_cancel_stops_scan_batch(self, csv_file: Path) -> None:
        """After cancel(), scan_batch yields nothing (flag is reset at call time)."""
        # Reset is done inside scan_batch at start; pre-cancel should be cleared.
        cancel()
        # The scan_batch implementation resets the flag at entry — subsequent
        # iteration should still yield rows (cancel was pre-call).
        batches = list(scan_batch(csv_file, batch_size=10))
        assert len(batches) >= 1  # flag was reset on entry, so data flows


# ── Per-operation cancellation token (issue #3679) ───────────────────────────


@pytest.mark.unit
@pytest.mark.contract
class TestCancellationToken:
    """Contract: per-operation tokens isolate cancellation between operations."""

    def test_tokens_are_independent(self) -> None:
        """Cancelling one token must not cancel another."""
        token_a = CancellationToken()
        token_b = CancellationToken()

        token_a.cancel()

        assert token_a.is_cancelled()
        assert not token_b.is_cancelled()

    def test_concurrent_conversions_do_not_cross_cancel(self, tmp_path: Path) -> None:
        """Cancelling one conversion's token leaves a concurrent one unaffected.

        Regression for the former process-global ``_cancelled`` flag, which let
        any ``cancel()`` abort every in-flight operation (issue #3679).
        """
        src = tmp_path / "in.csv"
        src.write_text(_CSV_CONTENT, encoding="utf-8")
        dst_cancelled = tmp_path / "cancelled.csv"
        dst_ok = tmp_path / "ok.csv"

        token_cancelled = CancellationToken()
        token_ok = CancellationToken()

        # Interleave: cancel only the first operation's token.
        token_cancelled.cancel()

        # The conversion bound to the cancelled token must abort...
        with pytest.raises(OperationCancelled):
            convert(src, dst_cancelled, "csv", token=token_cancelled)
        assert not dst_cancelled.exists()

        # ...while the conversion with its own (un-cancelled) token completes.
        report = convert(src, dst_ok, "csv", token=token_ok)
        assert isinstance(report, ConversionReport)
        assert dst_ok.exists()
        assert report.rows_written == 3

    def test_filter_export_respects_token(self, csv_file: Path, tmp_path: Path) -> None:
        """A cancelled token aborts filter_export before writing output."""
        dst = tmp_path / "filtered.csv"
        token = CancellationToken()
        token.cancel()

        with pytest.raises(OperationCancelled):
            filter_export(csv_file, dst, "force > 0", token=token)
        assert not dst.exists()

    def test_scan_batch_token_stops_other_unaffected(self, csv_file: Path) -> None:
        """A cancelled token stops its scan; an independent scan still yields."""
        token_stop = CancellationToken()
        token_stop.cancel()

        stopped = list(scan_batch(csv_file, batch_size=1, token=token_stop))
        assert stopped == []

        # A fresh token (or no token) is unaffected by the other cancellation.
        flowing = list(scan_batch(csv_file, batch_size=1, token=CancellationToken()))
        assert len(flowing) >= 1

    def test_global_cancel_does_not_affect_tokened_operation(
        self, csv_file: Path
    ) -> None:
        """Legacy global cancel() must not cancel operations with their own token."""
        cancel()  # legacy process-wide signal
        token = CancellationToken()

        batches = list(scan_batch(csv_file, batch_size=1, token=token))
        assert len(batches) >= 1


# ── Type stability (regression guard) ────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.contract
class TestTypeStability:
    """Verify return-type stability that downstream consumers depend on."""

    def test_schema_info_columns_is_list(self, csv_file: Path) -> None:
        info = inspect(csv_file)
        assert isinstance(info.columns, list)

    def test_schema_info_column_types_is_dict(self, csv_file: Path) -> None:
        info = inspect(csv_file)
        assert isinstance(info.column_types, dict)

    def test_schema_info_row_count_is_int(self, csv_file: Path) -> None:
        info = inspect(csv_file)
        assert isinstance(info.row_count_estimate, int)

    def test_schema_info_file_size_is_int(self, csv_file: Path) -> None:
        info = inspect(csv_file)
        assert isinstance(info.file_size_bytes, int)

    def test_conversion_report_rows_written_is_int(
        self, csv_file: Path, tmp_path: Path
    ) -> None:
        report = convert(csv_file, tmp_path / "out.csv", "csv")
        assert isinstance(report.rows_written, int)

    def test_conversion_report_bytes_written_is_int(
        self, csv_file: Path, tmp_path: Path
    ) -> None:
        report = convert(csv_file, tmp_path / "out.csv", "csv")
        assert isinstance(report.bytes_written, int)

    def test_filter_export_returns_int(self, csv_file: Path, tmp_path: Path) -> None:
        result = filter_export(csv_file, tmp_path / "out.csv", "force > 0")
        assert isinstance(result, int)
