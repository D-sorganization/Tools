"""Contract tests for the Rust-backed bulk data engine.

These tests define the first native data-plane slice for the Data Processor:
streaming CSV inspect, preview, and conversion. The Python UI should depend on
this small contract, not on Rust command-line details.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from data_processor.rust_engine import DataProcessorRustError, RustBulkDataEngine


def _write_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "time,force,note",
                "0.0,10.5,start",
                "0.1,11.0,mid",
                "0.2,9.75,end",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def engine() -> RustBulkDataEngine:
    return RustBulkDataEngine.from_repo_root()


def test_rust_engine_inspects_csv_without_pandas(
    engine: RustBulkDataEngine, tmp_path: Path
) -> None:
    csv_path = tmp_path / "sample.csv"
    _write_csv(csv_path)

    metadata = engine.inspect(csv_path)

    assert metadata.format == "csv"
    assert metadata.row_count == 3
    assert metadata.columns == ["time", "force", "note"]
    assert metadata.byte_size > 0


def test_rust_engine_previews_selected_columns(
    engine: RustBulkDataEngine, tmp_path: Path
) -> None:
    csv_path = tmp_path / "sample.csv"
    _write_csv(csv_path)

    preview = engine.preview(csv_path, rows=2, columns=["force", "note"])

    assert preview.columns == ["force", "note"]
    assert preview.rows == [
        {"force": "10.5", "note": "start"},
        {"force": "11.0", "note": "mid"},
    ]
    assert preview.rows_returned == 2


def test_rust_engine_converts_selected_columns(
    engine: RustBulkDataEngine, tmp_path: Path
) -> None:
    csv_path = tmp_path / "sample.csv"
    output_path = tmp_path / "selected.csv"
    _write_csv(csv_path)

    report = engine.convert(
        csv_path,
        output_path,
        output_format="csv",
        columns=["time", "force"],
    )

    assert report.rows_read == 3
    assert report.rows_written == 3
    assert report.columns == ["time", "force"]
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "time,force",
        "0.0,10.5",
        "0.1,11.0",
        "0.2,9.75",
    ]


def test_rust_engine_rejects_unsupported_formats(
    engine: RustBulkDataEngine, tmp_path: Path
) -> None:
    xlsx_path = tmp_path / "sample.xlsx"
    xlsx_path.write_bytes(b"not a workbook")

    with pytest.raises(DataProcessorRustError, match="Unsupported format"):
        engine.inspect(xlsx_path)
