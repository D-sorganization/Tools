"""Contract tests for Sidekick data I/O format support (issue #3256).

These tests guarantee that the format surface advertised by
:class:`FileFormatDetector` matches what :class:`DataReader` and
:class:`DataWriter` actually implement. Previously ``.h5``/``.hdf5`` and
``.feather`` were advertised as supported but had no reader/writer branch, so a
caller could detect a format and then immediately get
``ValueError("Unsupported or undetected format")`` from the same module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sidekick.data_processing.io import (
    DataReader,
    DataWriter,
    FileFormatDetector,
)

pytestmark = pytest.mark.contract


# Extensions whose detected format has a working reader AND writer branch and is
# round-trippable with the dependencies declared by the package.
_ROUNDTRIP_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".parquet",
    ".pq",
    ".json",
    ".feather",
    ".npy",
}

# Extensions that detect to a format with reader+writer branches but are not
# exercised here for round-trip because they need optional native engines
# (Excel -> openpyxl, MATLAB -> scipy, sqlite -> a query/table name). They are
# still asserted to map to an *implemented* format token below.
_IMPLEMENTED_NON_ROUNDTRIP = {
    ".xlsx": "excel",
    ".xls": "excel",
    ".mat": "matlab",
    ".db": "sqlite",
    ".sqlite": "sqlite",
}


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


def test_no_extension_maps_to_unimplemented_format() -> None:
    """Every advertised extension must resolve to a format the I/O layer knows.

    Regression guard: ``.h5``/``.hdf5`` used to detect as ``hdf5`` with no
    reader or writer branch. Reading the detector's source of truth, assert no
    advertised extension resolves to a token outside the implemented set.
    """
    implemented_formats = {
        "csv",
        "tsv",
        "excel",
        "parquet",
        "feather",
        "json",
        "numpy",
        "matlab",
        "sqlite",
    }
    for ext in FileFormatDetector.get_supported_extensions():
        fmt = FileFormatDetector.detect_format(f"data{ext}")
        assert fmt in implemented_formats, (
            f"extension {ext!r} advertised as {fmt!r}, which has no "
            f"reader/writer implementation"
        )


def test_hdf5_extensions_are_no_longer_advertised() -> None:
    extensions = FileFormatDetector.get_supported_extensions()
    assert ".h5" not in extensions
    assert ".hdf5" not in extensions


def test_feather_is_advertised_and_detected() -> None:
    assert ".feather" in FileFormatDetector.get_supported_extensions()
    assert FileFormatDetector.detect_format("x.feather") == "feather"


@pytest.mark.parametrize("ext", sorted(_ROUNDTRIP_EXTENSIONS))
def test_advertised_extension_round_trips(ext: str, tmp_path: Path) -> None:
    """Every round-trippable advertised extension must write then read back."""
    if ext in {".parquet", ".pq", ".feather"}:
        pytest.importorskip("pyarrow")

    df = _sample_frame()
    target = tmp_path / f"data{ext}"

    DataWriter.write_file(df, target)
    assert target.exists()

    result = DataReader.read_file(target)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Column-preserving formats should retain the original columns.
    if ext not in {".npy"}:
        assert list(result.columns) == list(df.columns)


@pytest.mark.parametrize("ext", sorted(_IMPLEMENTED_NON_ROUNDTRIP))
def test_non_roundtrip_extension_maps_to_implemented_format(ext: str) -> None:
    fmt = FileFormatDetector.detect_format(f"data{ext}")
    assert fmt == _IMPLEMENTED_NON_ROUNDTRIP[ext]


def test_detect_format_requires_path() -> None:
    with pytest.raises(ValueError, match="file_path must be provided"):
        FileFormatDetector.detect_format(None)  # type: ignore[arg-type]


def test_unknown_extension_returns_none_and_raises_on_read(tmp_path: Path) -> None:
    unknown = tmp_path / "data.unknownext"
    unknown.write_text("nothing")
    assert FileFormatDetector.detect_format(unknown) is None
    with pytest.raises(ValueError, match="Unsupported or undetected format"):
        DataReader.read_file(unknown)


def test_pickle_format_is_disabled_for_read() -> None:
    with pytest.raises(ValueError, match="Pickle format is disabled"):
        DataReader.read_file("x.pkl", format_type="pickle")


def test_pickle_format_is_disabled_for_write() -> None:
    with pytest.raises(ValueError, match="Pickle format is disabled"):
        DataWriter.write_file(_sample_frame(), "x.pkl", format_type="pickle")


def test_write_requires_dataframe() -> None:
    with pytest.raises(ValueError, match="df must be provided"):
        DataWriter.write_file(None, "x.csv")  # type: ignore[arg-type]
