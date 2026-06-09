from pathlib import Path

import pandas as pd
import pytest
from sidekick.data_processing.io import DataReader, DataWriter, FileFormatDetector


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["alpha", "beta"],
            "value": [1.5, 2.5],
        }
    )


def test_supported_extensions_map_only_to_implemented_handlers() -> None:
    """Advertised file formats must have read or write branches."""
    supported_formats = {
        FileFormatDetector.detect_format(Path(f"data{extension}"))
        for extension in FileFormatDetector.get_supported_extensions()
    }

    assert supported_formats == {
        "csv",
        "tsv",
        "excel",
        "parquet",
        "json",
        "numpy",
        "matlab",
        "sqlite",
    }


@pytest.mark.parametrize("extension", [".h5", ".hdf5", ".feather", ".pkl", ".pickle"])
def test_unsupported_or_disabled_formats_are_not_advertised(extension: str) -> None:
    assert FileFormatDetector.detect_format(Path(f"data{extension}")) is None


def test_pickle_override_remains_disabled(tmp_path: Path) -> None:
    path = tmp_path / "data.pkl"
    frame = _sample_frame()

    with pytest.raises(ValueError, match="disabled for security"):
        DataReader.read_file(path, format_type="pickle")

    with pytest.raises(ValueError, match="disabled for security"):
        DataWriter.write_file(frame, path, format_type="pickle")


@pytest.mark.parametrize(
    ("extension", "format_type"),
    [
        (".csv", "csv"),
        (".tsv", "tsv"),
        (".json", "json"),
    ],
)
def test_text_format_round_trips(
    tmp_path: Path, extension: str, format_type: str
) -> None:
    path = tmp_path / f"data{extension}"
    frame = _sample_frame()

    DataWriter.write_file(frame, path, format_type=format_type)
    loaded = DataReader.read_file(path, format_type=format_type)

    pd.testing.assert_frame_equal(loaded, frame)


def test_numpy_round_trip_uses_non_pickle_loader(tmp_path: Path) -> None:
    path = tmp_path / "data.npy"
    frame = pd.DataFrame([[1.5], [2.5]])

    DataWriter.write_file(frame, path)
    loaded = DataReader.read_file(path)

    pd.testing.assert_frame_equal(loaded, frame)


def test_sqlite_round_trip_uses_explicit_table_name_and_query(tmp_path: Path) -> None:
    path = tmp_path / "data.sqlite"
    frame = _sample_frame()

    DataWriter.write_file(frame, path, table_name="measurements")
    loaded = DataReader.read_file(path, query="SELECT name, value FROM measurements")

    pd.testing.assert_frame_equal(loaded, frame)


def test_reader_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported or undetected format"):
        DataReader.read_file(tmp_path / "data.unknown")


def test_writer_rejects_missing_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="df must be provided"):
        DataWriter.write_file(None, tmp_path / "data.csv")  # type: ignore[arg-type]
