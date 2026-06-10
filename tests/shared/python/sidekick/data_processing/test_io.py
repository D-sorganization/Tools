from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sidekick.data_processing.formats import SUPPORTED_FORMATS
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

    assert supported_formats == SUPPORTED_FORMATS


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


@given(
    extension=st.sampled_from(FileFormatDetector.get_supported_extensions()),
    prefix=st.text(
        alphabet=st.characters(
            whitelist_categories=("Ll", "Lu", "Nd"),
            whitelist_characters=("_", "-"),
        ),
        min_size=1,
        max_size=24,
    ),
)
def test_format_detection_is_case_insensitive_for_supported_extensions(
    extension: str, prefix: str
) -> None:
    expected = FileFormatDetector.detect_format(Path(f"data{extension}"))

    actual = FileFormatDetector.detect_format(Path(f"{prefix}{extension.upper()}"))

    assert actual == expected


@given(
    extension=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
        min_size=1,
        max_size=8,
    ).filter(
        lambda value: (
            f".{value.lower()}" not in FileFormatDetector.get_supported_extensions()
        )
    )
)
def test_unknown_extensions_never_map_to_supported_formats(extension: str) -> None:
    assert FileFormatDetector.detect_format(Path(f"data.{extension}")) is None


def test_failed_sqlite_read_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing sqlite read must still close the connection (#3277).

    Pointing the reader at a valid sqlite file but querying a missing table
    raises; the connection opened for the call must be closed regardless, or a
    consumer importing files in a loop leaks descriptors / leaves the DB locked.
    """
    import sqlite3

    path = tmp_path / "data.sqlite"
    DataWriter.write_file(_sample_frame(), path, table_name="measurements")

    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def _tracking_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", _tracking_connect)

    with pytest.raises(Exception):  # noqa: B017,PT011 - sqlite raises OperationalError
        DataReader.read_file(path, query="SELECT * FROM no_such_table")

    assert opened, "expected a sqlite connection to be opened"
    for conn in opened:
        # Closed connections raise ProgrammingError when reused.
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


def test_failed_sqlite_write_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing sqlite write must still close the connection (#3277)."""
    import sqlite3

    path = tmp_path / "data_write.sqlite"

    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def _tracking_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", _tracking_connect)

    class _Unserializable:
        pass

    frame = pd.DataFrame({"bad": [_Unserializable()]})
    with pytest.raises(Exception):  # noqa: B017 - to_sql raises on bad dtype
        DataWriter.write_file(frame, path, format_type="sqlite")

    assert opened, "expected a sqlite connection to be opened"
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
