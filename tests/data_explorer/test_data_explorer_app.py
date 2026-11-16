import builtins
from pathlib import Path
from typing import Any

import pytest

from data_explorer import data_explorer_app
from data_explorer.data_explorer_app import load_dataset


def test_load_dataset_parses_quoted_commas_and_escaped_quotes(tmp_path: Path) -> None:
    path = tmp_path / "plant.csv"
    path.write_text('  name  ,"Plant, West","tag ""A"""  \n', encoding="utf-8")

    result = load_dataset(path)

    assert result["columns"] == ["name", "Plant, West", 'tag "A"']


def test_load_dataset_trims_whitespace_around_unquoted_headers(tmp_path: Path) -> None:
    path = tmp_path / "values.csv"
    path.write_text("  time , value  \n0,1\n", encoding="utf-8")

    result = load_dataset(path)

    assert result["columns"] == ["time", "value"]


def test_discover_datasets_finds_supported_files_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    expected = [
        tmp_path / "a.csv",
        nested / "b.c3d",
        nested / "c.h5",
        tmp_path / "d.hdf5",
        nested / "e.json",
    ]
    for path in expected:
        path.write_bytes(b"data")
    (tmp_path / "ignored.txt").write_text("ignored", encoding="utf-8")

    assert data_explorer_app.discover_datasets(tmp_path) == sorted(expected)


def test_discover_datasets_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Search directory not found"):
        data_explorer_app.discover_datasets(tmp_path / "missing")


def test_load_dataset_returns_json_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text('{"time": [0, 1], "temperature": [20, 21]}', encoding="utf-8")

    result = load_dataset(path)

    assert result["format"] == "json"
    assert result["columns"] == ["time", "temperature"]


def test_load_dataset_returns_no_columns_for_json_array(tmp_path: Path) -> None:
    path = tmp_path / "runs.json"
    path.write_text('[{"time": 0}]', encoding="utf-8")

    assert load_dataset(path)["columns"] == []


def test_load_dataset_rejects_missing_and_unsupported_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset(tmp_path / "missing.csv")

    path = tmp_path / "run.bin"
    path.write_bytes(b"data")
    with pytest.raises(ValueError, match="Unsupported format: .bin"):
        load_dataset(path)


def test_load_dataset_returns_empty_columns_when_csv_cannot_be_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "locked.csv"
    path.write_text("time,value\n0,1\n", encoding="utf-8")
    real_open = builtins.open

    def deny_target(file: Any, *args: Any, **kwargs: Any) -> Any:
        if file == path:
            raise PermissionError("denied")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny_target)

    assert load_dataset(path)["columns"] == []
