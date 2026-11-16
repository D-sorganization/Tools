"""File helper error paths preserve defaults and existing destination bytes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from utils import file_utils

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("operation", "extra"),
    [
        (file_utils.safe_read_json, ()),
        (file_utils.safe_write_json, ({},)),
        (file_utils.ensure_directory, ()),
        (file_utils.find_file_upwards, ()),
        (file_utils.safe_read_text, ()),
        (file_utils.safe_write_text, ("text",)),
    ],
)
def test_missing_path_or_filename_is_rejected(
    operation: Any, extra: tuple[Any, ...]
) -> None:
    missing: Any = None
    with pytest.raises(ValueError, match="must be provided"):
        operation(missing, *extra)


def test_json_directory_read_returns_supplied_default_and_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    default = {"fallback": True}
    assert file_utils.safe_read_json(tmp_path, default) is default
    assert "Error reading JSON file" in caplog.text


def test_text_directory_read_returns_supplied_default_and_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert file_utils.safe_read_text(tmp_path, default="fallback") == "fallback"
    assert "Error reading text file" in caplog.text


def test_json_invalid_utf8_returns_supplied_default(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_bytes(b'\xff{"value": 1}')
    default = {"fallback": True}
    assert file_utils.safe_read_json(path, default) is default


def test_text_invalid_utf8_returns_supplied_default(tmp_path: Path) -> None:
    path = tmp_path / "settings.txt"
    path.write_bytes(b"\xffinvalid utf8")
    assert file_utils.safe_read_text(path, default="fallback") == "fallback"


def test_directory_creation_failure_does_not_modify_parent_file(tmp_path: Path) -> None:
    parent = tmp_path / "parent-file"
    parent.write_text("keep", encoding="utf-8")
    assert file_utils.ensure_directory(parent / "child") is False
    assert parent.read_text(encoding="utf-8") == "keep"


def test_failed_atomic_json_replace_preserves_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "settings.json"
    original = b'{"version": 1}\n'
    destination.write_bytes(original)

    def fail_replace(source: Any, target: Any) -> None:
        raise OSError("fixture replace failure")

    monkeypatch.setattr(file_utils.os, "replace", fail_replace)
    assert file_utils.safe_write_json(destination, {"version": 2}) is False
    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_write_cleanup_error_does_not_mask_original_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "settings.txt"
    destination.write_text("keep", encoding="utf-8")
    original_failure = OSError("fixture replace failure")

    def fail_replace(source: Any, target: Any) -> None:
        raise original_failure

    def fail_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("fixture cleanup failure")

    monkeypatch.setattr(file_utils.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(OSError) as exc:
        file_utils.atomic_write_text(destination, "new")
    assert exc.value is original_failure
    assert destination.read_text(encoding="utf-8") == "keep"
