"""Tests for python.src.utils.validation module.

Covers:
- validate_path
- validate_file_extension
- validate_python_version
- validate_not_none
- validate_not_empty
- validate_in_range
"""

from __future__ import annotations

from pathlib import Path

from utils.validation import (
    validate_file_extension,
    validate_in_range,
    validate_not_empty,
    validate_not_none,
    validate_path,
    validate_python_version,
)


class TestValidatePath:
    """Tests for validate_path function."""

    def test_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        valid, msg = validate_path(f, must_exist=True, must_be_file=True)
        assert valid is True
        assert msg is None

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        valid, msg = validate_path(tmp_path / "missing", must_exist=True)
        assert valid is False
        assert "not exist" in msg

    def test_must_be_dir(self, tmp_path: Path) -> None:
        valid, msg = validate_path(tmp_path, must_be_dir=True)
        assert valid is True

    def test_file_not_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("data")
        valid, msg = validate_path(f, must_be_dir=True)
        assert valid is False

    def test_within_boundary(self, tmp_path: Path) -> None:
        f = tmp_path / "sub" / "file.txt"
        f.parent.mkdir(parents=True)
        f.write_text("x")
        valid, msg = validate_path(f, must_be_within=tmp_path)
        assert valid is True

    def test_outside_boundary(self, tmp_path: Path) -> None:
        valid, msg = validate_path(
            tmp_path.parent, must_exist=True, must_be_within=tmp_path
        )
        assert valid is False
        assert "Security" in msg


class TestValidateFileExtension:
    """Tests for validate_file_extension."""

    def test_allowed_extension(self) -> None:
        valid, msg = validate_file_extension("test.py", [".py", ".txt"])
        assert valid is True

    def test_disallowed_extension(self) -> None:
        valid, msg = validate_file_extension("test.exe", [".py", ".txt"])
        assert valid is False

    def test_case_insensitive(self) -> None:
        valid, msg = validate_file_extension("test.PY", ["py"])
        assert valid is True

    def test_dot_normalization(self) -> None:
        """Extensions without dots should be normalized."""
        valid, msg = validate_file_extension("file.csv", ["csv"])
        assert valid is True


class TestValidatePythonVersion:
    """Tests for validate_python_version."""

    def test_current_version_passes(self) -> None:
        valid, ver = validate_python_version(min_major=3, min_minor=10)
        assert valid is True
        assert "3." in ver

    def test_unreachable_version_fails(self) -> None:
        valid, ver = validate_python_version(min_major=99)
        assert valid is False


class TestValidateNotNone:
    """Tests for validate_not_none."""

    def test_with_value(self) -> None:
        valid, msg = validate_not_none("hello")
        assert valid is True
        assert msg is None

    def test_with_none(self) -> None:
        valid, msg = validate_not_none(None, name="field")
        assert valid is False
        assert "field" in msg


class TestValidateNotEmpty:
    """Tests for validate_not_empty."""

    def test_non_empty_string(self) -> None:
        valid, msg = validate_not_empty("hello")
        assert valid is True

    def test_empty_string(self) -> None:
        valid, msg = validate_not_empty("", name="name")
        assert valid is False

    def test_empty_list(self) -> None:
        valid, msg = validate_not_empty([])
        assert valid is False

    def test_non_empty_dict(self) -> None:
        valid, msg = validate_not_empty({"key": "val"})
        assert valid is True


class TestValidateInRange:
    """Tests for validate_in_range."""

    def test_within_range(self) -> None:
        valid, msg = validate_in_range(5, min_val=0, max_val=10)
        assert valid is True

    def test_below_range(self) -> None:
        valid, msg = validate_in_range(-1, min_val=0)
        assert valid is False
        assert "less than" in msg

    def test_above_range(self) -> None:
        valid, msg = validate_in_range(11, max_val=10)
        assert valid is False
        assert "greater than" in msg

    def test_no_bounds(self) -> None:
        valid, msg = validate_in_range(999)
        assert valid is True

    def test_boundary_inclusive(self) -> None:
        valid, msg = validate_in_range(10, min_val=10, max_val=10)
        assert valid is True
