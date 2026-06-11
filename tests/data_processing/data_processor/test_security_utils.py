"""Tests for data_processor.security_utils."""

from __future__ import annotations

from pathlib import Path

import pytest
from data_processor.security_utils import (
    FileSizeError,
    PathValidationError,
    validate_file_path,
    validate_python_expression,
)


def test_validate_python_expression_empty_raises() -> None:
    with pytest.raises(ValueError, match="expr must not be empty"):
        validate_python_expression("")


def test_validate_file_path_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="file_path must not be empty"):
        validate_file_path("")


def test_validate_file_path_accepts_case_insensitive_extensions(
    tmp_path: Path,
) -> None:
    sample_file = tmp_path / "example.csv"
    sample_file.write_text("a,b\n1,2\n", encoding="utf-8")

    validated = validate_file_path(
        sample_file,
        allowed_extensions={".CSV"},
        allow_anywhere=True,
    )

    assert validated == sample_file.resolve()


def test_validate_file_path_non_extension_format_raises(tmp_path: Path) -> None:
    sample_file = tmp_path / "example.csv"
    sample_file.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="allowed_extensions must contain"):
        validate_file_path(
            sample_file,
            allowed_extensions={"csv"},
            allow_anywhere=True,
        )


def test_check_file_size_nonpositive_limit_raises(tmp_path: Path) -> None:
    sample_file = tmp_path / "small.txt"
    sample_file.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="max_size_bytes must be positive"):
        from data_processor.security_utils import check_file_size

        check_file_size(sample_file, max_size_bytes=0)


def test_validate_file_path_rejects_wrong_extension(tmp_path: Path) -> None:
    sample_file = tmp_path / "example.csv"
    sample_file.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(PathValidationError, match="Unsupported file extension"):
        validate_file_path(
            sample_file,
            allowed_extensions={".xlsx"},
            allow_anywhere=True,
        )


def test_check_file_size_exceeds_limit_raises(tmp_path: Path) -> None:
    sample_file = tmp_path / "small.txt"
    sample_file.write_text("0123456789", encoding="utf-8")

    from data_processor.security_utils import check_file_size

    with pytest.raises(FileSizeError, match="File too large"):
        check_file_size(sample_file, max_size_bytes=2)
