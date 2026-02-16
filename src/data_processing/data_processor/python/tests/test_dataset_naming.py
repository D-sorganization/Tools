"""Tests for data_processor.core.dataset_naming module."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from data_processor.core.dataset_naming import (
    generate_dataset_name,
    generate_timestamped_name,
    generate_unique_name,
    parse_dataset_name,
    sanitize_dataset_name,
    validate_dataset_name,
)


class TestValidateDatasetName:
    """Tests for validate_dataset_name."""

    def test_valid_simple(self) -> None:
        assert validate_dataset_name("my_data") is True

    def test_valid_with_dots(self) -> None:
        assert validate_dataset_name("my.data") is True

    def test_valid_with_hyphens(self) -> None:
        assert validate_dataset_name("my-data-2024") is True

    def test_empty_string(self) -> None:
        assert validate_dataset_name("") is False

    def test_whitespace_only(self) -> None:
        assert validate_dataset_name("   ") is False

    def test_leading_whitespace(self) -> None:
        assert validate_dataset_name(" data") is False

    def test_trailing_whitespace(self) -> None:
        assert validate_dataset_name("data ") is False

    def test_path_separator_forward(self) -> None:
        assert validate_dataset_name("path/to/file") is False

    def test_path_separator_backslash(self) -> None:
        assert validate_dataset_name("path\\to\\file") is False

    def test_invalid_chars(self) -> None:
        for ch in ['<', '>', ':', '"', '|', '?', '*']:
            assert validate_dataset_name(f"data{ch}file") is False

    def test_valid_with_spaces(self) -> None:
        assert validate_dataset_name("my data") is True


class TestSanitizeDatasetName:
    """Tests for sanitize_dataset_name."""

    def test_already_valid(self) -> None:
        assert sanitize_dataset_name("clean_name") == "clean_name"

    def test_strips_whitespace(self) -> None:
        assert sanitize_dataset_name("  hello  ") == "hello"

    def test_replaces_invalid_chars(self) -> None:
        result = sanitize_dataset_name("my<file>name")
        assert "<" not in result
        assert ">" not in result

    def test_collapses_multiple_underscores(self) -> None:
        result = sanitize_dataset_name("a___b")
        assert "__" not in result

    def test_strips_leading_trailing_underscores(self) -> None:
        result = sanitize_dataset_name("___hello___")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_empty_becomes_default(self) -> None:
        assert sanitize_dataset_name("") == "data"

    def test_all_invalid_becomes_default(self) -> None:
        assert sanitize_dataset_name("<<<>>>") == "data"


class TestGenerateDatasetName:
    """Tests for generate_dataset_name."""

    def test_default_has_base(self) -> None:
        name = generate_dataset_name()
        assert name.startswith("data")

    def test_custom_base(self) -> None:
        name = generate_dataset_name(base_name="experiment")
        assert name.startswith("experiment")

    def test_no_timestamp(self) -> None:
        name = generate_dataset_name(
            include_timestamp=False, include_date=False
        )
        assert name == "data"

    def test_with_filter(self) -> None:
        name = generate_dataset_name(
            include_filter=True,
            filter_type="Butterworth",
            include_timestamp=False,
            include_date=False,
        )
        assert "butterworth" in name.lower()


class TestGenerateTimestampedName:
    """Tests for generate_timestamped_name."""

    def test_contains_timestamp(self) -> None:
        name = generate_timestamped_name("output")
        assert "output" in name
        # Should contain a date-like pattern
        assert re.search(r"\d{8}_\d{6}", name) is not None

    def test_with_extension(self) -> None:
        name = generate_timestamped_name("data", extension=".csv")
        assert name.endswith(".csv")

    def test_extension_without_dot(self) -> None:
        name = generate_timestamped_name("data", extension="csv")
        assert name.endswith(".csv")


class TestGenerateUniqueName:
    """Tests for generate_unique_name."""

    def test_unique_when_no_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_unique_name(tmpdir, "data", ".csv")
            assert result == "data.csv"

    def test_adds_suffix_on_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "data.csv").touch()
            result = generate_unique_name(tmpdir, "data", ".csv")
            assert result == "data_1.csv"

    def test_increments_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "data.csv").touch()
            (Path(tmpdir) / "data_1.csv").touch()
            result = generate_unique_name(tmpdir, "data", ".csv")
            assert result == "data_2.csv"

    def test_max_attempts_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create conflicts for all attempts
            (Path(tmpdir) / "x.csv").touch()
            for i in range(1, 4):
                (Path(tmpdir) / f"x_{i}.csv").touch()
            with pytest.raises(RuntimeError):
                generate_unique_name(tmpdir, "x", ".csv", max_attempts=3)


class TestParseDatasetName:
    """Tests for parse_dataset_name."""

    def test_simple_name(self) -> None:
        result = parse_dataset_name("data.csv")
        assert result["base_name"] == "data"
        assert result["extension"] == ".csv"

    def test_with_timestamp(self) -> None:
        result = parse_dataset_name("data_20240115_103000.csv")
        assert result["timestamp"] is not None
        assert "20240115_103000" in result["timestamp"]

    def test_with_date(self) -> None:
        result = parse_dataset_name("data_2024-01-15.csv")
        assert result["timestamp"] is not None
        assert "2024-01-15" in result["timestamp"]

    def test_with_filter_type(self) -> None:
        result = parse_dataset_name("data_butterworth.csv")
        assert result["filter_type"] == "butterworth"

    def test_no_extension(self) -> None:
        result = parse_dataset_name("data")
        assert result["extension"] == ""
