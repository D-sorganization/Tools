"""Tests for folder_tool/folder_tool_constants.py.

Validates constant values, the validate_constants() function,
get_constants_info() introspection, and export_constants_documentation().
"""

from __future__ import annotations

from pathlib import Path

import pytest

import folder_tool.folder_tool_constants as ctk
from folder_tool.folder_tool_constants import (
    DEFAULT_CHUNK_SIZE,
    MAX_ARCHIVE_SIZE_RATIO,
    MAX_DIALOG_HEIGHT,
    MAX_DIALOG_WIDTH,
    MAX_FILE_SIZE_MB,
    MAX_RETRY_ATTEMPTS,
    MIN_DIALOG_HEIGHT,
    MIN_DIALOG_WIDTH,
    MIN_FILE_SIZE_BYTES,
    PROGRESS_BACKUP_PERCENT,
    PROGRESS_MAIN_OP_PERCENT,
    PROGRESS_ZIP_PERCENT,
    export_constants_documentation,
    get_constants_info,
    validate_constants,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constant value sanity checks
# ─────────────────────────────────────────────────────────────────────────────


class TestConstantValues:
    def test_max_file_size_positive(self) -> None:
        assert MAX_FILE_SIZE_MB > 0

    def test_min_file_size_non_negative(self) -> None:
        assert MIN_FILE_SIZE_BYTES >= 0

    def test_min_less_than_max_in_bytes(self) -> None:
        assert MIN_FILE_SIZE_BYTES < MAX_FILE_SIZE_MB * 1024 * 1024

    def test_chunk_size_positive(self) -> None:
        assert DEFAULT_CHUNK_SIZE > 0

    def test_retry_attempts_positive(self) -> None:
        assert MAX_RETRY_ATTEMPTS > 0

    def test_archive_ratio_between_zero_and_one(self) -> None:
        assert 0 < MAX_ARCHIVE_SIZE_RATIO < 1

    def test_dialog_sizes_consistent(self) -> None:
        assert MAX_DIALOG_WIDTH > MIN_DIALOG_WIDTH
        assert MAX_DIALOG_HEIGHT > MIN_DIALOG_HEIGHT

    def test_progress_percents_in_range(self) -> None:
        for val in [
            PROGRESS_BACKUP_PERCENT,
            PROGRESS_MAIN_OP_PERCENT,
            PROGRESS_ZIP_PERCENT,
        ]:
            assert 0 <= val <= 100

    def test_total_progress_leq_100(self) -> None:
        total = (
            PROGRESS_BACKUP_PERCENT + PROGRESS_MAIN_OP_PERCENT + PROGRESS_ZIP_PERCENT
        )
        assert total <= 100


# ─────────────────────────────────────────────────────────────────────────────
# validate_constants()
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateConstants:
    def test_passes_with_real_values(self) -> None:
        """validate_constants() should pass on the current constants without error."""
        validate_constants()  # must not raise

    def test_detects_negative_max_file_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ctk, "MAX_FILE_SIZE_MB", -1)
        with pytest.raises(ValueError, match="MAX_FILE_SIZE_MB must be positive"):
            validate_constants()

    def test_detects_negative_min_file_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ctk, "MIN_FILE_SIZE_BYTES", -1)
        with pytest.raises(
            ValueError, match="MIN_FILE_SIZE_BYTES must be non-negative"
        ):
            validate_constants()

    def test_detects_bad_archive_ratio_above_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ctk, "MAX_ARCHIVE_SIZE_RATIO", 1.5)
        with pytest.raises(ValueError, match="MAX_ARCHIVE_SIZE_RATIO must be between"):
            validate_constants()

    def test_detects_bad_archive_ratio_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ctk, "MAX_ARCHIVE_SIZE_RATIO", 0.0)
        with pytest.raises(ValueError, match="MAX_ARCHIVE_SIZE_RATIO must be between"):
            validate_constants()

    def test_detects_inverted_dialog_dimensions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ctk, "MAX_DIALOG_WIDTH", 100)
        monkeypatch.setattr(ctk, "MIN_DIALOG_WIDTH", 200)
        with pytest.raises(ValueError, match="MAX_DIALOG_WIDTH must be greater than"):
            validate_constants()

    def test_detects_bad_progress_over_100(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ctk, "PROGRESS_BACKUP_PERCENT", 150)
        with pytest.raises(ValueError, match="PROGRESS_BACKUP_PERCENT must be between"):
            validate_constants()

    def test_detects_total_progress_exceeds_100(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ctk, "PROGRESS_BACKUP_PERCENT", 50)
        monkeypatch.setattr(ctk, "PROGRESS_MAIN_OP_PERCENT", 50)
        monkeypatch.setattr(ctk, "PROGRESS_ZIP_PERCENT", 10)
        with pytest.raises(ValueError, match="Total progress allocation exceeds 100"):
            validate_constants()


# ─────────────────────────────────────────────────────────────────────────────
# get_constants_info()
# ─────────────────────────────────────────────────────────────────────────────


class TestGetConstantsInfo:
    def test_returns_dict(self) -> None:
        info = get_constants_info()
        assert isinstance(info, dict)

    def test_all_known_constants_present(self) -> None:
        info = get_constants_info()
        expected_keys = {
            "MAX_FILE_SIZE_MB",
            "MIN_FILE_SIZE_BYTES",
            "DEFAULT_CHUNK_SIZE",
            "MAX_RETRY_ATTEMPTS",
            "MAX_ARCHIVE_SIZE_RATIO",
        }
        assert expected_keys.issubset(info.keys())

    def test_each_entry_has_required_fields(self) -> None:
        info = get_constants_info()
        for name, entry in info.items():
            assert "value" in entry, f"{name} missing 'value'"
            assert "units" in entry, f"{name} missing 'units'"
            assert "source" in entry, f"{name} missing 'source'"

    def test_max_file_size_value_matches_constant(self) -> None:
        info = get_constants_info()
        assert info["MAX_FILE_SIZE_MB"]["value"] == str(MAX_FILE_SIZE_MB)

    def test_values_are_strings(self) -> None:
        info = get_constants_info()
        for name, entry in info.items():
            assert isinstance(entry["value"], str), f"{name}.value must be str"


# ─────────────────────────────────────────────────────────────────────────────
# export_constants_documentation()
# ─────────────────────────────────────────────────────────────────────────────


class TestExportConstantsDocumentation:
    def test_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "docs" / "constants.md"
        result = export_constants_documentation(str(out))
        assert result is True
        assert out.exists()

    def test_file_contains_constant_names(self, tmp_path: Path) -> None:
        out = tmp_path / "constants.md"
        export_constants_documentation(str(out))
        content = out.read_text(encoding="utf-8")
        assert "MAX_FILE_SIZE_MB" in content
        assert "DEFAULT_CHUNK_SIZE" in content

    def test_file_contains_markdown_headers(self, tmp_path: Path) -> None:
        out = tmp_path / "constants.md"
        export_constants_documentation(str(out))
        content = out.read_text(encoding="utf-8")
        assert "# Folder Tool Constants Documentation" in content
        assert "## Constants Overview" in content

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "a" / "b" / "c" / "constants.md"
        result = export_constants_documentation(str(out))
        assert result is True
        assert out.exists()

    def test_returns_true_on_success(self, tmp_path: Path) -> None:
        result = export_constants_documentation(str(tmp_path / "out.md"))
        assert result is True
