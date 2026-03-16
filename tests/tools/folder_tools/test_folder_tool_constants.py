"""Tests for the folder_tool_constants module."""

from __future__ import annotations

import tempfile
from pathlib import Path

from folder_tool.folder_tool_constants import (
    DEFAULT_CHUNK_SIZE,
    MAX_ARCHIVE_SIZE_RATIO,
    MAX_COUNTER_ATTEMPTS,
    MAX_DIALOG_HEIGHT,
    MAX_DIALOG_WIDTH,
    MAX_FALLBACK_CONTENT_SIZE,
    MAX_FILE_SIZE_MB,
    MAX_LOG_ENTRIES,
    MAX_RETRY_ATTEMPTS,
    MAX_STATUS_LENGTH,
    MAX_TEXT_CONTENT_SIZE,
    MAX_TITLE_LENGTH,
    MAX_UI_UPDATE_FREQUENCY,
    MIN_DIALOG_HEIGHT,
    MIN_DIALOG_WIDTH,
    MIN_FILE_SIZE_BYTES,
    PROGRESS_BACKUP_PERCENT,
    PROGRESS_INCREMENT,
    PROGRESS_MAIN_OP_PERCENT,
    PROGRESS_START_MAIN,
    PROGRESS_START_ZIP,
    PROGRESS_ZIP_PERCENT,
    export_constants_documentation,
    get_constants_info,
    validate_constants,
)


class TestConstants:
    """Test that constants have sensible values."""

    def test_file_size_constants_positive(self) -> None:
        assert MAX_FILE_SIZE_MB > 0
        assert MIN_FILE_SIZE_BYTES >= 0
        assert DEFAULT_CHUNK_SIZE > 0

    def test_ui_constants_positive(self) -> None:
        assert MAX_LOG_ENTRIES > 0
        assert PROGRESS_INCREMENT > 0
        assert MAX_STATUS_LENGTH > 0
        assert MAX_UI_UPDATE_FREQUENCY > 0

    def test_dialog_size_constraints(self) -> None:
        assert MAX_DIALOG_WIDTH > MIN_DIALOG_WIDTH
        assert MAX_DIALOG_HEIGHT > MIN_DIALOG_HEIGHT

    def test_archive_ratio_range(self) -> None:
        assert 0 < MAX_ARCHIVE_SIZE_RATIO < 1

    def test_retry_constant(self) -> None:
        assert MAX_RETRY_ATTEMPTS > 0

    def test_text_constants(self) -> None:
        assert MAX_TEXT_CONTENT_SIZE > 0
        assert MAX_TITLE_LENGTH > 0
        assert MAX_COUNTER_ATTEMPTS > 0
        assert MAX_FALLBACK_CONTENT_SIZE > 0

    def test_progress_constants_in_range(self) -> None:
        for val in [
            PROGRESS_BACKUP_PERCENT,
            PROGRESS_MAIN_OP_PERCENT,
            PROGRESS_ZIP_PERCENT,
            PROGRESS_START_MAIN,
            PROGRESS_START_ZIP,
        ]:
            assert 0 <= val <= 100

    def test_progress_total_not_exceeding_100(self) -> None:
        total = (
            PROGRESS_BACKUP_PERCENT + PROGRESS_MAIN_OP_PERCENT + PROGRESS_ZIP_PERCENT
        )
        assert total <= 100


class TestValidateConstants:
    """Test the validate_constants function."""

    def test_validate_constants_succeeds(self) -> None:
        """Should not raise any exception with default constants."""
        validate_constants()


class TestGetConstantsInfo:
    """Test the get_constants_info function."""

    def test_returns_dict(self) -> None:
        info = get_constants_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_all_entries_have_required_keys(self) -> None:
        info = get_constants_info()
        for name, meta in info.items():
            assert "value" in meta, f"{name} missing 'value'"
            assert "units" in meta, f"{name} missing 'units'"
            assert "source" in meta, f"{name} missing 'source'"

    def test_known_constant_present(self) -> None:
        info = get_constants_info()
        assert "MAX_FILE_SIZE_MB" in info
        assert info["MAX_FILE_SIZE_MB"]["units"] == "MB"


class TestExportConstantsDocumentation:
    """Test the export_constants_documentation function."""

    def test_export_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "constants_doc.md")
            result = export_constants_documentation(out_path)
            assert result is True
            assert Path(out_path).exists()

    def test_export_content_has_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "constants_doc.md")
            export_constants_documentation(out_path)
            content = Path(out_path).read_text(encoding="utf-8")
            assert "# Folder Tool Constants Documentation" in content
            assert "MAX_FILE_SIZE_MB" in content

    def test_export_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "sub" / "dir" / "doc.md")
            result = export_constants_documentation(out_path)
            assert result is True
            assert Path(out_path).exists()
