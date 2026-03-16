"""Unit tests for folder_tool/folder_tool_constants.py."""

from unittest.mock import patch

import pytest

import folder_tool.folder_tool_constants as ftc


class TestFolderToolConstants:
    def test_validate_constants_success(self):
        ftc.validate_constants()

    def test_validate_max_file_size_invalid(self):
        with patch.object(ftc, "MAX_FILE_SIZE_MB", -1):
            with pytest.raises(ValueError, match="MAX_FILE_SIZE_MB must be positive"):
                ftc.validate_constants()

    def test_validate_min_file_size_invalid(self):
        with patch.object(ftc, "MIN_FILE_SIZE_BYTES", -1):
            with pytest.raises(
                ValueError, match="MIN_FILE_SIZE_BYTES must be non-negative"
            ):
                ftc.validate_constants()

    def test_validate_min_file_size_greater_than_max(self):
        with patch.object(
            ftc, "MIN_FILE_SIZE_BYTES", ftc.MAX_FILE_SIZE_MB * 1024 * 1024 + 1
        ):
            with pytest.raises(
                ValueError,
                match="MIN_FILE_SIZE_BYTES must be less than MAX_FILE_SIZE_MB",
            ):
                ftc.validate_constants()

    def test_validate_max_status_length_invalid(self):
        with patch.object(ftc, "MAX_STATUS_LENGTH", 0):
            with pytest.raises(ValueError, match="MAX_STATUS_LENGTH must be positive"):
                ftc.validate_constants()

    def test_validate_max_ui_update_frequency_invalid(self):
        with patch.object(ftc, "MAX_UI_UPDATE_FREQUENCY", 0):
            with pytest.raises(
                ValueError, match="MAX_UI_UPDATE_FREQUENCY must be positive"
            ):
                ftc.validate_constants()

    def test_validate_dialog_width_invalid(self):
        with patch.object(ftc, "MAX_DIALOG_WIDTH", 100):
            with patch.object(ftc, "MIN_DIALOG_WIDTH", 200):
                with pytest.raises(
                    ValueError,
                    match="MAX_DIALOG_WIDTH must be greater than MIN_DIALOG_WIDTH",
                ):
                    ftc.validate_constants()

    def test_validate_dialog_height_invalid(self):
        with patch.object(ftc, "MAX_DIALOG_HEIGHT", 100):
            with patch.object(ftc, "MIN_DIALOG_HEIGHT", 200):
                with pytest.raises(
                    ValueError,
                    match="MAX_DIALOG_HEIGHT must be greater than MIN_DIALOG_HEIGHT",
                ):
                    ftc.validate_constants()

    def test_validate_archive_size_ratio_invalid_high(self):
        with patch.object(ftc, "MAX_ARCHIVE_SIZE_RATIO", 1.5):
            with pytest.raises(
                ValueError, match="MAX_ARCHIVE_SIZE_RATIO must be between 0 and 1"
            ):
                ftc.validate_constants()

    def test_validate_archive_size_ratio_invalid_low(self):
        with patch.object(ftc, "MAX_ARCHIVE_SIZE_RATIO", -0.5):
            with pytest.raises(
                ValueError, match="MAX_ARCHIVE_SIZE_RATIO must be between 0 and 1"
            ):
                ftc.validate_constants()

    def test_validate_retry_attempts_invalid(self):
        with patch.object(ftc, "MAX_RETRY_ATTEMPTS", 0):
            with pytest.raises(ValueError, match="MAX_RETRY_ATTEMPTS must be positive"):
                ftc.validate_constants()

    def test_validate_max_text_content_size_invalid(self):
        with patch.object(ftc, "MAX_TEXT_CONTENT_SIZE", 0):
            with pytest.raises(
                ValueError, match="MAX_TEXT_CONTENT_SIZE must be positive"
            ):
                ftc.validate_constants()

    def test_validate_max_title_length_invalid(self):
        with patch.object(ftc, "MAX_TITLE_LENGTH", 0):
            with pytest.raises(ValueError, match="MAX_TITLE_LENGTH must be positive"):
                ftc.validate_constants()

    def test_validate_max_counter_attempts_invalid(self):
        with patch.object(ftc, "MAX_COUNTER_ATTEMPTS", 0):
            with pytest.raises(
                ValueError, match="MAX_COUNTER_ATTEMPTS must be positive"
            ):
                ftc.validate_constants()

    def test_validate_progress_negative(self):
        with patch.object(ftc, "PROGRESS_BACKUP_PERCENT", -10):
            with pytest.raises(ValueError, match="must be between 0 and 100"):
                ftc.validate_constants()

    def test_validate_progress_over_100(self):
        with patch.object(ftc, "PROGRESS_BACKUP_PERCENT", 110):
            with pytest.raises(ValueError, match="must be between 0 and 100"):
                ftc.validate_constants()

    def test_validate_total_progress_exceeds(self):
        with patch.object(ftc, "PROGRESS_BACKUP_PERCENT", 50):
            with patch.object(ftc, "PROGRESS_MAIN_OP_PERCENT", 50):
                with patch.object(ftc, "PROGRESS_ZIP_PERCENT", 50):
                    with pytest.raises(
                        ValueError, match="Total progress allocation exceeds 100%"
                    ):
                        ftc.validate_constants()

    def test_get_constants_info(self):
        res = ftc.get_constants_info()
        assert "MAX_FILE_SIZE_MB" in res
        assert "value" in res["MAX_FILE_SIZE_MB"]
        assert "units" in res["MAX_FILE_SIZE_MB"]
        assert "source" in res["MAX_FILE_SIZE_MB"]

    def test_export_constants_documentation_success(self, tmp_path):
        out_path = tmp_path / "docs.txt"
        assert ftc.export_constants_documentation(str(out_path))
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "MAX_FILE_SIZE_MB" in content

    def test_export_constants_documentation_error(self, tmp_path):
        out_path = tmp_path / "docs.txt"
        # Cause TypeError inside the file write logic
        with patch("pathlib.Path.write_text", side_effect=TypeError("err")):
            assert not ftc.export_constants_documentation(str(out_path))
