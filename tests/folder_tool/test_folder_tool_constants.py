"""Unit tests for folder_tool_constants.py."""

from pathlib import Path

import folder_tool_constants
import pytest
from folder_tool_constants import (
    export_constants_documentation,
    get_constants_info,
    validate_constants,
)


class TestFolderToolConstants:
    """Test suite for folder tool constants validation and export."""

    def test_validate_constants_passes_initially(self):
        """Test that default constants pass validation out of the box."""
        # Should not raise any exception
        validate_constants()

    def test_get_constants_info_returns_dict(self):
        """Test that info dictionary is populated correctly."""
        info = get_constants_info()
        assert isinstance(info, dict)
        assert "MAX_FILE_SIZE_MB" in info
        assert "value" in info["MAX_FILE_SIZE_MB"]
        assert "units" in info["MAX_FILE_SIZE_MB"]
        assert "source" in info["MAX_FILE_SIZE_MB"]

    def test_export_constants_documentation_success(self, tmp_path: Path):
        """Test successful export of documentation."""
        out_file = tmp_path / "docs.md"
        result = export_constants_documentation(str(out_file))
        assert result is True
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "Folder Tool Constants Documentation" in content
        assert "MAX_FILE_SIZE_MB" in content

    def test_validate_file_size_constraints(self, monkeypatch):
        """Test file size validation boundary conditions."""
        monkeypatch.setattr(folder_tool_constants, "MAX_FILE_SIZE_MB", -1)
        with pytest.raises(ValueError, match="MAX_FILE_SIZE_MB must be positive"):
            validate_constants()

        monkeypatch.setattr(folder_tool_constants, "MAX_FILE_SIZE_MB", 1024)
        monkeypatch.setattr(folder_tool_constants, "MIN_FILE_SIZE_BYTES", -1)
        with pytest.raises(
            ValueError, match="MIN_FILE_SIZE_BYTES must be non-negative"
        ):
            validate_constants()

        monkeypatch.setattr(
            folder_tool_constants, "MIN_FILE_SIZE_BYTES", 2000 * 1024 * 1024
        )
        with pytest.raises(ValueError, match="MIN_FILE_SIZE_BYTES must be less than"):
            validate_constants()

    def test_validate_ui_constraints(self, monkeypatch):
        """Test UI dimension validations."""
        monkeypatch.setattr(folder_tool_constants, "MAX_DIALOG_WIDTH", 100)
        monkeypatch.setattr(folder_tool_constants, "MIN_DIALOG_WIDTH", 200)
        with pytest.raises(
            ValueError, match="MAX_DIALOG_WIDTH must be greater than MIN_DIALOG_WIDTH"
        ):
            validate_constants()

        monkeypatch.setattr(folder_tool_constants, "MAX_DIALOG_WIDTH", 800)
        monkeypatch.setattr(folder_tool_constants, "MIN_DIALOG_WIDTH", 400)
        monkeypatch.setattr(folder_tool_constants, "MAX_DIALOG_HEIGHT", 100)
        monkeypatch.setattr(folder_tool_constants, "MIN_DIALOG_HEIGHT", 200)
        with pytest.raises(
            ValueError, match="MAX_DIALOG_HEIGHT must be greater than MIN_DIALOG_HEIGHT"
        ):
            validate_constants()

    def test_validate_ui_constraints_lengths(self, monkeypatch):
        """Test remaining UI constant validations."""
        monkeypatch.setattr(folder_tool_constants, "MAX_STATUS_LENGTH", 0)
        with pytest.raises(ValueError, match="MAX_STATUS_LENGTH must be positive"):
            validate_constants()

        monkeypatch.setattr(folder_tool_constants, "MAX_STATUS_LENGTH", 10)
        monkeypatch.setattr(folder_tool_constants, "MAX_UI_UPDATE_FREQUENCY", 0)
        with pytest.raises(
            ValueError, match="MAX_UI_UPDATE_FREQUENCY must be positive"
        ):
            validate_constants()

        monkeypatch.setattr(folder_tool_constants, "MAX_UI_UPDATE_FREQUENCY", 10)
        monkeypatch.setattr(folder_tool_constants, "MAX_ARCHIVE_SIZE_RATIO", 2.0)
        with pytest.raises(
            ValueError, match="MAX_ARCHIVE_SIZE_RATIO must be between 0 and 1"
        ):
            validate_constants()

        monkeypatch.setattr(folder_tool_constants, "MAX_ARCHIVE_SIZE_RATIO", 0.5)
        monkeypatch.setattr(folder_tool_constants, "MAX_RETRY_ATTEMPTS", 0)
        with pytest.raises(ValueError, match="MAX_RETRY_ATTEMPTS must be positive"):
            validate_constants()

    def test_validate_new_constants(self, monkeypatch):
        """Test text content, title and counter constraints."""
        monkeypatch.setattr(folder_tool_constants, "MAX_TEXT_CONTENT_SIZE", 0)
        with pytest.raises(ValueError, match="MAX_TEXT_CONTENT_SIZE must be positive"):
            validate_constants()

        monkeypatch.setattr(folder_tool_constants, "MAX_TEXT_CONTENT_SIZE", 100)
        monkeypatch.setattr(folder_tool_constants, "MAX_TITLE_LENGTH", 0)
        with pytest.raises(ValueError, match="MAX_TITLE_LENGTH must be positive"):
            validate_constants()

        monkeypatch.setattr(folder_tool_constants, "MAX_TITLE_LENGTH", 100)
        monkeypatch.setattr(folder_tool_constants, "MAX_COUNTER_ATTEMPTS", 0)
        with pytest.raises(ValueError, match="MAX_COUNTER_ATTEMPTS must be positive"):
            validate_constants()

    def test_export_constants_documentation_failure(self, tmp_path: Path):
        """Test export failure handled explicitly."""
        # Let's mock `get_constants_info` to raise ValueError.
        import folder_tool_constants

        def mock_get(*args):
            raise ValueError("Mock Error")

        folder_tool_constants.get_constants_info = mock_get
        assert folder_tool_constants.export_constants_documentation("dummy") is False

    def test_validate_progress_constraints(self, monkeypatch):
        """Test progress percentage flow consistency."""
        monkeypatch.setattr(folder_tool_constants, "PROGRESS_BACKUP_PERCENT", 101)
        with pytest.raises(ValueError, match="must be between 0 and 100"):
            validate_constants()

        monkeypatch.setattr(folder_tool_constants, "PROGRESS_BACKUP_PERCENT", 50)
        monkeypatch.setattr(folder_tool_constants, "PROGRESS_MAIN_OP_PERCENT", 50)
        monkeypatch.setattr(folder_tool_constants, "PROGRESS_ZIP_PERCENT", 50)
        with pytest.raises(ValueError, match="Total progress allocation exceeds 100%"):
            validate_constants()
