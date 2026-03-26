"""Unit tests for folder_tool/archive_ops.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from folder_tool.archive_ops import ArchiveOperationsMixin
from folder_tool.Folders_Tool_r0 import MAX_FILE_SIZE_MB


class DummyVar:
    def __init__(self, value=False):
        self._val = value

    def get(self):
        return self._val


class DummyApp(ArchiveOperationsMixin):
    def __init__(self):
        self.safe_extract_var = DummyVar(False)
        self.cancel_operation = False
        self.source_folders = []

    def _get_unique_path(self, path: str) -> str:
        return path + "_unique"

    def update_progress(self, value, status):
        pass


@pytest.fixture
def app():
    return DummyApp()


class TestArchiveOperationsMixin:
    def test_safe_extract_archive_success(self, app, tmp_path):
        archive_path = tmp_path / "test.zip"
        archive_path.write_text("dummy archive")

        with patch.object(app, "_validate_archive_input", return_value=(archive_path, 100)):
            with patch.object(app, "_prepare_extraction_directory"):
                with patch("folder_tool.archive_ops.shutil.unpack_archive"):
                    with patch.object(app, "_cleanup_original_archive"):
                        success, msg = app.safe_extract_archive(str(archive_path))
                        assert success
                        assert "Successfully extracted" in msg

    def test_safe_extract_archive_safe_mode(self, app, tmp_path):
        app.safe_extract_var._val = True
        archive_path = tmp_path / "test.zip"
        archive_path.write_text("dummy archive")

        with patch.object(app, "_validate_archive_input", return_value=(archive_path, 100)):
            with patch.object(app, "_prepare_extraction_directory"):
                with patch("folder_tool.archive_ops.shutil.unpack_archive"):
                    with patch.object(app, "_validate_extraction_result"):
                        with patch.object(app, "_cleanup_original_archive"):
                            success, msg = app.safe_extract_archive(str(archive_path))
                            assert success

    def test_safe_extract_archive_error(self, app, tmp_path):
        archive_path = tmp_path / "test.zip"
        archive_path.write_text("a")

        with patch.object(app, "_validate_archive_input", return_value=(archive_path, 100)):
            with patch.object(
                app, "_prepare_extraction_directory", side_effect=OSError("dir error")
            ):
                with patch.object(app, "_cleanup_failed_extraction") as mock_cleanup:
                    success, msg = app.safe_extract_archive(str(archive_path))
                    assert not success
                    assert "Failed to extract" in msg
                    mock_cleanup.assert_called_once()

    def test_validate_archive_input_empty(self, app):
        with pytest.raises(ValueError, match="Archive path must be non-empty string"):
            app._validate_archive_input(None)

        with pytest.raises(ValueError, match="Archive path must be non-empty string"):
            app._validate_archive_input("")

    def test_validate_archive_input_not_exists(self, app, tmp_path):
        with pytest.raises(FileNotFoundError, match="Archive file not found"):
            app._validate_archive_input(str(tmp_path / "nonexistent.zip"))

    def test_validate_archive_input_not_file(self, app, tmp_path):
        with pytest.raises(ValueError, match="Archive path is not a file"):
            app._validate_archive_input(str(tmp_path))

    def test_validate_archive_input_unreadable(self, app, tmp_path):
        archive = tmp_path / "test.zip"
        archive.write_text("a")
        with patch("os.access", return_value=False):
            with pytest.raises(PermissionError, match="Cannot read archive file"):
                app._validate_archive_input(str(archive))

    def test_validate_archive_input_large_and_unsupported(self, app, tmp_path):
        archive = tmp_path / "test.txt"
        archive.write_text("a")

        original_stat = Path.stat

        def mock_stat(self_path, **kwargs):
            if self_path.name == "test.txt":
                m = MagicMock()
                m.st_size = int((MAX_FILE_SIZE_MB * 1024 * 1024) + 100)
                m.st_mode = 33188
                return m
            return original_stat(self_path, **kwargs)

        with patch.object(Path, "stat", side_effect=mock_stat, autospec=True):
            obj, size = app._validate_archive_input(str(archive))
            assert size > (MAX_FILE_SIZE_MB * 1024 * 1024)

    def test_prepare_extraction_directory_success(self, app, tmp_path):
        extract_dir = tmp_path / "extract"
        app._prepare_extraction_directory(str(extract_dir), extract_dir)
        assert extract_dir.exists()

    def test_prepare_extraction_directory_creation_fails(self, app, tmp_path):
        extract_dir = tmp_path / "extract"
        with patch.object(
            Path, "exists", side_effect=[False, False]
        ):  # first for mkdir exist_ok, 2nd for check
            with pytest.raises(Exception, match="Failed to create extraction directory"):
                app._prepare_extraction_directory(str(extract_dir), extract_dir)

    def test_prepare_extraction_directory_unwritable(self, app, tmp_path):
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        with patch("os.access", return_value=False):
            with pytest.raises(PermissionError, match="Cannot write to extraction directory"):
                app._prepare_extraction_directory(str(extract_dir), extract_dir)

    def test_validate_extraction_result_no_dir(self, app, tmp_path):
        extract_dir = tmp_path / "extract"
        with pytest.raises(Exception, match="destination folder was not created"):
            app._validate_extraction_result(str(extract_dir), extract_dir, 100)

    def test_validate_extraction_result_empty_dir(self, app, tmp_path):
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        with pytest.raises(Exception, match="destination folder is empty"):
            app._validate_extraction_result(str(extract_dir), extract_dir, 100)

    def test_validate_extraction_result_read_error(self, app, tmp_path):
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        (extract_dir / "file.txt").write_text("hello")

        with patch("os.path.getsize", side_effect=OSError("getsize error")):
            with pytest.raises(Exception, match="no files found in extracted folder"):
                app._validate_extraction_result(str(extract_dir), extract_dir, 100)

    def test_validate_extraction_result_small_size(self, app, tmp_path):
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        (extract_dir / "file.txt").write_text("a")

        # 1 byte is < size * ratio if size is large
        app._validate_extraction_result(str(extract_dir), extract_dir, 1000)

    def test_validate_extraction_result_success(self, app, tmp_path):
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        (extract_dir / "file.txt").write_text("hello world")

        app._validate_extraction_result(str(extract_dir), extract_dir, 1)

    def test_cleanup_original_archive_success(self, app, tmp_path):
        archive = tmp_path / "test.zip"
        archive.write_text("a")
        app._cleanup_original_archive(archive)
        assert not archive.exists()

    def test_cleanup_original_archive_error(self, app, tmp_path):
        archive = tmp_path / "test.zip"
        archive.write_text("a")
        with patch.object(Path, "unlink", side_effect=OSError("delete error")):
            app._cleanup_original_archive(archive)

    def test_cleanup_failed_extraction_success(self, app, tmp_path):
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        app._cleanup_failed_extraction(extract_dir, str(extract_dir))
        assert not extract_dir.exists()

    def test_cleanup_failed_extraction_error(self, app, tmp_path):
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        with patch("shutil.rmtree", side_effect=OSError("rmtree error")):
            app._cleanup_failed_extraction(extract_dir, str(extract_dir))

    def test_bulk_unzip_enhanced_empty(self, app, tmp_path):
        app.source_folders = [str(tmp_path)]
        res = app._bulk_unzip_enhanced()
        assert res == ["No archives found to extract."]

    def test_bulk_unzip_enhanced_success_and_fail(self, app, tmp_path):
        app.source_folders = [str(tmp_path)]
        (tmp_path / "arch1.zip").write_text("a")
        (tmp_path / "arch2.rar").write_text("a")
        (tmp_path / "arch3.7z").write_text("a")

        def mock_extract(path):
            if "arch2" in path:
                return False, "failed"
            return True, "extracted"

        with patch.object(app, "safe_extract_archive", side_effect=mock_extract):
            res = app._bulk_unzip_enhanced()
            assert "Successfully extracted: 2, Failed: 1" in res[0]

    def test_bulk_unzip_enhanced_cancel(self, app, tmp_path):
        app.source_folders = [str(tmp_path)]
        (tmp_path / "arch1.zip").write_text("a")

        app.cancel_operation = True

        with patch.object(app, "safe_extract_archive") as mock_extract:
            _ = app._bulk_unzip_enhanced()
            mock_extract.assert_not_called()

    def test_bulk_unzip_enhanced_deleted_mid_stream(self, app, tmp_path):
        app.source_folders = [str(tmp_path)]
        arch1 = tmp_path / "arch1.zip"
        arch1.write_text("a")

        original_exists = Path.exists

        def mock_exists(self_path, *args, **kwargs):
            if self_path.name == "arch1.zip":
                return False
            return original_exists(self_path, *args, **kwargs)

        with patch.object(Path, "exists", side_effect=mock_exists, autospec=True):
            res = app._bulk_unzip_enhanced()
            assert "Processed 1 archive(s). Successfully extracted: 0, Failed: 0" in res[0]
