"""Unit tests for folder_tool_archive.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from folder_tool.folder_tool_archive import MAX_UI_UPDATE_FREQUENCY, ArchiveMixin


class DummyApp(ArchiveMixin):
    def __init__(self):
        self.dest_folder = None
        self.cancel_operation = False

    def _get_unique_path(self, path: str) -> str:
        return path + "_unique"

    def update_progress(self, value: float, status: str) -> None:
        pass


@pytest.fixture
def app():
    return DummyApp()


class TestArchiveMixin:
    def test_validate_dest_folder_empty(self, app):
        with pytest.raises(ValueError, match="Destination folder not set"):
            app._validate_dest_folder()

    def test_validate_dest_folder_not_str(self, app):
        app.dest_folder = 123
        with pytest.raises(ValueError, match="Destination folder must be a string"):
            app._validate_dest_folder()

    def test_validate_dest_folder_not_exist(self, app, tmp_path):
        app.dest_folder = str(tmp_path / "nonexistent")
        with pytest.raises(
            FileNotFoundError, match="Destination folder does not exist"
        ):
            app._validate_dest_folder()

    def test_validate_dest_folder_not_dir(self, app, tmp_path):
        file_path = tmp_path / "file.txt"
        file_path.write_text("hello")
        app.dest_folder = str(file_path)
        with pytest.raises(ValueError, match="Destination path is not a directory"):
            app._validate_dest_folder()

    def test_validate_dest_folder_no_access(self, app, tmp_path):
        app.dest_folder = str(tmp_path)
        with patch("os.access", return_value=False):
            with pytest.raises(PermissionError, match="Cannot read destination folder"):
                app._validate_dest_folder()

    def test_validate_dest_folder_empty_dir(self, app, tmp_path):
        app.dest_folder = str(tmp_path)
        with pytest.raises(
            ValueError, match="Destination folder is empty - nothing to archive"
        ):
            app._validate_dest_folder()

    def test_validate_dest_folder_iterdir_permission(self, app, tmp_path):
        app.dest_folder = str(tmp_path)
        with patch.object(Path, "iterdir", side_effect=PermissionError("denied")):
            with pytest.raises(
                PermissionError, match="Cannot access destination folder contents"
            ):
                app._validate_dest_folder()

    def test_validate_dest_folder_success(self, app, tmp_path):
        app.dest_folder = str(tmp_path)
        (tmp_path / "file.txt").write_text("hello")
        res = app._validate_dest_folder()
        assert res == tmp_path

    def test_count_zip_contents(self, app, tmp_path):
        app.dest_folder = str(tmp_path)
        (tmp_path / "file1.txt").write_text("abc")
        (tmp_path / "file2.txt").write_text("defg")

        # Will also create a directory
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file3.txt").write_text("hijkl")

        count, size = app._count_zip_contents()
        assert count == 3
        assert size == 12  # 3 + 4 + 5

        # Test permission error catching
        with patch("os.access", side_effect=PermissionError):
            count, size = app._count_zip_contents()
            assert count == 0
            assert size == 0

    def test_add_files_to_zip(self, app, tmp_path):
        app.dest_folder = str(tmp_path)
        (tmp_path / "file1.txt").write_text("abc")

        zipf = MagicMock()
        processed, size, failed = app._add_files_to_zip(zipf, 1)

        assert processed == 1
        assert size == 3
        assert failed == 0
        zipf.write.assert_called_once()

    def test_add_files_to_zip_cancel(self, app, tmp_path):
        app.dest_folder = str(tmp_path)
        (tmp_path / "file1.txt").write_text("abc")
        app.cancel_operation = True

        zipf = MagicMock()
        with pytest.raises(Exception, match="ZIP creation cancelled by user"):
            app._add_files_to_zip(zipf, 1)

    def test_add_files_to_zip_failed_access(self, app, tmp_path):
        app.dest_folder = str(tmp_path)
        (tmp_path / "file1.txt").write_text("abc")

        zipf = MagicMock()
        with patch("os.access", return_value=False):
            processed, size, failed = app._add_files_to_zip(zipf, 1)
            assert failed == 1
            assert processed == 0

    def test_add_files_to_zip_missing_file(self, app, tmp_path):
        app.dest_folder = str(tmp_path)
        (tmp_path / "file1.txt").write_text("abc")

        zipf = MagicMock()
        with patch.object(Path, "exists", return_value=False):
            processed, size, failed = app._add_files_to_zip(zipf, 1)
            assert failed == 1
            assert processed == 0

    def test_add_files_to_zip_write_exception(self, app, tmp_path):
        app.dest_folder = str(tmp_path)
        (tmp_path / "file1.txt").write_text("abc")

        zipf = MagicMock()
        zipf.write.side_effect = OSError("write error")
        processed, size, failed = app._add_files_to_zip(zipf, 1)
        assert failed == 1
        assert processed == 0

    def test_add_files_to_zip_ui_update_and_fail(self, app, tmp_path):
        app.dest_folder = str(tmp_path)
        # Create enough files to trigger the UI update modulo check
        for i in range(MAX_UI_UPDATE_FREQUENCY + 1):
            (tmp_path / f"file{i}.txt").write_text("abc")

        zipf = MagicMock()

        # Make one file throw IOError
        def mock_write(file, arc):
            if "file0" in str(file):
                raise OSError("disk error")

        zipf.write.side_effect = mock_write

        with patch.object(app, "update_progress") as mock_update:
            processed, size, failed = app._add_files_to_zip(
                zipf, MAX_UI_UPDATE_FREQUENCY + 1
            )
            assert processed == MAX_UI_UPDATE_FREQUENCY
            assert failed == 1
            mock_update.assert_called()

    def test_create_output_zip_success(self, app, tmp_path):
        app.dest_folder = str(tmp_path / "src")
        src = tmp_path / "src"
        src.mkdir()
        (src / "file1.txt").write_text("abc")

        # By default mock zip verification
        zip_path = app.create_output_zip()
        assert zip_path.endswith(".zip")
        # Ensure zip actually created
        assert Path(zip_path).exists()

    def test_create_output_zip_no_files(self, app, tmp_path):
        app.dest_folder = str(tmp_path / "src")
        src = tmp_path / "src"
        src.mkdir()
        (src / "file1.txt").write_text("abc")

        with patch.object(app, "_count_zip_contents", return_value=(0, 0)):
            with pytest.raises(Exception, match="No accessible files found"):
                app.create_output_zip()

    @patch("zipfile.ZipFile")
    def test_create_output_zip_not_created(self, mock_zip, app, tmp_path):
        app.dest_folder = str(tmp_path / "src")
        src = tmp_path / "src"
        src.mkdir()
        (src / "file1.txt").write_text("abc")

        # We mock zipfile so it doesn't create the file
        with pytest.raises(Exception, match="ZIP file was not created"):
            app.create_output_zip()

    def test_create_output_zip_fallback_unique(self, app, tmp_path):
        app.dest_folder = str(tmp_path / "src")
        src = tmp_path / "src"
        src.mkdir()
        (src / "file1.txt").write_text("abc")

        # Let's create the default ZIP file so that exists() returns True naturally
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"processed_files_{timestamp}.zip"
        zip_path = tmp_path / zip_filename
        zip_path.touch()

        zip_result = app.create_output_zip()
        assert zip_result.endswith("_unique")

    def test_create_output_zip_empty_size(self, app, tmp_path):
        app.dest_folder = str(tmp_path / "src")
        src = tmp_path / "src"
        src.mkdir()
        (src / "file1.txt").write_text("abc")

        original_stat = Path.stat

        class MockStat:
            st_size = 0

        def mock_stat(self_path, **kwargs):
            if "processed_files" in str(self_path.name):
                return MockStat()
            return original_stat(self_path, **kwargs)

        with patch.object(Path, "stat", side_effect=mock_stat, autospec=True):
            with pytest.raises(Exception, match="ZIP file is empty"):
                app.create_output_zip()

    def test_create_output_zip_stat_error(self, app, tmp_path):
        app.dest_folder = str(tmp_path / "src")
        src = tmp_path / "src"
        src.mkdir()
        (src / "file1.txt").write_text("abc")

        original_stat = Path.stat
        stat_calls = [0]

        def mock_stat(self_path, **kwargs):
            if "processed_files" in str(self_path.name):
                stat_calls[0] += 1
                if stat_calls[0] > 2:  # 1st exists() check, 2nd exists() check, 3rd is stat()
                    raise OSError("stat error")
            return original_stat(self_path, **kwargs)

        with patch.object(Path, "stat", side_effect=mock_stat, autospec=True):
            zip_path = app.create_output_zip()
        # Ensure we assert OUTSIDE the mock
        assert Path(zip_path).exists()

    def test_create_output_zip_with_failed_files(self, app, tmp_path):
        app.dest_folder = str(tmp_path / "src")
        src = tmp_path / "src"
        src.mkdir()
        (src / "file1.txt").write_text("abc")

        # To avoid ZIP file is empty, we must actually write to zipf in the mock
        def mock_add(zipf, total_files):
            file_path = str(src / "file1.txt")
            zipf.write(file_path, "file1.txt")
            return (1, 3, 1)

        with patch.object(app, "_add_files_to_zip", side_effect=mock_add):
            zip_path = app.create_output_zip()
            assert Path(zip_path).exists()

    def test_create_output_zip_failed_cleanup(self, app, tmp_path):
        app.dest_folder = str(tmp_path / "src")
        src = tmp_path / "src"
        src.mkdir()
        (src / "file1.txt").write_text("abc")

        # Mock a successful counting, then an error in addition
        with patch.object(app, "_count_zip_contents", return_value=(1, 3)):
            with patch.object(
                app, "_add_files_to_zip", side_effect=OSError("count error")
            ):
                original_exists = Path.exists

                def mock_exists(self_path, *args, **kwargs):
                    if "processed_files" in str(self_path.name):
                        return True
                    # Let other paths fallback
                    return original_exists(self_path, *args, **kwargs)

                with patch.object(Path, "exists", side_effect=mock_exists, autospec=True):
                    with patch.object(
                        Path, "unlink", side_effect=OSError("unlink error")
                    ):
                        with pytest.raises(
                            Exception, match="Failed to create ZIP archive: count error"
                        ):
                            app.create_output_zip()

    def test_create_output_zip_failed_cleanup_success(self, app, tmp_path):
        app.dest_folder = str(tmp_path / "src")
        src = tmp_path / "src"
        src.mkdir()
        (src / "file1.txt").write_text("abc")

        def mock_unlink(self_path, *args, **kwargs):
            pass

        with patch.object(
            app, "_count_zip_contents", side_effect=(OSError("count error"))
        ):
            original_exists = Path.exists

            def mock_exists(self_path, *args, **kwargs):
                if "processed_files" in str(self_path.name):
                    return True
                return original_exists(self_path, *args, **kwargs)

            with patch.object(Path, "exists", side_effect=mock_exists, autospec=True):
                with patch.object(Path, "unlink", side_effect=mock_unlink):
                    with pytest.raises(Exception):
                        app.create_output_zip()

    def test_create_output_zip_path_type_error(self, app, tmp_path):
        app.dest_folder = str(tmp_path / "src")
        src = tmp_path / "src"
        src.mkdir()
        (src / "file1.txt").write_text("abc")

        # Make the division operation fail since it is caught in the try block
        # zip_path = dest_path_obj.parent / zip_filename
        with patch.object(Path, "__truediv__", side_effect=TypeError("bad div")):
            with pytest.raises(
                ValueError, match="Cannot determine ZIP location: bad div"
            ):
                app.create_output_zip()
