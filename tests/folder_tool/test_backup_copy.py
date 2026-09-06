"""Unit tests for folder_tool/backup_copy.py."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from folder_tool.backup_copy import BackupCopyMixin


class DummyApp(BackupCopyMixin):
    def __init__(self):
        self.source_folders = []
        self.cancel_operation = False

    def update_status(self, msg):
        pass

    def update_progress(self, val, msg):
        pass


@pytest.fixture
def app():
    return DummyApp()


class TestBackupCopyMixin:
    def test_validated_source_folders_empty(self):
        with pytest.raises(ValueError, match="No source folders"):
            BackupCopyMixin._validated_source_folders([])

        with pytest.raises(ValueError, match="Source folders must be a list"):
            BackupCopyMixin._validated_source_folders("not_a_list")

    def test_validated_source_folders_invalid_type(self):
        with pytest.raises(ValueError, match="No valid source folders"):
            BackupCopyMixin._validated_source_folders([None, 123])

    def test_validated_source_folders_not_exists(self, tmp_path):
        bad_path = tmp_path / "does_not_exist"
        with pytest.raises(ValueError, match="No valid source folders"):
            BackupCopyMixin._validated_source_folders([str(bad_path)])

    def test_validated_source_folders_no_access(self, tmp_path):
        folder = tmp_path / "folder"
        folder.mkdir()
        with patch("os.access", return_value=False):
            with pytest.raises(ValueError, match="No valid source folders"):
                BackupCopyMixin._validated_source_folders([str(folder)])

    def test_validated_source_folders_success(self, tmp_path):
        f1 = tmp_path / "f1"
        f1.mkdir()
        res = BackupCopyMixin._validated_source_folders([str(f1)])
        assert res == [str(f1)]

    def test_backup_single_folder_not_exists(self, app, tmp_path):
        folder = tmp_path / "does_not_exist"
        assert not app._backup_single_folder(str(folder), tmp_path)

    def test_backup_single_folder_copy_fails(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"

        with patch("shutil.copytree", side_effect=OSError("copy error")):
            assert not app._backup_single_folder(str(src), dest)

    def test_backup_single_folder_cleanup_fails(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"

        with patch("shutil.copytree", side_effect=OSError("copy error")):
            with patch("shutil.rmtree", side_effect=OSError("rmtree error")):
                # In order for rmtree to be called, dest path must exist
                original_exists = Path.exists

                def mock_exists(self_path, *args, **kwargs):
                    if "dest" in str(self_path):
                        return True
                    return original_exists(self_path, *args, **kwargs)

                with patch.object(
                    Path, "exists", side_effect=mock_exists, autospec=True
                ):
                    assert not app._backup_single_folder(str(src), dest)

    def test_backup_single_folder_empty_dest(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"

        def mock_copytree(src, dst):
            Path(dst).mkdir()

        with patch("shutil.copytree", side_effect=mock_copytree):
            assert not app._backup_single_folder(str(src), dest)

    def test_backup_single_folder_success(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "file.txt").write_text("a")
        dest = tmp_path / "dest"
        assert app._backup_single_folder(str(src), dest)
        assert (dest / "src" / "file.txt").exists()

    def test_backup_single_folder_unique_path_error(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        (dest / "src").mkdir(parents=True)

        with patch.object(app, "_get_unique_path", side_effect=OSError("path error")):
            assert not app._backup_single_folder(str(src), dest)

    def test_cleanup_backup_dir(self, tmp_path):
        d = tmp_path / "dir"
        d.mkdir()
        BackupCopyMixin._cleanup_backup_dir(d)
        assert not d.exists()

        d2 = tmp_path / "dir2"
        d2.mkdir()
        with patch("shutil.rmtree", side_effect=OSError("rm error")):
            BackupCopyMixin._cleanup_backup_dir(d2)

    def test_create_backup_cancel(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]
        app.cancel_operation = True
        assert app.create_backup() is None

    def test_create_backup_mkdir_fail(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]

        with patch.object(Path, "mkdir", side_effect=OSError("mkdir error")):
            with pytest.raises(OSError):
                app.create_backup()

    def test_create_backup_base_not_exists(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]

        original_exists = Path.exists

        def mock_exists(self_path, *args, **kwargs):
            if self_path.name.startswith("backup_"):
                return False
            return original_exists(self_path, *args, **kwargs)

        with patch.object(Path, "exists", side_effect=mock_exists, autospec=True):
            with pytest.raises(OSError):
                app.create_backup()

    def test_create_backup_no_access(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]

        with patch(
            "os.access", side_effect=[True, False]
        ):  # 1st for validate_src, 2nd for backup_base
            with pytest.raises(PermissionError):
                app.create_backup()

    def test_create_backup_all_failed(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]

        with patch.object(app, "_backup_single_folder", return_value=False):
            assert app.create_backup() is None

    def test_create_backup_empty_result(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]

        # backup succeeds but iterdir is empty
        with patch.object(app, "_backup_single_folder", return_value=True):
            original_iterdir = Path.iterdir

            def mock_iterdir(self_path, *args, **kwargs):
                if self_path.name.startswith("backup_"):
                    return iter([])
                return original_iterdir(self_path, *args, **kwargs)

            with patch.object(Path, "iterdir", side_effect=mock_iterdir, autospec=True):
                assert app.create_backup() is None

    def test_create_backup_success(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "file.txt").write_text("a")
        app.source_folders = [str(src)]

        res = app.create_backup()
        assert res is not None
        assert Path(res).exists()

    def test_safe_copy_file_invalid_inputs(self, app, tmp_path):
        with pytest.raises((AssertionError, ValueError)):
            app._safe_copy_file(None, "dest")
        with pytest.raises(ValueError):
            app._safe_copy_file("", "dest")
        with pytest.raises(ValueError):
            app._safe_copy_file("src", None)
        with pytest.raises(ValueError):
            app._safe_copy_file("src", "")

    def test_safe_copy_file_not_exists(self, app, tmp_path):
        with pytest.raises(FileNotFoundError):
            app._safe_copy_file(str(tmp_path / "none.txt"), "dest")

    def test_safe_copy_file_not_file(self, app, tmp_path):
        d = tmp_path / "dir"
        d.mkdir()
        with pytest.raises(ValueError):
            app._safe_copy_file(str(d), "dest")

    def test_safe_copy_file_no_access(self, app, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("a")
        with patch("os.access", return_value=False):
            with pytest.raises(PermissionError):
                app._safe_copy_file(str(f), "dest")

    def test_safe_copy_file_size_warning(self, app, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("a")
        dest = tmp_path / "dest.txt"

        original_stat = Path.stat

        def mock_stat(self_path, **kwargs):
            if self_path.name == "file.txt":
                from unittest.mock import MagicMock

                m = MagicMock()
                m.st_size = 0
                m.st_mode = 33188
                return m
            return original_stat(self_path, **kwargs)

        with patch.object(Path, "stat", side_effect=mock_stat, autospec=True):
            app._safe_copy_file(str(f), str(dest))

    def test_safe_copy_file_size_error_stat(self, app, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("a")
        dest = tmp_path / "dest.txt"

        original_stat = Path.stat

        def mock_stat(self_path, **kwargs):
            if self_path.name == "dest.txt":
                raise OSError("stat error")
            return original_stat(self_path, **kwargs)

        with patch.object(Path, "stat", side_effect=mock_stat, autospec=True):
            # Stat errors during size checks/verification are safely handled and
            # return False
            assert not app._safe_copy_file(str(f), str(dest))

    def test_verify_copy_no_dest(self, app, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        assert not app._verify_copy(src, dest, "src", "dest")

    def test_verify_copy_success(self, app, tmp_path):
        src = tmp_path / "src"
        src.write_text("a")
        dest = tmp_path / "dest"
        dest.write_text("a")

        assert app._verify_copy(src, dest, str(src), str(dest))

    def test_verify_copy_mismatch(self, app, tmp_path):
        src = tmp_path / "src"
        src.write_text("a")
        dest = tmp_path / "dest"
        dest.write_text("abc")
        assert not app._verify_copy(src, dest, str(src), str(dest))

    def test_verify_copy_stat_error(self, app, tmp_path):
        src = tmp_path / "src"
        src.write_text("a")
        dest = tmp_path / "dest"
        dest.write_text("a")

        original_stat = Path.stat

        def mock_stat(self_path, **kwargs):
            if self_path.name == "src":
                raise OSError("stat err")
            return original_stat(self_path, **kwargs)

        with patch.object(Path, "stat", side_effect=mock_stat, autospec=True):
            assert not app._verify_copy(src, dest, str(src), str(dest))

    def test_prepare_dest_directory_fails(self, app, tmp_path):
        d = tmp_path / "dir"
        with patch("os.access", return_value=False):
            with pytest.raises(PermissionError):
                app._prepare_dest_directory(d)

    def test_safe_copy_file_retry_fail(self, app, tmp_path):
        src = tmp_path / "src"
        src.write_text("a")
        dest = tmp_path / "dest"

        with patch("shutil.copy2", side_effect=OSError("copy failed")):
            with pytest.raises(OSError):
                app._safe_copy_file(str(src), str(dest))

    def test_safe_copy_file_retry_verify_fail(self, app, tmp_path):
        src = tmp_path / "src"
        src.write_text("a")
        dest = tmp_path / "dest"

        with patch.object(app, "_verify_copy", return_value=False):
            assert not app._safe_copy_file(str(src), str(dest))

    def test_get_unique_path_invalid(self, app):
        with pytest.raises(ValueError):
            app._get_unique_path("")

    def test_get_unique_path_not_exists(self, app, tmp_path):
        p = tmp_path / "none.txt"
        assert app._get_unique_path(str(p)) == str(p)

    def test_get_unique_path_format_error(self, app, tmp_path):
        p = tmp_path / "file.txt"
        with patch.object(Path, "is_absolute", side_effect=OSError("abs error")):
            with pytest.raises(ValueError):
                app._get_unique_path(str(p))

    def test_get_unique_path_drive_missing(self, app):
        if sys.platform == "win32":
            with pytest.raises(ValueError):
                app._get_unique_path("Z:\\missing.txt")

    def test_get_unique_path_exists_error(self, app, tmp_path):
        p = tmp_path / "file.txt"
        original_exists = Path.exists

        def mock_exists(self_path, *args, **kwargs):
            if self_path.name == "file.txt":
                raise OSError("exist error")
            return original_exists(self_path, *args, **kwargs)

        with patch.object(Path, "exists", side_effect=mock_exists, autospec=True):
            assert app._get_unique_path(str(p)) == str(p)

    def test_get_unique_path_is_file_error(self, app, tmp_path):
        p = tmp_path / "file.txt"
        p.write_text("a")

        def mock_is_file(self_path, *args, **kwargs):
            raise OSError("is_file error")

        with patch.object(Path, "is_file", side_effect=mock_is_file, autospec=True):
            res = app._get_unique_path(str(p))
            assert "(1)" in res

    def test_get_unique_path_success(self, app, tmp_path):
        p = tmp_path / "file.txt"
        p.write_text("a")
        res = app._get_unique_path(str(p))
        assert "(1)" in res

    def test_get_unique_path_exhaust_counter(self, app, tmp_path):
        p = tmp_path / "file.txt"
        p.write_text("a")

        # mock that everything exists up to counter limit
        original_exists = Path.exists

        def mock_exists(self_path, *args, **kwargs):
            if "(" in str(self_path):
                return True
            return original_exists(self_path, *args, **kwargs)

        with patch.object(Path, "exists", side_effect=mock_exists, autospec=True):
            res = app._get_unique_path(str(p))
            # it should fallback to timestamp
            assert "_" in res and "(" not in res

    def test_get_unique_path_counter_exist_check_error(self, app, tmp_path):
        p = tmp_path / "file.txt"
        p.write_text("a")

        original_exists = Path.exists

        def mock_exists(self_path, *args, **kwargs):
            if "(" in str(self_path):
                raise OSError("fail check counter")
            return original_exists(self_path, *args, **kwargs)

        with patch.object(Path, "exists", side_effect=mock_exists, autospec=True):
            res = app._get_unique_path(str(p))
            assert "(1)" in res

    def test_create_backup_parent_path_error(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]

        class MockExceptionProp(property):
            def __get__(self, obj, type=None):
                raise OSError("parent err")

        with patch.object(Path, "parent", MockExceptionProp()):
            with pytest.raises(ValueError):
                app.create_backup()
