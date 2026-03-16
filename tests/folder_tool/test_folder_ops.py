"""Unit tests for folder_tool/folder_ops.py."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from folder_tool.folder_ops import FolderOperationsMixin


class DummyVar:
    def __init__(self, val=False):
        self._val = val

    def get(self):
        return self._val


class DummyApp(FolderOperationsMixin):
    def __init__(self):
        self.dest_folder = ""
        self.source_folders = []
        self.cancel_operation = False
        self.preview_mode_var = DummyVar(False)

    def validate_file_filters(self, path):
        return True

    def get_organized_path(self, source_path, dest_folder):
        return str(Path(dest_folder) / Path(source_path).name)

    def _get_unique_path(self, path):
        return str(path)

    def _safe_copy_file(self, src, dest):
        return True

    def update_progress(self, val, msg):
        pass


@pytest.fixture
def app():
    return DummyApp()


class TestFolderOperationsMixin:
    def test_combine_folders_enhanced_success(self, app, tmp_path):
        src1 = tmp_path / "src1"
        src1.mkdir()
        (src1 / "f1.txt").write_text("a")
        (src1 / "f2.txt").write_text("b")

        app.source_folders = [str(src1)]
        app.dest_folder = str(tmp_path / "dest")

        with patch.object(
            app,
            "_get_unique_path",
            side_effect=lambda x: str(x) + "_uniq" if "f2" in str(x) else str(x),
        ):
            res = app._combine_folders_enhanced()
            assert "Processed 2 files." in res[0]
            assert "Renamed 1 files" in res[1]

    def test_combine_folders_enhanced_filter_skip(self, app, tmp_path):
        src1 = tmp_path / "src1"
        src1.mkdir()
        (src1 / "f1.txt").write_text("a")

        app.source_folders = [str(src1)]
        app.dest_folder = str(tmp_path / "dest")

        with patch.object(app, "validate_file_filters", return_value=False):
            res = app._combine_folders_enhanced()
            assert "Processed 0 files" in res[0]
            assert "Skipped 1 files" in res[2]

    def test_combine_folders_enhanced_copy_fail(self, app, tmp_path):
        src1 = tmp_path / "src1"
        src1.mkdir()
        (src1 / "f1.txt").write_text("a")

        app.source_folders = [str(src1)]
        app.dest_folder = str(tmp_path / "dest")

        with patch.object(app, "_safe_copy_file", return_value=False):
            res = app._combine_folders_enhanced()
            assert "Failed to copy 1 files" in res[3]

    def test_combine_folders_enhanced_preview(self, app, tmp_path):
        src1 = tmp_path / "src1"
        src1.mkdir()
        (src1 / "f1.txt").write_text("a")

        app.source_folders = [str(src1)]
        app.dest_folder = str(tmp_path / "dest")
        app.preview_mode_var._val = True

        res = app._combine_folders_enhanced()
        assert "PREVIEW MODE" in res[0]

    def test_combine_folders_enhanced_cancel(self, app, tmp_path):
        src1 = tmp_path / "src1"
        src1.mkdir()
        (src1 / "f1.txt").write_text("a")

        app.source_folders = [str(src1)]
        app.dest_folder = str(tmp_path / "dest")
        app.cancel_operation = True

        res = app._combine_folders_enhanced()
        assert "Processed 0 files" in res[0]

    def test_combine_folders_enhanced_exception(self, app, tmp_path):
        src1 = tmp_path / "src1"
        src1.mkdir()
        (src1 / "f1.txt").write_text("a")

        app.source_folders = [str(src1)]
        app.dest_folder = str(tmp_path / "dest")

        with patch.object(app, "_safe_copy_file", side_effect=ValueError("copy err")):
            res = app._combine_folders_enhanced()
            assert "Failed to copy 1 files" in res[3]
            assert "ERROR copying 'f1.txt': copy err" in "".join(res)

    def test_perform_deduplication_cancel_dialog(self, app, tmp_path):
        app.preview_mode_var._val = False
        with patch("tkinter.messagebox.askyesno", return_value=False):
            assert app._perform_deduplication(str(tmp_path)) == [
                "Deduplication cancelled by user."
            ]

    def test_perform_deduplication_success(self, app, tmp_path):
        d = tmp_path / "dedup"
        d.mkdir()
        (d / "f1.txt").write_text("a")
        (d / "f1 (1).txt").write_text("a")
        (d / "f1 (2).txt").write_text("a")

        # mock stat to control which is kept
        def mock_stat(self_path, **kwargs):
            m = MagicMock()
            if "(2)" in self_path.name:
                m.st_mtime = 100
            elif "(1)" in self_path.name:
                m.st_mtime = 50
            else:
                m.st_mtime = 10
            return m

        with patch.object(Path, "stat", side_effect=mock_stat, autospec=True):
            app.preview_mode_var._val = False
            with patch("tkinter.messagebox.askyesno", return_value=True):
                # mock unlink to avoid real deletion, we just want to verify logic
                with patch.object(Path, "unlink") as mock_unlink:
                    res = app._perform_deduplication(str(d))
                    assert mock_unlink.call_count == 2
                    assert "Deleted a total of 2 files" in res[1]

    def test_perform_deduplication_preview(self, app, tmp_path):
        d = tmp_path / "dedup"
        d.mkdir()
        (d / "f1.txt").write_text("a")
        (d / "f1 (1).txt").write_text("a")

        app.preview_mode_var._val = True
        with patch.object(Path, "unlink") as mock_unlink:
            res = app._perform_deduplication(str(d))
            mock_unlink.assert_not_called()
            assert "Would delete a total of 1 files" in res[1]

    def test_perform_deduplication_unlink_error(self, app, tmp_path):
        d = tmp_path / "dedup"
        d.mkdir()
        (d / "f1.txt").write_text("a")
        (d / "f1 (1).txt").write_text("a")

        app.preview_mode_var._val = False
        with patch("tkinter.messagebox.askyesno", return_value=True):
            with patch.object(Path, "unlink", side_effect=OSError("unlink err")):
                res = app._perform_deduplication(str(d))
                assert "FAILED to delete" in "".join(res)

    def test_perform_deduplication_stat_not_found(self, app, tmp_path):
        d = tmp_path / "dedup"
        d.mkdir()
        (d / "f1.txt").write_text("a")
        (d / "f1 (1).txt").write_text("a")

        app.preview_mode_var._val = False
        with patch("tkinter.messagebox.askyesno", return_value=True):
            with patch.object(Path, "stat", side_effect=FileNotFoundError):
                res = app._perform_deduplication(str(d))
                assert "Deleted a total of 0 files" in res[1]

    def test_perform_deduplication_cancel_mid_stream(self, app, tmp_path):
        d = tmp_path / "dedup"
        d.mkdir()
        (d / "f1.txt").write_text("a")

        app.cancel_operation = True
        app.preview_mode_var._val = True
        res = app._perform_deduplication(str(d))
        assert "a total of 0" in res[1]

    def test_run_deduplicate_main_op(self, app, tmp_path):
        app.source_folders = [str(tmp_path)]
        app.preview_mode_var._val = True
        res = app._run_deduplicate_main_op()
        assert "---" in res

    def test_run_deduplicate_main_op_cancel(self, app, tmp_path):
        app.source_folders = [str(tmp_path)]
        app.cancel_operation = True
        res = app._run_deduplicate_main_op()
        assert res == []

    def test_flatten_folders_success(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        sub = src / "sub"
        sub.mkdir()
        (sub / "f1.txt").write_text("a")

        app.source_folders = [str(src)]
        app.dest_folder = str(tmp_path / "dest")

        res = app._flatten_folders()
        assert "Flattened 1 files to destination root level" in res[0]

    def test_flatten_folders_cancel(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")

        app.source_folders = [str(src)]
        app.dest_folder = str(tmp_path / "dest")
        app.cancel_operation = True

        res = app._flatten_folders()
        assert "Flattened 0 files" in res[0]

    def test_flatten_folders_copy_fail(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")

        app.source_folders = [str(src)]
        app.dest_folder = str(tmp_path / "dest")

        with patch.object(app, "_safe_copy_file", return_value=False):
            res = app._flatten_folders()
            assert "Failed to copy 1 files" in res[2]

    def test_flatten_folders_filter_skip(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")

        app.source_folders = [str(src)]
        app.dest_folder = str(tmp_path / "dest")

        with patch.object(app, "validate_file_filters", return_value=False):
            res = app._flatten_folders()
            assert "Skipped 1 files" in res[1]

    def test_count_total_files(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")
        (src / "f2.txt").write_text("a")
        app.source_folders = [str(src)]
        assert app._count_total_files() == 2

    def test_prune_empty_folders_success(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        sub1 = src / "sub1"
        sub1.mkdir()
        sub2 = src / "sub2"  # empty
        sub2.mkdir()
        (sub1 / "f1.txt").write_text("a")

        app.source_folders = [str(src)]
        app.dest_folder = str(tmp_path / "dest")

        res = app._prune_empty_folders()
        assert "non-empty source folder(s)" in res[0]
        assert "Copied a total of 1 files." in res[1]
        assert "empty folders" in res[2]

    def test_prune_empty_folders_cancel(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]
        app.dest_folder = str(tmp_path / "dest")
        app.cancel_operation = True
        
        res = app._prune_empty_folders()
        assert "Processed " in res[0] and "non-empty source folder" in res[0]

    def test_prune_empty_folders_cancel_inner(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")
        app.source_folders = [str(src)]
        app.dest_folder = str(tmp_path / "dest")

        # Monkey patch os.walk to allow us to set cancel_operation mid-flight
        original_walk = os.walk

        def mock_walk(*args, **kwargs):
            app.cancel_operation = True
            return original_walk(*args, **kwargs)

        with patch("os.walk", side_effect=mock_walk):
            res = app._prune_empty_folders()
            assert "Processed 0 non-empty source" in res[0]

    def test_copy_single_file_in_prune_filter_skip(self, app, tmp_path):
        log = []
        with patch.object(app, "validate_file_filters", return_value=False):
            res = app._copy_single_file_in_prune(
                Path("src"), Path("dest"), "f1.txt", log
            )
            assert res == (0, 0)

    def test_copy_single_file_in_prune_fail(self, app, tmp_path):
        log = []
        with patch.object(app, "_safe_copy_file", return_value=False):
            res = app._copy_single_file_in_prune(
                Path("src"), Path("dest"), "f1.txt", log
            )
            assert res == (0, 1)

    def test_copy_single_file_in_prune_exception(self, app, tmp_path):
        log = []
        with patch.object(app, "_safe_copy_file", side_effect=TypeError("err")):
            res = app._copy_single_file_in_prune(
                Path("src"), Path("dest"), "f1.txt", log
            )
            assert res == (0, 1)

    def test_copy_single_file_in_prune_preview(self, app, tmp_path):
        log = []
        app.preview_mode_var._val = True
        with patch.object(app, "_safe_copy_file") as mock_copy:
            res = app._copy_single_file_in_prune(
                Path("src"), Path("dest"), "f1.txt", log
            )
            assert res == (1, 0)
            mock_copy.assert_not_called()
