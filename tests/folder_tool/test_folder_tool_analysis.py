"""Unit tests for folder_tool/folder_tool_analysis.py."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from folder_tool.folder_tool_analysis import AnalysisMixin


class DummyVar:
    def __init__(self, val=""):
        self._val = val

    def get(self):
        return self._val


class DummyApp(AnalysisMixin):
    def __init__(self):
        self.source_folders = []
        self.dest_folder = ""
        self.cancel_operation = False
        self.filter_extensions = DummyVar("")
        self.min_file_size = DummyVar("0")
        self.max_file_size = DummyVar("100")

    def _validate_constants(self):
        pass

    def validate_size_inputs(self):
        return True


@pytest.fixture
def app():
    return DummyApp()


class TestAnalysisMixin:
    def test_validate_source_folders_empty(self, app):
        with pytest.raises(ValueError, match="No source folders"):
            app._validate_source_folders()

    def test_validate_source_folders_not_list(self, app):
        app.source_folders = "str"
        with pytest.raises(ValueError, match="must be a list"):
            app._validate_source_folders()

    def test_validate_source_folders_invalid_type(self, app, tmp_path):
        app.source_folders = [None, 123]
        with pytest.raises(ValueError, match="No valid source folders"):
            app._validate_source_folders()

    def test_validate_source_folders_not_exists(self, app, tmp_path):
        app.source_folders = [str(tmp_path / "missing")]
        with pytest.raises(ValueError, match="No valid source folders"):
            app._validate_source_folders()

    def test_validate_source_folders_no_access(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]
        with patch("os.access", return_value=False):
            with pytest.raises(ValueError, match="No valid source folders"):
                app._validate_source_folders()

    def test_validate_source_folders_success(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]
        assert app._validate_source_folders() == [str(src)]

    def test_analyze_single_folder_success(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        f1 = src / "f1.txt"
        f1.write_text("a")
        f2 = src / "f2.txt"
        f2.write_text("a")

        file_types = {"": 0, ".txt": 0}
        size_by_type = {"": 0, ".txt": 0}
        largest = []

        files, size, errors = app._analyze_single_folder(
            str(src), file_types, size_by_type, largest
        )
        assert files == 2
        assert size == 2
        assert errors == 0

    def test_analyze_single_folder_cancel(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.cancel_operation = True

        file_types = {}
        size_by_type = {}
        largest = []

        files, size, errors = app._analyze_single_folder(
            str(src), file_types, size_by_type, largest
        )
        assert files == 0

    def test_analyze_single_folder_cancel_inner(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")

        # Monkey patch os.walk to allow mid-flight cancellation
        original_walk = os.walk

        def mock_walk(*args, **kwargs):
            app.cancel_operation = True
            return original_walk(*args, **kwargs)

        with patch("os.walk", side_effect=mock_walk):
            files, size, errors = app._analyze_single_folder(str(src), {}, {}, [])
            assert files == 0

    def test_analyze_single_folder_no_access_file(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")

        with patch("os.access", return_value=False):
            files, size, errors = app._analyze_single_folder(str(src), {}, {}, [])
            assert errors == 1

    def test_analyze_single_folder_file_too_small(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")

        with patch("os.path.getsize", return_value=-1):  # Below MIN_FILE_SIZE_BYTES
            files, size, errors = app._analyze_single_folder(str(src), {}, {}, [])
            assert files == 0

    def test_analyze_single_folder_file_too_large(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")

        from folder_tool.Folders_Tool_r0 import MAX_FILE_SIZE_MB

        with patch("os.path.getsize", return_value=MAX_FILE_SIZE_MB * 1024 * 1024 + 1):
            ft = {".txt": 0}
            st = {".txt": 0}
            largest = []
            files, size, errors = app._analyze_single_folder(str(src), ft, st, largest)
            assert files == 1

    def test_analyze_single_folder_getsize_error(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")

        def mock_getsize(self_path):
            raise OSError("err")

        with patch("os.path.getsize", side_effect=mock_getsize):
            files, size, errors = app._analyze_single_folder(str(src), {}, {}, [])
            assert errors == 1

    def test_analyze_single_folder_largest_limit(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        for i in range(15):
            (src / f"f{i}.txt").write_text("a" * i)

        ft = {".txt": 0}
        st = {".txt": 0}
        largest = []
        app._analyze_single_folder(str(src), ft, st, largest)
        assert len(largest) <= 10

    def test_format_report_summary(self, app):
        ft = {".txt": 5, ".pdf": 2}
        st = {".txt": 500, ".pdf": 2000000}
        largest = [(Path("file.pdf"), 2000000)]
        errors = ["error 1"]
        res = app._format_report_summary(7, 2000500, ft, st, largest, errors, 1)
        res_str = "\n".join(res)
        assert "TOTAL FILES: 7" in res_str
        assert "error 1" in res_str

    def test_generate_analysis_report_cancel(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]
        app.cancel_operation = True
        assert app.generate_analysis_report() is None

    def test_generate_analysis_report_success(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")
        app.source_folders = [str(src)]

        res = app.generate_analysis_report()
        assert res is not None
        assert "TOTAL FILES: 1" in res

    def test_generate_analysis_report_folder_errors(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")

        app.source_folders = [str(src)]
        with patch.object(app, "_analyze_single_folder", return_value=(0, 0, 1)):
            res = app.generate_analysis_report()
            assert "1 access errors" in res

    def test_generate_analysis_report_os_error(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]

        with patch.object(app, "_analyze_single_folder", side_effect=OSError("err")):
            res = app.generate_analysis_report()
            assert "ERROR: Error accessing folder" in res

    def test_validate_inputs_no_source(self, app, tmp_path):
        with patch("tkinter.messagebox.showerror") as mock_err:
            assert not app.validate_inputs()
            mock_err.assert_called()

    def test_validate_inputs_no_dest(self, app, tmp_path):
        src = tmp_path / "src"
        app.source_folders = [str(src)]
        with patch("tkinter.messagebox.showerror") as mock_err:
            assert not app.validate_inputs()
            mock_err.assert_called()

    def test_validate_inputs_dest_is_source(self, app, tmp_path):
        src = tmp_path / "src"
        app.source_folders = [str(src)]
        app.dest_folder = str(src)
        with patch("tkinter.messagebox.showerror") as mock_err:
            assert not app.validate_inputs()
            mock_err.assert_called()

    def test_validate_inputs_size_invalid(self, app, tmp_path):
        src = tmp_path / "src"
        app.source_folders = [str(src)]
        app.dest_folder = "dest"

        with patch.object(app, "validate_size_inputs", return_value=False):
            assert not app.validate_inputs()

    def test_validate_inputs_extension_no_dot(self, app, tmp_path):
        src = tmp_path / "src"
        app.source_folders = [str(src)]
        app.dest_folder = "dest"
        app.filter_extensions._val = "txt"

        with patch("tkinter.messagebox.showwarning") as mock_warn:
            assert not app.validate_inputs()
            mock_warn.assert_called()

    def test_validate_inputs_extension_success(self, app, tmp_path):
        src = tmp_path / "src"
        app.source_folders = [str(src)]
        app.dest_folder = "dest"
        app.filter_extensions._val = ".txt, .pdf"
        assert app.validate_inputs()

    def test_validate_application_state_success(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]
        app.dest_folder = str(tmp_path / "dest")
        Path(app.dest_folder).mkdir()
        app.filter_extensions._val = ".txt"

        res = app.validate_application_state()
        assert res["source_folders_exist"]
        assert res["destination_exists"]
        assert res["size_inputs_valid"]
        assert res["extension_filter_valid"]
        assert res["constants_valid"]

    def test_validate_application_state_empty(self, app):
        res = app.validate_application_state()
        assert res["source_folders_exist"]  # Returns True if empty
        assert res["destination_exists"]

    def test_validate_application_state_size_value_error(self, app):
        app.min_file_size._val = "invalid"
        res = app.validate_application_state()
        assert not res["size_inputs_valid"]

    def test_validate_application_state_extension_invalid(self, app):
        app.filter_extensions._val = "txt"
        res = app.validate_application_state()
        assert not res["extension_filter_valid"]

    def test_validate_application_state_constants_invalid(self, app):
        with patch.object(app, "_validate_constants", side_effect=ValueError):
            res = app.validate_application_state()
            assert not res["constants_valid"]
