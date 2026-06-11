"""Unit tests for folder_tool/folder_tool_processing.py."""

from unittest.mock import patch

# Load the un-namespaced graph first to prevent circular imports during namespaced load
import Folders_Tool_r0  # noqa: F401
import pytest

# Import the namespaced module to use in the test so `--cov` instruments it
from folder_tool.folder_tool_processing import ProcessingMixin


class DummyVar:
    def __init__(self, val=""):
        self._val = val

    def get(self):
        return self._val


class DummyApp(ProcessingMixin):
    def __init__(self):
        self.operation_mode = DummyVar()
        self.cancel_operation = False
        self.backup_before_var = DummyVar(False)
        self.unzip_var = DummyVar(False)
        self.deduplicate_var = DummyVar(False)
        self.zip_output_var = DummyVar(False)
        self.dest_folder = "dest"

    def validate_inputs(self, check_destination=True):
        return True

    def update_status(self, msg):
        pass

    def update_progress(self, val, msg):
        pass

    def generate_analysis_report(self):
        return "Report"

    def show_text_dialog(self, title, text):
        pass

    def _run_deduplicate_main_op(self):
        return ["Log 1"]

    def create_backup(self):
        return "backup_path"

    def _bulk_unzip_enhanced(self):
        return ["Log 1"]

    def _combine_folders_enhanced(self):
        return ["Combined"]

    def _flatten_folders(self):
        return ["Flattened"]

    def _prune_empty_folders(self):
        return ["Pruned"]

    def _perform_deduplication(self, path):
        return ["Deduped"]

    def create_output_zip(self):
        return "out.zip"


@pytest.fixture
def app():
    return DummyApp()


class TestProcessingMixin:
    def test_run_processing_analyze(self, app):
        app.operation_mode._val = "analyze"
        with patch.object(app, "_run_analyze_mode") as mock_analyze:
            app.run_processing()
            mock_analyze.assert_called_once()

    def test_run_processing_deduplicate(self, app):
        app.operation_mode._val = "deduplicate"
        with patch.object(app, "_run_deduplicate_mode") as mock_dedup:
            app.run_processing()
            mock_dedup.assert_called_once()

    def test_run_processing_combine(self, app):
        app.operation_mode._val = "combine"
        with patch.object(app, "_run_destination_workflow") as mock_dest:
            app.run_processing()
            mock_dest.assert_called_once_with("combine")

    def test_run_analyze_mode_invalid(self, app):
        with patch.object(app, "validate_inputs", return_value=False):
            with patch.object(app, "generate_analysis_report") as mock_gen:
                app._run_analyze_mode()
                mock_gen.assert_not_called()

    def test_run_analyze_mode_success(self, app):
        with patch("tkinter.messagebox.showinfo") as mock_info:
            app._run_analyze_mode()
            mock_info.assert_called_once()

    def test_run_analyze_mode_oserror(self, app):
        with patch.object(app, "generate_analysis_report", side_effect=OSError("err")):
            with patch("tkinter.messagebox.showerror") as mock_err:
                app._run_analyze_mode()
                mock_err.assert_called_once()

    def test_run_deduplicate_mode_invalid(self, app):
        with patch.object(app, "validate_inputs", return_value=False):
            with patch.object(app, "_run_deduplicate_main_op") as mock_op:
                app._run_deduplicate_mode()
                mock_op.assert_not_called()

    def test_run_deduplicate_mode_success(self, app):
        with patch("tkinter.messagebox.showinfo") as mock_info:
            app._run_deduplicate_mode()
            mock_info.assert_called_once()

    def test_run_deduplicate_mode_oserror(self, app):
        with patch.object(app, "_run_deduplicate_main_op", side_effect=OSError("err")):
            with patch("tkinter.messagebox.showerror") as mock_err:
                app._run_deduplicate_mode()
                mock_err.assert_called_once()

    def test_run_destination_workflow_invalid(self, app):
        with patch.object(app, "validate_inputs", return_value=False):
            with patch.object(app, "_run_pre_processing") as mock_pre:
                app._run_destination_workflow("combine")
                mock_pre.assert_not_called()

    def test_run_destination_workflow_backup_cancel(self, app):
        app.backup_before_var._val = True
        app.cancel_operation = True
        with patch.object(app, "create_backup", return_value=None):
            with patch.object(app, "_run_pre_processing") as mock_pre:
                app._run_destination_workflow("combine")
                mock_pre.assert_not_called()

    def test_run_destination_workflow_pre_fail(self, app):
        with patch.object(app, "_run_pre_processing", return_value=False):
            with patch.object(app, "_run_main_operation") as mock_main:
                app._run_destination_workflow("combine")
                mock_main.assert_not_called()

    def test_run_destination_workflow_main_fail(self, app):
        with patch.object(app, "_run_pre_processing", return_value=True):
            with patch.object(app, "_run_main_operation", return_value=None):
                with patch.object(app, "_run_post_processing") as mock_post:
                    app._run_destination_workflow("combine")
                    mock_post.assert_not_called()

    def test_run_destination_workflow_success(self, app):
        app.backup_before_var._val = True
        with patch("tkinter.messagebox.showinfo") as mock_info:
            app._run_destination_workflow("combine")
            mock_info.assert_called_once()

    def test_run_destination_workflow_cancel_after_post(self, app):
        app.cancel_operation = True
        with patch("tkinter.messagebox.showinfo") as mock_info:
            app._run_destination_workflow("combine")
            mock_info.assert_not_called()

    def test_run_pre_processing_skip(self, app):
        assert app._run_pre_processing()

    def test_run_pre_processing_cancel(self, app):
        app.unzip_var._val = True
        app.cancel_operation = True
        assert not app._run_pre_processing()

    def test_run_pre_processing_no_proceed(self, app):
        app.unzip_var._val = True
        with patch("tkinter.messagebox.askyesno", return_value=False):
            assert not app._run_pre_processing()

    def test_run_pre_processing_proceed(self, app):
        app.unzip_var._val = True
        with patch("tkinter.messagebox.askyesno", return_value=True):
            assert app._run_pre_processing()

    def test_run_pre_processing_oserror(self, app):
        app.unzip_var._val = True
        with patch.object(app, "_bulk_unzip_enhanced", side_effect=OSError("err")):
            with patch("tkinter.messagebox.showerror") as mock_err:
                assert not app._run_pre_processing()
                mock_err.assert_called_once()

    def test_run_main_operation_combine(self, app):
        assert "Combined" in app._run_main_operation("combine")

    def test_run_main_operation_flatten(self, app):
        assert "Flattened" in app._run_main_operation("flatten")

    def test_run_main_operation_prune(self, app):
        assert "Pruned" in app._run_main_operation("prune")

    def test_run_main_operation_cancel(self, app):
        app.cancel_operation = True
        assert app._run_main_operation("combine") is None

    def test_run_main_operation_oserror(self, app):
        with patch.object(app, "_combine_folders_enhanced", side_effect=OSError("err")):
            with patch("tkinter.messagebox.showerror") as mock_err:
                assert app._run_main_operation("combine") is None
                mock_err.assert_called_once()

    def test_run_post_processing_dedupe_success(self, app):
        app.deduplicate_var._val = True
        res = app._run_post_processing("Init", None)
        assert "Deduped" in res

    def test_run_post_processing_dedupe_oserror(self, app):
        app.deduplicate_var._val = True
        with patch.object(app, "_perform_deduplication", side_effect=OSError("err")):
            res = app._run_post_processing("Init", None)
            assert "Deduplication FAILED" in res

    def test_run_post_processing_zip_success(self, app):
        app.zip_output_var._val = True
        res = app._run_post_processing("Init", None)
        assert "out.zip" in res

    def test_run_post_processing_zip_oserror(self, app):
        app.zip_output_var._val = True
        with patch.object(app, "create_output_zip", side_effect=OSError("err")):
            res = app._run_post_processing("Init", None)
            assert "ZIP Creation FAILED" in res

    def test_run_post_processing_backup(self, app):
        res = app._run_post_processing("Init", "mybackup")
        assert "mybackup" in res
