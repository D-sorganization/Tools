"""Unit tests for folder_tool/folder_tool_ui_processing.py."""

import tkinter as tk
from unittest.mock import MagicMock, patch

import pytest

from folder_tool.folder_tool_ui_processing import UIProcessingMixin


class DummyVar:
    def __init__(self, val=None):
        self._val = val

    def set(self, val):
        self._val = val

    def get(self):
        return self._val


class DummyApp(UIProcessingMixin):
    def __init__(self):
        self.root = MagicMock()
        self.source_folders = []
        self.cancel_operation = False
        self.run_button = MagicMock()
        self.cancel_button = MagicMock()
        self.progress_var = DummyVar()
        self.status_var = DummyVar()
        self.source_info_label = MagicMock()
        self._callback_ran = False

    def run_processing(self):
        self._callback_ran = True


@pytest.fixture
def app():
    return DummyApp()


class TestUIProcessingMixin:
    def test_show_text_dialog_success(self, app):
        with patch.object(app, "_create_dialog_window") as mock_win:
            mock_win.return_value = (MagicMock(), 100, 100)
            with patch.object(app, "_create_text_area") as mock_txt:
                with patch.object(app, "_create_dialog_buttons") as mock_btn:
                    with patch.object(app, "_finalize_dialog") as mock_fin:
                        app.show_text_dialog("title", "content")
                        mock_win.assert_called_once()
                        mock_txt.assert_called_once()
                        mock_btn.assert_called_once()
                        mock_fin.assert_called_once()

    def test_show_text_dialog_type_error(self, app):
        with pytest.raises(ValueError):
            app.show_text_dialog(123, "content")

    def test_show_text_dialog_tcl_error(self, app):
        with patch.object(app, "_create_dialog_window", side_effect=tk.TclError("err")):
            with patch.object(app, "_show_fallback_messagebox") as mock_fall:
                with pytest.raises(tk.TclError):
                    app.show_text_dialog("title", "content")
                mock_fall.assert_called_once()

    def test_show_text_dialog_value_error(self, app):
        with patch.object(app, "_create_dialog_window", side_effect=ValueError("err")):
            with patch.object(app, "_show_fallback_messagebox") as mock_fall:
                with pytest.raises(ValueError):
                    app.show_text_dialog("title", "content")
                mock_fall.assert_called_once()

    def test_validate_dialog_inputs_empty_title(self, app):
        with pytest.raises(ValueError, match="cannot be empty"):
            app._validate_dialog_inputs("   ", "content")

    def test_validate_dialog_inputs_empty_content(self, app):
        with pytest.raises(ValueError, match="cannot be empty"):
            app._validate_dialog_inputs("title", "   ")

    def test_validate_dialog_inputs_long_content(self, app):
        from folder_tool.Folders_Tool_r0 import MAX_TEXT_CONTENT_SIZE

        long_str = "a" * (MAX_TEXT_CONTENT_SIZE + 1000)
        res = app._validate_dialog_inputs("title", long_str)
        assert len(res) < len(long_str)
        assert "[Content truncated" in res

    def test_create_dialog_window(self, app):
        with patch("tkinter.Toplevel") as mock_top:
            mock_win = MagicMock()
            mock_top.return_value = mock_win
            dialog, w, h = app._create_dialog_window("title", "a\nb\nc")
            assert dialog == mock_win
            mock_win.title.assert_called_with("title")

    def test_create_text_area(self, app):
        dialog = MagicMock()
        with patch("tkinter.Text"):
            with patch("tkinter.ttk.Frame"):
                with patch("tkinter.ttk.Scrollbar"):
                    res = app._create_text_area(dialog, "content")
                    assert res is not None

    def test_create_text_area_error(self, app):
        dialog = MagicMock()
        with patch("tkinter.Text") as mock_txt:
            mock_inst = MagicMock()
            mock_txt.return_value = mock_inst
            mock_inst.insert.side_effect = [
                ValueError("err"),
                None,
            ]  # Fail first time, success second time (fallback)
            with patch("tkinter.ttk.Frame"):
                with patch("tkinter.ttk.Scrollbar"):
                    res = app._create_text_area(dialog, "content")
                    assert res == mock_inst
                    assert mock_inst.insert.call_count == 2

    def test_create_dialog_buttons(self, app):
        dialog = MagicMock()
        with patch("tkinter.ttk.Frame"):
            with patch("tkinter.ttk.Button") as mock_btn:
                mock_b = MagicMock()
                mock_btn.return_value = mock_b
                res = app._create_dialog_buttons(dialog, "content")
                assert res == mock_b

    def test_finalize_dialog(self, app):
        dialog = MagicMock()
        app._finalize_dialog(dialog, 100, 100)
        dialog.focus_set.assert_called_once()
        dialog.wait_window.assert_called_once()

    def test_show_fallback_messagebox(self, app):
        with patch("tkinter.messagebox.showinfo") as mock_info:
            app._show_fallback_messagebox("title", "content")
            mock_info.assert_called_once()

    def test_show_fallback_messagebox_truncated(self, app):
        from folder_tool.Folders_Tool_r0 import MAX_FALLBACK_CONTENT_SIZE

        with patch("tkinter.messagebox.showinfo") as mock_info:
            app._show_fallback_messagebox(
                "title", "a" * (MAX_FALLBACK_CONTENT_SIZE + 10)
            )
            mock_info.assert_called_once()

    def test_update_source_info_empty(self, app):
        app.update_source_info()
        app.source_info_label.config.assert_called_with(text="")

    def test_update_source_info_missing_folder(self, app, tmp_path):
        app.source_folders = [str(tmp_path / "missing")]
        app.update_source_info()
        app.source_info_label.config.assert_called_with(
            text="Warning: No accessible source folders", foreground="red"
        )

    def test_update_source_info_no_access(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        app.source_folders = [str(src)]
        with patch("os.access", return_value=False):
            app.update_source_info()
            app.source_info_label.config.assert_called_with(
                text="Warning: No accessible source folders", foreground="red"
            )

    def test_update_source_info_success(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")
        app.source_folders = [str(src)]

        app.update_source_info()
        args, kwargs = app.source_info_label.config.call_args
        assert "Total: 1 files" in kwargs["text"]
        assert kwargs["foreground"] == "blue"

    def test_update_source_info_partial_access(self, app, tmp_path):
        src1 = tmp_path / "src1"
        src1.mkdir()
        (src1 / "f1.txt").write_text("a")

        src2 = tmp_path / "src2"
        src2.mkdir()

        app.source_folders = [str(src1), str(src2)]

        def mock_access(path, mode):
            if "src2" in str(path):
                return False
            return True

        with patch("os.access", side_effect=mock_access):
            app.update_source_info()
            args, kwargs = app.source_info_label.config.call_args
            assert "Total: 1 files" in kwargs["text"]
            assert kwargs["foreground"] == "orange"

    def test_update_source_info_oserror_file(self, app, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("a")
        app.source_folders = [str(src)]

        def mock_getsize(path):
            raise OSError("err")

        with patch("os.path.getsize", side_effect=mock_getsize):
            app.update_source_info()
            args, kwargs = app.source_info_label.config.call_args
            assert "Total: 0 files" in kwargs["text"]

    def test_run_processing_threaded(self, app):
        with patch("threading.Thread") as mock_thread:
            app.run_processing_threaded()
            mock_thread.assert_called_once()
            assert app.run_button.config.called
            assert app.cancel_button.config.called
            assert not app.cancel_operation

    def test_cancel_processing(self, app):
        with patch.object(app, "update_status") as mock_status:
            app.cancel_processing()
            assert app.cancel_operation
            mock_status.assert_called_with("Cancelling operation...")

    def test_processing_complete(self, app):
        with patch.object(app, "update_status") as mock_status:
            app.processing_complete()
            assert app.run_button.config.called
            assert app.cancel_button.config.called
            assert app.progress_var._val == 0
            mock_status.assert_called_with("Ready")

    def test_update_progress_success(self, app):
        with patch.object(app, "update_status") as mock_status:
            app.update_progress(50, "halfway")
            assert app.progress_var._val == 50
            app.root.update_idletasks.assert_called_once()
            mock_status.assert_called_with("halfway")

    def test_update_progress_invalid_type(self, app):
        app.update_progress("invalid", "msg")
        assert app.progress_var._val is None  # Unchanged

    def test_update_progress_clamp(self, app):
        app.update_progress(150)
        assert app.progress_var._val == 100
        app.update_progress(-10)
        assert app.progress_var._val == 0

    def test_update_status_success(self, app):
        app.update_status("test")
        assert app.status_var._val == "test"
        app.root.update_idletasks.assert_called_once()

    def test_update_status_long(self, app):
        from folder_tool.Folders_Tool_r0 import MAX_STATUS_LENGTH

        long_str = "a" * (MAX_STATUS_LENGTH + 10)
        app.update_status(long_str)
        assert len(app.status_var._val) == MAX_STATUS_LENGTH
        assert app.status_var._val.endswith("...")

    def test_update_progress_exception(self, app):
        # Trigger ValueError when setting progress_var by mocking set
        app.progress_var.set = MagicMock(side_effect=ValueError("err"))
        app.update_progress(50)
        # Should catch exception internally and not raise
        assert app.progress_var.set.called

    def test_update_status_exception(self, app):
        app.status_var.set = MagicMock(side_effect=RuntimeError("err"))
        app.update_status("msg")
        assert app.status_var.set.called
