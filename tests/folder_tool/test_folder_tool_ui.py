"""Unit tests for folder_tool/folder_tool_ui.py."""

import tkinter as tk
from unittest.mock import MagicMock, patch

import pytest

from folder_tool.folder_tool_ui import UICreationMixin


class DummyVar:
    def __init__(self, val=""):
        self._val = val

    def set(self, val):
        self._val = val

    def get(self):
        return self._val


class DummyApp(UICreationMixin):
    def __init__(self):
        try:
            self.root = tk.Tk()
        except tk.TclError:
            self.root = MagicMock()
        self.source_folders = []
        self.dest_folder = ""
        self.unzip_var = DummyVar()
        self.safe_extract_var = DummyVar()
        self.deduplicate_var = DummyVar()
        self.operation_mode = DummyVar("combine")
        self.zip_output_var = DummyVar()
        self.filter_extensions = DummyVar()
        self.organize_by_type_var = DummyVar()
        self.organize_by_date_var = DummyVar()
        self.min_file_size = DummyVar()
        self.max_file_size = DummyVar()
        self.preview_mode_var = DummyVar()
        self.backup_before_var = DummyVar()
        self.progress_var = DummyVar()
        self.status_var = DummyVar()

    def update_source_info(self):
        pass

    def run_processing_threaded(self):
        pass

    def cancel_processing(self):
        pass


@pytest.fixture
def app():
    app_instance = DummyApp()
    yield app_instance
    try:
        root = app_instance.root
        if hasattr(root, "destroy") and not isinstance(root, MagicMock):
            root.destroy()
    except Exception:
        pass


class TestUICreationMixin:
    def test_setup_application_icon_success(self, app):
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(app, "_load_ico_icon") as mock_ico:
                app._setup_application_icon()
                mock_ico.assert_called_once()

    def test_setup_application_icon_fallback(self, app):
        with patch("pathlib.Path.exists", return_value=False):
            with patch.object(app, "_load_png_fallback") as mock_png:
                app._setup_application_icon()
                mock_png.assert_called_once()

    def test_setup_application_icon_error(self, app):
        with patch.object(app, "_set_windows_app_id", side_effect=OSError("err")):
            app._setup_application_icon()

    def test_set_windows_app_id_success(self, app):
        mock_windll = MagicMock()
        with patch("sys.platform", "win32"):
            with patch("ctypes.windll", mock_windll, create=True):
                app._set_windows_app_id()
                mock_windll.shell32.SetCurrentProcessExplicitAppUserModelID.assert_called_once()

    def test_set_windows_app_id_error(self, app):
        mock_windll = MagicMock()
        set_id = mock_windll.shell32.SetCurrentProcessExplicitAppUserModelID
        set_id.side_effect = TypeError("mock error")
        with patch("sys.platform", "win32"):
            with patch("ctypes.windll", mock_windll, create=True):
                app._set_windows_app_id()

    def test_load_ico_icon_success(self, app):
        with patch.object(app.root, "iconbitmap") as mock_bmp:
            with patch.object(app.root, "iconphoto") as mock_photo:
                with patch("PIL.Image.open") as mock_open:
                    mock_img = MagicMock()
                    mock_img.mode = "RGB"
                    mock_resized = MagicMock()
                    mock_resized.mode = "RGB"
                    mock_resized.convert.return_value = MagicMock()
                    mock_img.resize.return_value = mock_resized
                    mock_open.return_value = mock_img
                    with patch("PIL.ImageTk.PhotoImage"):
                        app._load_ico_icon("test.ico")
                        mock_bmp.assert_called_with("test.ico")
                        mock_photo.assert_called()

    def test_load_ico_icon_resize_error(self, app):
        with patch.object(app.root, "iconbitmap"):
            with patch("PIL.Image.open") as mock_open:
                mock_img = MagicMock()
                mock_img.resize.side_effect = ValueError("err")
                mock_open.return_value = mock_img
                app._load_ico_icon("test.ico")

    def test_load_png_fallback_success(self, app):
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(app.root, "iconphoto") as mock_photo:
                with patch("PIL.Image.open") as mock_open:
                    mock_img = MagicMock()
                    mock_img.mode = "RGBA"
                    mock_img.resize.return_value = mock_img
                    mock_open.return_value = mock_img
                    with patch("PIL.ImageTk.PhotoImage"):
                        app._load_png_fallback("dir")
                        mock_photo.assert_called()

    def test_load_png_fallback_not_found(self, app):
        with patch("pathlib.Path.exists", return_value=False):
            app._load_png_fallback("dir")

    def test_create_scrollable_interface(self, app):
        # Just ensure it executes without crashing
        app.create_scrollable_interface()
        assert hasattr(app, "source_frame")
        assert hasattr(app, "run_button")

    def test_on_mode_change(self, app):
        app.create_scrollable_interface()
        app.operation_mode._val = "analyze"
        app.on_mode_change()
        assert app.mode_description.cget("text").startswith("Analyzes folder contents")

    def test_select_source_folders_success(self, app):
        app.create_scrollable_interface()
        with patch("tkinter.filedialog.askdirectory", return_value="/fake/dir"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("os.access", return_value=True):
                    app.select_source_folders()
                    assert "/fake/dir" in app.source_folders

    def test_select_source_folders_exists_error(self, app):
        app.create_scrollable_interface()
        with patch("tkinter.filedialog.askdirectory", return_value="/fake/dir"):
            with patch("pathlib.Path.exists", return_value=False):
                with patch("tkinter.messagebox.showerror") as mock_err:
                    app.select_source_folders()
                    mock_err.assert_called_once()
                    assert "/fake/dir" not in app.source_folders

    def test_select_source_folders_access_error(self, app):
        app.create_scrollable_interface()
        with patch("tkinter.filedialog.askdirectory", return_value="/fake/dir"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("os.access", return_value=False):
                    with patch("tkinter.messagebox.showerror") as mock_err:
                        app.select_source_folders()
                        mock_err.assert_called_once()
                        assert "/fake/dir" not in app.source_folders

    def test_select_source_folders_already_in_list(self, app):
        app.create_scrollable_interface()
        app.source_folders = ["/fake/dir"]
        with patch("tkinter.filedialog.askdirectory", return_value="/fake/dir"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("os.access", return_value=True):
                    with patch("tkinter.messagebox.showinfo") as mock_info:
                        app.select_source_folders()
                        mock_info.assert_called_once()

    def test_remove_selected_source_empty(self, app):
        app.create_scrollable_interface()
        with patch("tkinter.messagebox.showinfo") as mock_info:
            app.remove_selected_source()
            mock_info.assert_called_once()

    def test_remove_selected_source_single(self, app):
        app.create_scrollable_interface()
        app.source_folders = ["/fix/a", "/fix/b"]
        app.source_listbox.insert(tk.END, "/fix/a")
        app.source_listbox.insert(tk.END, "/fix/b")
        app.source_listbox.selection_set(0)

        with patch("tkinter.messagebox.askyesno", return_value=True):
            app.remove_selected_source()
            assert "/fix/a" not in app.source_folders

    def test_remove_selected_source_multiple(self, app):
        app.create_scrollable_interface()
        app.source_folders = ["/fix/a", "/fix/b"]
        app.source_listbox.insert(tk.END, "/fix/a")
        app.source_listbox.insert(tk.END, "/fix/b")
        app.source_listbox.selection_set(0)
        app.source_listbox.selection_set(1)

        with patch("tkinter.messagebox.askyesno", return_value=True):
            app.remove_selected_source()
            assert len(app.source_folders) == 0

    def test_select_dest_folder_success(self, app):
        app.create_scrollable_interface()
        with patch("tkinter.filedialog.askdirectory", return_value="/fake/dest"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("os.access", return_value=True):
                    app.select_dest_folder()
                    assert app.dest_folder == "/fake/dest"

    def test_select_dest_folder_exists_error(self, app):
        app.create_scrollable_interface()
        with patch("tkinter.filedialog.askdirectory", return_value="/fake/dest"):
            with patch("pathlib.Path.exists", return_value=False):
                with patch("tkinter.messagebox.showerror") as mock_err:
                    app.select_dest_folder()
                    mock_err.assert_called_once()
                    assert app.dest_folder == ""

    def test_select_dest_folder_access_error(self, app):
        app.create_scrollable_interface()
        with patch("tkinter.filedialog.askdirectory", return_value="/fake/dest"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("os.access", return_value=False):
                    with patch("tkinter.messagebox.showerror") as mock_err:
                        app.select_dest_folder()
                        mock_err.assert_called_once()
                        assert app.dest_folder == ""

    def test_mousewheel_scroll(self, app):
        app.create_scrollable_interface()
        # Find the canvas bound to mousewheel
        canvas = None
        for child in app.root.winfo_children():
            if isinstance(child, tk.Canvas):
                canvas = child
                break

        event = MagicMock()
        event.delta = 120
        # Call the bound function
        canvas.event_generate("<MouseWheel>", delta=120)

        # Or mock the event directly to the handler we bound
        # Since it's a bound event, it's easier to just call it on the callback
        # But for coverage we can mock the entire UI
