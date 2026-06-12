"""Regression tests for data-processor background thread failures."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from threading import Event
from unittest.mock import MagicMock, patch

from data_processor.ui.background_worker import start_background_thread

try:
    import tkinter as tkinter_stub
    from tkinter import filedialog, messagebox, ttk

    _USING_TKINTER_FALLBACK = False
except ImportError:
    tkinter_stub = types.ModuleType("tkinter")
    filedialog = types.ModuleType("tkinter.filedialog")
    messagebox = types.ModuleType("tkinter.messagebox")
    ttk = types.ModuleType("tkinter.ttk")
    _USING_TKINTER_FALLBACK = True

    class _TkWidget:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            pass

    for _widget_name in (
        "Button",
        "Checkbutton",
        "Entry",
        "Frame",
        "Label",
        "LabelFrame",
        "Progressbar",
        "Radiobutton",
        "Scrollbar",
        "Style",
    ):
        setattr(ttk, _widget_name, _TkWidget)

    tkinter_stub.TclError = RuntimeError
    tkinter_stub.Tk = _TkWidget
    tkinter_stub.Toplevel = _TkWidget
    tkinter_stub.Text = _TkWidget
    tkinter_stub.Canvas = _TkWidget
    tkinter_stub.Listbox = _TkWidget
    tkinter_stub.StringVar = _TkWidget
    tkinter_stub.BooleanVar = _TkWidget
    tkinter_stub.DoubleVar = _TkWidget
    tkinter_stub.END = "end"
    tkinter_stub.LEFT = "left"
    tkinter_stub.BOTH = "both"
    tkinter_stub.Y = "y"
    tkinter_stub.VERTICAL = "vertical"

if _USING_TKINTER_FALLBACK:
    messagebox.showerror = MagicMock()
    messagebox.showwarning = MagicMock()
    messagebox.showinfo = MagicMock()
    filedialog.askdirectory = MagicMock()
tkinter_stub.filedialog = filedialog
tkinter_stub.messagebox = messagebox
tkinter_stub.ttk = ttk
if _USING_TKINTER_FALLBACK:
    sys.modules["tkinter"] = tkinter_stub
    sys.modules["tkinter.filedialog"] = filedialog
    sys.modules["tkinter.messagebox"] = messagebox
    sys.modules["tkinter.ttk"] = ttk
else:
    sys.modules.setdefault("tkinter", tkinter_stub)
    sys.modules.setdefault("tkinter.filedialog", filedialog)
    sys.modules.setdefault("tkinter.messagebox", messagebox)
    sys.modules.setdefault("tkinter.ttk", ttk)


class _CtkWidget:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        pass


customtkinter_stub = types.ModuleType("customtkinter")
for _name in (
    "CTk",
    "CTkFrame",
    "CTkScrollableFrame",
    "CTkButton",
    "CTkLabel",
    "CTkTextbox",
    "CTkToplevel",
    "CTkCheckBox",
    "CTkOptionMenu",
    "CTkProgressBar",
    "StringVar",
    "BooleanVar",
    "CTkFont",
):
    setattr(customtkinter_stub, _name, _CtkWidget)
sys.modules.setdefault("customtkinter", customtkinter_stub)

numba_stub = types.ModuleType("numba")


def _jit(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
    def decorator(func):
        return func

    return decorator


numba_stub.jit = _jit
sys.modules.setdefault("numba", numba_stub)

from data_processor.ui.folder_tool_tab import FolderToolMixin
from data_processor.ui.format_converter_tab import FormatConverterMixin


class ImmediateAfterMixin:
    """Test double that runs UI callbacks immediately."""

    def after(self, ms: int, func: Callable[[], None]) -> None:
        assert ms == 0
        func()


def test_tkinter_test_double_keeps_ttk_importable() -> None:
    """Keep this module's optional tkinter shim compatible with later tests."""
    from tkinter import ttk as imported_ttk

    assert imported_ttk is not None


class DummyFolderTool(ImmediateAfterMixin, FolderToolMixin):
    def __init__(self) -> None:
        self.folder_status_var = MagicMock()
        self.folder_progress_bar = MagicMock()
        self.folder_run_button = MagicMock()
        self.folder_cancel_button = MagicMock()


class DummyFormatConverter(ImmediateAfterMixin, FormatConverterMixin):
    def __init__(self) -> None:
        self.converter_log_text = MagicMock()
        self.converter_status_label = MagicMock()
        self.converter_progress = MagicMock()
        self.converter_convert_button = MagicMock()


def test_background_thread_routes_unhandled_exception_to_ui_callback() -> None:
    owner = ImmediateAfterMixin()
    completed = Event()
    errors: list[tuple[BaseException, str]] = []

    def fail() -> None:
        raise RuntimeError("daemon failure")

    def on_error(exc: BaseException, traceback_text: str) -> None:
        errors.append((exc, traceback_text))
        completed.set()

    thread = start_background_thread(
        owner,
        fail,
        name="test-background-failure",
        on_error=on_error,
    )
    thread.join(timeout=1)

    assert completed.wait(timeout=1)
    assert isinstance(errors[0][0], RuntimeError)
    assert "daemon failure" in str(errors[0][0])
    assert "RuntimeError: daemon failure" in errors[0][1]


def test_folder_processing_error_resets_controls_and_warns_user() -> None:
    app = DummyFolderTool()

    with patch("data_processor.ui.folder_tool_tab.messagebox.showerror") as showerror:
        app._folder_handle_processing_error(RuntimeError("walk failed"), "traceback")

    app.folder_status_var.set.assert_called_once_with(
        "Folder processing failed: walk failed"
    )
    app.folder_progress_bar.set.assert_called_once_with(0)
    app.folder_run_button.configure.assert_called_once_with(state="normal")
    app.folder_cancel_button.configure.assert_called_once_with(state="disabled")
    showerror.assert_called_once_with(
        "Folder Processing Failed", "Folder processing failed: walk failed"
    )


def test_conversion_error_resets_controls_logs_and_warns_user() -> None:
    app = DummyFormatConverter()

    with patch(
        "data_processor.ui.format_converter_tab.messagebox.showerror"
    ) as showerror:
        app._handle_conversion_thread_error(RuntimeError("writer failed"), "traceback")

    app.converter_status_label.configure.assert_called_once_with(
        text="Conversion failed"
    )
    app.converter_progress.set.assert_called_once_with(0)
    app.converter_convert_button.configure.assert_called_once_with(state="normal")
    app.converter_log_text.insert.assert_called_once()
    inserted_message = app.converter_log_text.insert.call_args.args[1]
    assert "Conversion failed: writer failed" in inserted_message
    showerror.assert_called_once_with(
        "Conversion Failed", "Conversion failed: writer failed"
    )
