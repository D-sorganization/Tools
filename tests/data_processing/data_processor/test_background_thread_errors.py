"""Regression tests for data-processor background thread failures."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from threading import Event
from unittest.mock import MagicMock, patch

from data_processor.ui.background_worker import start_background_thread

tkinter_stub = types.ModuleType("tkinter")
tkinter_stub.filedialog = types.SimpleNamespace()
tkinter_stub.messagebox = types.SimpleNamespace(
    showerror=MagicMock(),
    showwarning=MagicMock(),
    showinfo=MagicMock(),
)
sys.modules.setdefault("tkinter", tkinter_stub)


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

io_stub = types.ModuleType("upstream_drift_tools.data_processing.io")
io_stub.DataReader = object
io_stub.DataWriter = object
io_stub.FileFormatDetector = object
sys.modules.setdefault("upstream_drift_tools", types.ModuleType("upstream_drift_tools"))
sys.modules.setdefault(
    "upstream_drift_tools.data_processing",
    types.ModuleType("upstream_drift_tools.data_processing"),
)
sys.modules.setdefault("upstream_drift_tools.data_processing.io", io_stub)

from data_processor.ui.folder_tool_tab import FolderToolMixin
from data_processor.ui.format_converter_tab import FormatConverterMixin


class ImmediateAfterMixin:
    """Test double that runs UI callbacks immediately."""

    def after(self, ms: int, func: Callable[[], None]) -> None:
        assert ms == 0
        func()


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
