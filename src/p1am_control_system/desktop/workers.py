from collections.abc import Callable
from typing import Any
from weakref import ref

try:
    import requests as _requests_import
except ImportError:
    _requests: Any | None = None
else:
    _requests = _requests_import

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget

RequestTimeout = float | tuple[float, float]
ButtonRestore = bool | Callable[[bool], bool]


def _request_timeout(timeout: RequestTimeout) -> tuple[float, float]:
    """Return an explicit connect/read timeout tuple for requests."""
    if isinstance(timeout, tuple):
        return timeout
    return (min(0.5, timeout), timeout)


class HttpWorker(QThread):
    """
    Asynchronous worker for making HTTP requests without blocking the GUI thread.
    """

    success = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        method: str,
        url: str,
        data: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        timeout: RequestTimeout = 2.0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.method = method.upper()
        self.url = url
        self.data = data
        self.json = json
        self.timeout = _request_timeout(timeout)

    def run(self) -> None:
        if _requests is None:
            self.error.emit("requests dependency is not installed")
            return

        try:
            if self.method == "GET":
                resp = _requests.get(self.url, params=self.data, timeout=self.timeout)
            elif self.method == "POST":
                resp = _requests.post(
                    self.url, data=self.data, json=self.json, timeout=self.timeout
                )
            else:
                self.error.emit(f"Unsupported method: {self.method}")
                return

            resp.raise_for_status()

            try:
                data = resp.json()
            except ValueError:
                data = {"text": resp.text}

            self.success.emit(data)

        except _requests.exceptions.RequestException as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Unexpected error: {str(e)}")


def start_http_request(
    owner: QWidget,
    attr_name: str,
    worker: HttpWorker,
    *,
    busy_button: QPushButton | None = None,
    busy_text: str | None = None,
    restore_button: ButtonRestore = True,
) -> HttpWorker:
    """Start *worker* with standard HMI busy-state handling."""
    QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
    owner_ref = ref(owner)
    button_ref = ref(busy_button) if busy_button is not None else None
    was_enabled = busy_button.isEnabled() if busy_button is not None else False
    old_text = busy_button.text() if busy_button is not None else ""

    if busy_button is not None:
        count = int(busy_button.property("_p1am_http_busy_count") or 0)
        if count == 0:
            busy_button.setProperty("_p1am_http_old_text", old_text)
        busy_button.setProperty("_p1am_http_busy_count", count + 1)
        busy_button.setEnabled(False)
        if busy_text is not None:
            busy_button.setText(busy_text)

    def _restore() -> None:
        app = QApplication.instance()
        if app is not None and QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

        button = button_ref() if button_ref is not None else None
        if button is None:
            return
        try:
            count = int(button.property("_p1am_http_busy_count") or 1) - 1
            button.setProperty("_p1am_http_busy_count", max(0, count))
            if count > 0:
                return
            button.setText(str(button.property("_p1am_http_old_text") or old_text))
            enabled = (
                restore_button(was_enabled)
                if callable(restore_button)
                else was_enabled and restore_button
            )
            button.setEnabled(enabled)
        except RuntimeError:
            return

    worker.finished.connect(_restore)
    if owner_ref() is not None:
        setattr(owner, attr_name, worker)
    worker.start()
    return worker
