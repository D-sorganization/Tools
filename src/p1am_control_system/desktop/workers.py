from typing import Any

import requests
from PyQt6.QtCore import QThread, pyqtSignal


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
        timeout: float = 2.0,
    ) -> None:
        super().__init__()
        self.method = method.upper()
        self.url = url
        self.data = data
        self.json = json
        self.timeout = timeout

    def run(self) -> None:
        try:
            if self.method == "GET":
                resp = requests.get(self.url, params=self.data, timeout=self.timeout)
            elif self.method == "POST":
                resp = requests.post(
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

        except requests.exceptions.RequestException as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Unexpected error: {str(e)}")
