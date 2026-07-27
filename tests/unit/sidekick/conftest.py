"""Local conftest making the shared Sidekick package importable."""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_PYTHON = REPO_ROOT / "src" / "shared" / "python"
if str(SHARED_PYTHON) not in sys.path:
    sys.path.insert(0, str(SHARED_PYTHON))


class _ProfileStore:
    """Minimal profile store used to isolate the standalone window."""

    def __init__(self) -> None:
        self.profiles: dict[str, Any] = {}
        self.last: str | None = None

    def save_profile(self, name: str, payload: Any) -> None:
        self.profiles[name] = payload

    def load_profile(self, name: str) -> Any:
        if name not in self.profiles:
            raise KeyError(name)
        return self.profiles[name]

    def set_last_profile(self, name: str) -> None:
        self.last = name


@pytest.fixture
def profile_store() -> _ProfileStore:
    """Return an isolated profile store for standalone window tests."""
    return _ProfileStore()


@pytest.fixture(scope="session")
def qt_app() -> Any:
    """Return a session-scoped QApplication when PyQt6 is available."""
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:  # pragma: no cover - depends on local Qt install
        pytest.skip("Qt widgets unavailable")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


@pytest.fixture
def qtbot(qt_app: Any) -> Generator[Any, None, None]:
    """Provide a mock qtbot fixture that cleans up registered widgets."""

    class MockQtBot:
        def __init__(self) -> None:
            self._widgets: list[Any] = []

        def addWidget(self, widget: Any) -> None:
            self._widgets.append(widget)

        def waitUntil(self, callback: Any, timeout: int = 1000) -> None:
            import time

            try:
                from upstream_drift_tools.ui.tools_sidebar.qt_compat import (
                    QtWidgets,
                )

                deadline = time.time() + timeout / 1000.0
                while time.time() < deadline:
                    if callback():
                        return
                    QtWidgets.QApplication.processEvents()
                    time.sleep(0.01)
            except Exception:
                pass
            if not callback():
                raise TimeoutError("waitUntil condition not met within timeout")

    bot = MockQtBot()
    yield bot
    for w in bot._widgets:
        try:
            w.close()
            w.deleteLater()
        except Exception:
            pass
