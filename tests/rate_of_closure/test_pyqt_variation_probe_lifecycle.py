"""Lifecycle contract for the isolated Variation rendered probe."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import sip
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget

from tests.rate_of_closure.pyqt_probe_lifecycle import shutdown_probe

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class _ProbeTab(QWidget):
    """Small worker-owner stand-in used to assert shutdown ordering."""

    def __init__(self) -> None:
        super().__init__()
        self.stopped = False

    def stop(self) -> None:
        """Record cooperative worker shutdown."""
        self.stopped = True


def test_shutdown_probe_stops_workers_and_deletes_window(qapp: QApplication) -> None:
    """Rendered probes must release Qt ownership before process exit."""
    window = QMainWindow()
    tab = _ProbeTab()
    window.setCentralWidget(tab)
    window.show()

    shutdown_probe(qapp, window, tab)

    assert tab.stopped
    assert sip.isdeleted(window)
