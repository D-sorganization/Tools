"""Deterministic Qt teardown shared by rendered subprocess probes."""

from __future__ import annotations

from typing import Protocol

from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication, QMainWindow


class WorkerOwner(Protocol):
    """Widget contract for cooperative background-worker shutdown."""

    def stop(self) -> None:
        """Stop and join any owned workers."""


def shutdown_probe(
    application: QApplication,
    window: QMainWindow,
    worker_owner: WorkerOwner,
) -> None:
    """Release worker and Qt ownership before the probe process returns."""
    worker_owner.stop()
    window.close()
    application.processEvents()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()
    application.quit()
