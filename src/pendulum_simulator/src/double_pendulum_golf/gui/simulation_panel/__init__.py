"""
Shared simulation panel for double and triple pendulum tabs.

This package was originally a single module ``simulation_panel.py``. The
implementation has been split into focused submodules:

- ``_worker``         — background ``_SimWorker`` and ``_SimViewer`` protocol
- ``_export_mixin``   — image / CSV / video export helpers
- ``_simulation_panel`` — the main ``SimulationPanel`` QWidget

The public API (``SimulationPanel``) is re-exported here so existing imports
``from .simulation_panel import SimulationPanel`` keep working.
"""

import subprocess
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ._simulation_panel import SimulationPanel
from ._worker import _SimViewer, _SimWorker

__all__ = [
    "SimulationPanel",
    "_SimViewer",
    "_SimWorker",
    "QMessageBox",
    "QFileDialog",
    "subprocess",
]
