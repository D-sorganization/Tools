"""
FunctionGeneratorDialog — wraps the shared Signal Toolkit widget
(signal_toolkit.widget.SignalToolkitWidget) inside a modal dialog so
users can design a torque profile waveform and import it as polynomial
coefficients directly into the pendulum controls.

Integration points
------------------
- The Signal Toolkit's ``signal_generated`` signal fires with
  ``(joint_name, coefficients)`` when the user clicks "Apply to Joint".
- The dialog maps the joint name back to the parent ControlsWidget
  via the ``torque_imported(joint, coeffs)`` signal.

Closes #1153: migrated from the legacy function_generator package to
the shared signal_toolkit widget from upstream_drift_tools.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ── Try to import the shared Signal Toolkit widget ────────────────────────
_WIDGET_AVAILABLE = False
_WIDGET_IMPORT_ERROR: str | None = None
_SignalToolkitWidget: type | None = None

try:
    import os
    import sys
    from pathlib import Path

    # signal_toolkit.widget_processing imports 'shared.python.safe_eval' which
    # needs the Tools/src/ root on sys.path.  Walk up to find it.
    _p = Path(__file__).resolve().parent
    for _ in range(10):
        if (_p / "shared" / "python").is_dir():
            _np = os.path.normpath(str(_p))
            if _np not in [os.path.normpath(s) for s in sys.path]:
                sys.path.insert(0, str(_p))
            break
        _p = _p.parent

    from signal_toolkit.widget import SignalToolkitWidget as _STWidget

    _SignalToolkitWidget = _STWidget
    _WIDGET_AVAILABLE = True
    logger.info("Signal Toolkit widget loaded successfully")
except ImportError as _exc:
    _WIDGET_IMPORT_ERROR = str(_exc)
    logger.warning("Signal Toolkit widget import failed: %s", _WIDGET_IMPORT_ERROR)


class FunctionGeneratorDialog(QDialog):
    """Modal dialog hosting the Signal Toolkit widget.

    After designing a waveform the user clicks "Apply to Joint" inside the
    Signal Toolkit widget.  The dialog captures the ``signal_generated``
    signal and re-emits it as ``torque_imported(joint_name, coefficients_list)``.

    Signals
    -------
    torque_imported(str, list[float])
        Joint name (``"shoulder"``, ``"elbow"``, or ``"wrist"``)
        and the fitted polynomial coefficients.
    """

    torque_imported = pyqtSignal(str, object)

    def __init__(
        self,
        parent: QWidget | None = None,
        joint_names: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Signal Toolkit — Torque Profile Designer")
        self.setModal(True)
        self.resize(1200, 800)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("QDialog { background: #12121c; color: #d0d0e8; }")

        self._joint_names = joint_names or ["Shoulder", "Wrist"]
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(8)

        # Header
        hdr = QLabel(
            "Design a torque waveform and apply it to a joint",
        )
        hdr.setFont(QFont("Sans", 11))
        hdr.setStyleSheet("color: #9090b8; padding: 2px 0;")
        layout.addWidget(hdr)

        if _WIDGET_AVAILABLE and _SignalToolkitWidget is not None:
            self._stk_widget = _SignalToolkitWidget(use_builtin_theme=False)
            self._stk_widget.set_joints(self._joint_names)
            self._stk_widget.signal_generated.connect(self._on_signal_applied)
            layout.addWidget(self._stk_widget, stretch=1)
        else:
            error_detail = _WIDGET_IMPORT_ERROR or "Unknown reason"
            note = QLabel(
                f"⚠ Signal Toolkit widget not available.\n\n"
                f"Reason: {error_detail}\n\n"
                "Ensure the signal_toolkit package is on the Python path.\n"
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: #e0a060; padding: 8px; font-size: 11px;")
            layout.addWidget(note)

        # Close button
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_signal_applied(self, joint_name: str, coefficients: list) -> None:
        """Handle the signal_generated emission from SignalToolkitWidget."""
        logger.info("Signal applied to %s: %s", joint_name, coefficients)
        # Normalize joint name to lowercase for consistency
        self.torque_imported.emit(joint_name.lower(), list(coefficients))
        self.accept()
