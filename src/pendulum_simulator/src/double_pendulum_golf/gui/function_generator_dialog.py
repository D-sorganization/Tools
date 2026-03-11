"""
FunctionGeneratorDialog — wraps the shared Function Generator widget
(src/function_generator/…) inside a modal dialog so users can design
a torque profile waveform and import it as polynomial coefficients
directly into the pendulum controls.

Integration points
------------------
- The Function Generator's ``signal_generated`` signal fires whenever
  the waveform changes; we live-fit a polynomial to the signal values
  and show the fitted coefficients.
- "Import as Shoulder Torque" / "Import as Wrist Torque" buttons emit
  ``torque_imported(joint, coeffs)`` back to the parent ControlsWidget.
- The dialog tries to import FunctionGeneratorWidget from the sibling
  function_generator package; if not installed it falls back to a plain
  polynomial-coefficient entry dialog.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# ── Import shared style constants ─────────────────────────────────────────────
from .controls_utils import STYLE_LABEL, STYLE_EDIT, STYLE_BTN, STYLE_BTN_IMPORT

# ── Try to import the shared Function Generator ───────────────────────────────
_FUNCGEN_AVAILABLE = False
_FUNCGEN_IMPORT_ERROR: str | None = None
FunctionGeneratorWidget = None


def _find_sibling_package(marker_path: str) -> Path | None:
    """Walk up from this file to find a sibling package directory.

    Searches up to 10 parent levels for the given relative path.
    Returns the parent directory containing the marker, or None.

    Design by Contract
    ------------------
    Pre:  marker_path is a non-empty relative path string.
    Post: returns a valid directory Path or None.
    """
    assert marker_path, "marker_path must be non-empty"
    p = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = p / marker_path
        if candidate.exists():
            return p
        p = p.parent
    return None


try:
    import logging as _logging
    import os as _os

    _fg_logger = _logging.getLogger(__name__)

    _src_root = _find_sibling_package("function_generator/python")
    if _src_root is not None:
        _fg_root = _src_root / "function_generator" / "python"
        _shared_root = _src_root / "shared" / "python"

        # Normalize all paths to avoid Windows forward/back-slash mismatches
        _norm_paths: list[str] = [_os.path.normpath(str(p)) for p in sys.path]

        # Add all required paths BEFORE attempting the import:
        # 1. shared/python — for signal_toolkit and safe_eval
        # 2. src root — for 'shared.python.safe_eval' namespace resolution
        # 3. function_generator/python — for the function_generator package
        for _p in [_shared_root, _src_root, _fg_root]:
            _np = _os.path.normpath(str(_p))
            if _np not in _norm_paths:
                sys.path.insert(0, str(_p))
                _norm_paths.insert(0, _np)

        # Clean any partially-cached failed imports so we get a fresh attempt
        for _mod in list(sys.modules):
            if _mod.startswith("function_generator"):
                del sys.modules[_mod]

        from function_generator.ui.pyqt6.main_window import (
            FunctionGeneratorWidget as _FGWidget,
        )

        FunctionGeneratorWidget = _FGWidget
        _FUNCGEN_AVAILABLE = True
        _fg_logger.info("Function Generator package loaded from %s", _fg_root)
    else:
        _FUNCGEN_IMPORT_ERROR = (
            "function_generator/python directory not found in parent hierarchy"
        )
        _fg_logger.warning("Function Generator: %s", _FUNCGEN_IMPORT_ERROR)
except ImportError as _exc:
    _FUNCGEN_IMPORT_ERROR = str(_exc)
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "Function Generator import failed: %s", _FUNCGEN_IMPORT_ERROR
    )


class _FallbackPolyWidget(QWidget):
    """Minimal fallback when the function_generator package is not installed."""

    signal_generated = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        error_detail = _FUNCGEN_IMPORT_ERROR or "Unknown reason"
        note = QLabel(
            f"⚠ Function Generator package not available.\n\n"
            f"Reason: {error_detail}\n\n"
            "Ensure the function_generator package is on the Python path.\n\n"
            "Enter polynomial coefficients directly below (c0, c1, c2 …):",
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #e0a060; padding: 8px; font-size: 11px;")
        layout.addWidget(note)

        coeff_lbl = QLabel("Coefficients (comma-separated):")
        coeff_lbl.setStyleSheet(STYLE_LABEL)
        layout.addWidget(coeff_lbl)

        self.coeff_edit = QLineEdit("0, 0")
        self.coeff_edit.setStyleSheet(STYLE_EDIT)
        layout.addWidget(self.coeff_edit)

        layout.addStretch()

    def get_polynomial_coefficients(self) -> list[float]:
        parts = self.coeff_edit.text().split(",")
        coeffs: list[float] = []
        for p in parts:
            p = p.strip()
            if p:
                try:
                    coeffs.append(float(p))
                except ValueError:
                    pass
        return coeffs or [0.0]


class FunctionGeneratorDialog(QDialog):
    """Modal dialog hosting the Function Generator widget.

    After designing a waveform the user clicks one of:
    'Import as Shoulder Torque', 'Import as Elbow Torque', or 'Import as Wrist Torque'.
    The dialog fits a polynomial of selected order to the signal
    and emits ``torque_imported(joint_name, coefficients_list)``.

    Signals
    -------
    torque_imported(str, list[float])
        Joint name (``"shoulder"``, ``"elbow"``, or ``"wrist"``) and the fitted poly coefficients.
    """

    torque_imported = pyqtSignal(str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Function Generator — Torque Profile Designer")
        self.setModal(True)
        self.resize(1050, 660)
        self.setMinimumSize(750, 500)
        self.setStyleSheet(
            "QDialog { background: #12121c; color: #d0d0e8; }"
            "QGroupBox { color: #c0c0e0; border: 1px solid #3a3a58;"
            "border-radius: 6px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
            "QLabel { color: #b0b0cc; }",
        )

        self._current_signal: object = None
        self._fitted_coeffs: list[float] = [0.0]

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 8)
        main_layout.setSpacing(8)

        # Header
        hdr = QLabel(
            "Design a torque waveform and import it as polynomial coefficients",
        )
        hdr.setFont(QFont("Sans", 11))
        hdr.setStyleSheet("color: #9090b8; padding: 2px 0;")
        main_layout.addWidget(hdr)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Function generator widget ─────────────────────────────────
        if _FUNCGEN_AVAILABLE and FunctionGeneratorWidget is not None:
            self._fg_widget = FunctionGeneratorWidget(use_builtin_theme=False)
            self._fg_widget.signal_generated.connect(self._on_signal_generated)
        else:
            self._fg_widget = _FallbackPolyWidget()

        splitter.addWidget(self._fg_widget)

        # ── Polynomial fit panel ──────────────────────────────────────
        fit_container = QWidget()
        fit_layout = QVBoxLayout(fit_container)
        fit_layout.setContentsMargins(0, 0, 0, 0)
        fit_layout.setSpacing(6)

        fit_group = QGroupBox("Polynomial Fit to Waveform")
        fit_row = QHBoxLayout(fit_group)
        fit_row.setContentsMargins(8, 14, 8, 8)
        fit_row.setSpacing(10)

        order_lbl = QLabel("Poly order:")
        order_lbl.setStyleSheet(STYLE_LABEL)
        fit_row.addWidget(order_lbl)

        self._order_spin = QSpinBox()
        self._order_spin.setRange(0, 10)
        self._order_spin.setValue(3)
        self._order_spin.setFixedWidth(50)
        self._order_spin.setToolTip(
            "Polynomial order for fitting.\n0 = constant, 1 = linear, … 3+ = curved.",
        )
        self._order_spin.setStyleSheet(
            "background: #1e1e30; color: #e0e0f0; border: 1px solid #404060;"
            "border-radius: 3px; padding: 2px 4px;",
        )
        self._order_spin.valueChanged.connect(self._refit)
        fit_row.addWidget(self._order_spin)

        fit_row.addWidget(QLabel("Coefficients (c0, c1, c2 …):"))
        self._coeff_display = QLineEdit()
        self._coeff_display.setReadOnly(True)
        self._coeff_display.setPlaceholderText("Generate a waveform to fit…")
        self._coeff_display.setStyleSheet(STYLE_EDIT)
        fit_row.addWidget(self._coeff_display, stretch=1)

        self._btn_refit = QPushButton("↺ Re-fit")
        self._btn_refit.setStyleSheet(STYLE_BTN)
        self._btn_refit.clicked.connect(self._refit)
        fit_row.addWidget(self._btn_refit)

        fit_layout.addWidget(fit_group)

        # ── Import buttons ────────────────────────────────────────────
        import_row = QHBoxLayout()
        import_row.setSpacing(8)

        self._btn_import_shoulder = QPushButton("📥 Import → Shoulder Torque")
        self._btn_import_shoulder.setStyleSheet(STYLE_BTN_IMPORT)
        self._btn_import_shoulder.setToolTip(
            "Fit the displayed waveform to a polynomial and copy\n"
            "the coefficients to the Shoulder torque input.",
        )
        self._btn_import_shoulder.clicked.connect(lambda: self._import("shoulder"))

        self._btn_import_elbow = QPushButton("📥 Import → Elbow Torque")
        self._btn_import_elbow.setStyleSheet(STYLE_BTN_IMPORT)
        self._btn_import_elbow.setToolTip(
            "Fit the displayed waveform to a polynomial and copy\n"
            "the coefficients to the Elbow torque input.",
        )
        self._btn_import_elbow.clicked.connect(lambda: self._import("elbow"))

        self._btn_import_wrist = QPushButton("📥 Import → Wrist Torque")
        self._btn_import_wrist.setStyleSheet(STYLE_BTN_IMPORT)
        self._btn_import_wrist.setToolTip(
            "Fit the displayed waveform to a polynomial and copy\n"
            "the coefficients to the Wrist torque input.",
        )
        self._btn_import_wrist.clicked.connect(lambda: self._import("wrist"))

        import_row.addStretch()
        import_row.addWidget(self._btn_import_shoulder)
        import_row.addWidget(self._btn_import_elbow)
        import_row.addWidget(self._btn_import_wrist)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        import_row.addWidget(btn_box)
        fit_layout.addLayout(import_row)

        splitter.addWidget(fit_container)
        splitter.setSizes([430, 160])
        main_layout.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _on_signal_generated(self, signal: object) -> None:
        """Receive a generated Signal object and auto-refit."""
        self._current_signal = signal
        self._refit()

    def _refit(self) -> None:
        """Fit a polynomial to the current signal."""
        if self._current_signal is None:
            if isinstance(self._fg_widget, _FallbackPolyWidget):
                self._fitted_coeffs = self._fg_widget.get_polynomial_coefficients()
                self._coeff_display.setText(
                    ", ".join(f"{c:.4g}" for c in self._fitted_coeffs),
                )
            return

        sig: object = self._current_signal
        order = self._order_spin.value()

        try:
            from typing import Any

            sig_any: Any = sig
            t = np.asarray(sig_any.time, dtype=np.float64)
            y = np.asarray(sig_any.values, dtype=np.float64)
            if len(t) < order + 1:
                self._coeff_display.setText("Not enough samples for this order")
                return
            raw_coeffs = np.polyfit(t, y, order)  # highest power first
            # Reverse so c0 = constant term (matches our polynomial convention)
            self._fitted_coeffs = list(raw_coeffs[::-1])
            self._coeff_display.setText(
                ", ".join(f"{c:.4g}" for c in self._fitted_coeffs),
            )
        except Exception as exc:
            self._coeff_display.setText(f"Fit error: {exc}")

    def _import(self, joint: str) -> None:
        """Emit the fitted coefficients for the specified joint and close."""
        if not self._fitted_coeffs:
            return
        self.torque_imported.emit(joint, self._fitted_coeffs)
        self.accept()
