# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""P&ID Generator — PyQt6 main window.

Provides a file-picker GUI that selects a YAML spec file and output
location, then delegates generation to the ``programmatic_pid`` library.
"""

from __future__ import annotations

import logging
from pathlib import Path

from programmatic_pid import PIDDocument
from programmatic_pid.profiles import PROFILE_PRESETS
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class PIDGeneratorMainWindow(QMainWindow):
    """Main window for the P&ID Generator tool."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("P&ID Generator")
        self.setMinimumSize(800, 400)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        layout.addWidget(QLabel("<h2>P&ID Generator</h2>"))
        layout.addWidget(QLabel("Generate P&ID drawings from YAML specifications."))

        # Spec file row
        spec_row = QHBoxLayout()
        spec_label = QLabel("Spec YAML:")
        spec_label.setFixedWidth(100)
        self._spec_edit = QLineEdit()
        self._spec_edit.setPlaceholderText("Path to spec.yml ...")
        spec_browse = QPushButton("Browse\u2026")
        spec_row.addWidget(spec_label)
        spec_row.addWidget(self._spec_edit)
        spec_row.addWidget(spec_browse)
        layout.addLayout(spec_row)

        # Output DXF row
        out_row = QHBoxLayout()
        out_label = QLabel("Output DXF:")
        out_label.setFixedWidth(100)
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Path to output.dxf ...")
        out_browse = QPushButton("Browse\u2026")
        out_row.addWidget(out_label)
        out_row.addWidget(self._out_edit)
        out_row.addWidget(out_browse)
        layout.addLayout(out_row)

        # Profile row
        profile_row = QHBoxLayout()
        profile_label = QLabel("Profile:")
        profile_label.setFixedWidth(100)
        self._profile_combo = QComboBox()
        self._profile_combo.addItem("(default)")
        for profile_name in PROFILE_PRESETS:
            self._profile_combo.addItem(profile_name)
        profile_row.addWidget(profile_label)
        profile_row.addWidget(self._profile_combo)
        profile_row.addStretch()
        layout.addLayout(profile_row)

        layout.addStretch()

        generate_btn = QPushButton("Generate P&ID")
        generate_btn.setFixedHeight(40)
        layout.addWidget(generate_btn)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # Connections
        spec_browse.clicked.connect(self._browse_spec)
        out_browse.clicked.connect(self._browse_out)
        generate_btn.clicked.connect(self._generate)

    def _browse_spec(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Spec YAML", "", "YAML Files (*.yml *.yaml)"
        )
        if path:
            self._spec_edit.setText(path)
            if not self._out_edit.text():
                self._out_edit.setText(str(Path(path).with_suffix(".dxf")))

    def _browse_out(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save DXF Output", "", "DXF Files (*.dxf)"
        )
        if path:
            self._out_edit.setText(path)

    def _generate(self) -> None:
        spec_path = self._spec_edit.text().strip()
        out_path = self._out_edit.text().strip()
        profile_text = self._profile_combo.currentText()
        profile = None if profile_text == "(default)" else profile_text

        if not spec_path:
            QMessageBox.warning(
                self, "Missing Input", "Please select a spec YAML file."
            )
            return
        if not out_path:
            QMessageBox.warning(
                self, "Missing Output", "Please specify an output DXF path."
            )
            return

        self._status_label.setText("Generating\u2026")
        QApplication.processEvents()
        try:
            doc = PIDDocument.from_yaml(spec_path, profile=profile)
            doc.export_dxf(Path(out_path))
            svg_path = Path(out_path).with_suffix(".svg")
            doc.export_svg(svg_path)
            self._status_label.setText(
                f"\u2713 Generated: {Path(out_path).name}  +  {svg_path.name}"
            )
            QMessageBox.information(
                self,
                "Done",
                f"Generated:\n  {out_path}\n  {svg_path}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Generation failed")
            self._status_label.setText(f"\u2717 Error: {exc}")
            QMessageBox.critical(self, "Generation Failed", str(exc))
