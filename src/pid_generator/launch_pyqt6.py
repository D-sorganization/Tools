#!/usr/bin/env python3
"""P&ID Generator — PyQt6 launcher for the Tools monorepo.

Provides a minimal file-picker GUI that selects a YAML spec file and output
location, then delegates generation to the programmatic_pid library.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """Launch the P&ID Generator GUI."""
    try:
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
    except ImportError:
        print("PyQt6 not installed. Run: pip install PyQt6", file=sys.stderr)
        sys.exit(1)

    try:
        from programmatic_pid import PIDDocument
        from programmatic_pid.profiles import PROFILE_PRESETS
    except ImportError:
        print(
            "programmatic_pid not installed. Run: pip install -e '.[pid]'",
            file=sys.stderr,
        )
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("P&ID Generator")

    window = QMainWindow()
    window.setWindowTitle("P&ID Generator")
    window.setMinimumSize(800, 400)

    central = QWidget()
    window.setCentralWidget(central)
    layout = QVBoxLayout(central)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(12)

    layout.addWidget(QLabel("<h2>P&ID Generator</h2>"))
    layout.addWidget(QLabel("Generate P&ID drawings from YAML specifications."))

    # Spec file row
    spec_row = QHBoxLayout()
    spec_label = QLabel("Spec YAML:")
    spec_label.setFixedWidth(100)
    spec_edit = QLineEdit()
    spec_edit.setPlaceholderText("Path to spec.yml ...")
    spec_browse = QPushButton("Browse…")
    spec_row.addWidget(spec_label)
    spec_row.addWidget(spec_edit)
    spec_row.addWidget(spec_browse)
    layout.addLayout(spec_row)

    # Output DXF row
    out_row = QHBoxLayout()
    out_label = QLabel("Output DXF:")
    out_label.setFixedWidth(100)
    out_edit = QLineEdit()
    out_edit.setPlaceholderText("Path to output.dxf ...")
    out_browse = QPushButton("Browse…")
    out_row.addWidget(out_label)
    out_row.addWidget(out_edit)
    out_row.addWidget(out_browse)
    layout.addLayout(out_row)

    # Profile row
    profile_row = QHBoxLayout()
    profile_label = QLabel("Profile:")
    profile_label.setFixedWidth(100)
    profile_combo = QComboBox()
    profile_combo.addItem("(default)")
    for profile_name in PROFILE_PRESETS:
        profile_combo.addItem(profile_name)
    profile_row.addWidget(profile_label)
    profile_row.addWidget(profile_combo)
    profile_row.addStretch()
    layout.addLayout(profile_row)

    layout.addStretch()

    generate_btn = QPushButton("Generate P&ID")
    generate_btn.setFixedHeight(40)
    layout.addWidget(generate_btn)

    status_label = QLabel("")
    layout.addWidget(status_label)

    def browse_spec() -> None:
        path, _ = QFileDialog.getOpenFileName(
            window, "Select Spec YAML", "", "YAML Files (*.yml *.yaml)"
        )
        if path:
            spec_edit.setText(path)
            if not out_edit.text():
                out_edit.setText(str(Path(path).with_suffix(".dxf")))

    def browse_out() -> None:
        path, _ = QFileDialog.getSaveFileName(
            window, "Save DXF Output", "", "DXF Files (*.dxf)"
        )
        if path:
            out_edit.setText(path)

    def generate() -> None:
        spec_path = spec_edit.text().strip()
        out_path = out_edit.text().strip()
        profile_text = profile_combo.currentText()
        profile = None if profile_text == "(default)" else profile_text

        if not spec_path:
            QMessageBox.warning(
                window, "Missing Input", "Please select a spec YAML file."
            )
            return
        if not out_path:
            QMessageBox.warning(
                window, "Missing Output", "Please specify an output DXF path."
            )
            return

        status_label.setText("Generating…")
        app.processEvents()
        try:
            doc = PIDDocument.from_yaml(spec_path, profile=profile)
            doc.export_dxf(Path(out_path))
            svg_path = Path(out_path).with_suffix(".svg")
            doc.export_svg(svg_path)
            status_label.setText(
                f"✓ Generated: {Path(out_path).name}  +  {svg_path.name}"
            )
            QMessageBox.information(
                window,
                "Done",
                f"Generated:\n  {out_path}\n  {svg_path}",
            )
        except Exception as exc:
            logger.exception("Generation failed")
            status_label.setText(f"✗ Error: {exc}")
            QMessageBox.critical(window, "Generation Failed", str(exc))

    spec_browse.clicked.connect(browse_spec)
    out_browse.clicked.connect(browse_out)
    generate_btn.clicked.connect(generate)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
