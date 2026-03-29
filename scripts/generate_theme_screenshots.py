#!/usr/bin/env python3
"""Generate preview screenshots for all built-in themes.

Uses QT_QPA_PLATFORM=offscreen so no display server is required.
Output PNG files are written to docs/theme-previews/.

See issue #552.

Usage:
    QT_QPA_PLATFORM=offscreen python scripts/generate_theme_screenshots.py
    # On Windows (set env var first):
    set QT_QPA_PLATFORM=offscreen
    python scripts/generate_theme_screenshots.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure offscreen rendering is set before any Qt imports
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Bootstrap imports — use the sanctioned _bootstrap module
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from _bootstrap import bootstrap  # noqa: E402

bootstrap(__file__)

from PyQt6.QtCore import QSize  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shared.python.theme.colors import BUILTIN_THEMES  # noqa: E402
from shared.python.theme.stylesheets import generate_stylesheet  # noqa: E402


def _build_sample_widget() -> QMainWindow:
    """Build a representative sample window with common Qt widgets."""
    window = QMainWindow()
    window.setWindowTitle("Theme Preview")
    window.setFixedSize(QSize(640, 480))

    central = QWidget()
    window.setCentralWidget(central)
    layout = QVBoxLayout(central)

    # Title
    title = QLabel("Theme Preview")
    title.setStyleSheet("font-size: 18px; font-weight: bold;")
    layout.addWidget(title)

    # Group box with form inputs
    group = QGroupBox("Sample Inputs")
    group_layout = QVBoxLayout(group)

    row1 = QHBoxLayout()
    row1.addWidget(QLabel("Name:"))
    row1.addWidget(QLineEdit("John Doe"))
    group_layout.addLayout(row1)

    row2 = QHBoxLayout()
    row2.addWidget(QLabel("Option:"))
    combo = QComboBox()
    combo.addItems(["Option A", "Option B", "Option C"])
    row2.addWidget(combo)
    group_layout.addLayout(row2)

    layout.addWidget(group)

    # Buttons
    btn_row = QHBoxLayout()
    for label in ("Calculate", "Reset", "Export"):
        btn = QPushButton(label)
        btn_row.addWidget(btn)
    layout.addLayout(btn_row)

    # Table
    table = QTableWidget(4, 3)
    table.setHorizontalHeaderLabels(["Parameter", "Value", "Unit"])
    sample_data = [
        ("Temperature", "450.0", "K"),
        ("Pressure", "101.3", "kPa"),
        ("Flow Rate", "2.5", "m3/s"),
        ("Efficiency", "0.87", "-"),
    ]
    for row, (param, val, unit) in enumerate(sample_data):
        table.setItem(row, 0, QTableWidgetItem(param))
        table.setItem(row, 1, QTableWidgetItem(val))
        table.setItem(row, 2, QTableWidgetItem(unit))
    layout.addWidget(table)

    return window


def generate_screenshots(output_dir: Path | None = None) -> list[Path]:
    """Generate theme preview PNGs for all built-in themes.

    Args:
        output_dir: Directory to write PNGs into. Defaults to
            ``docs/theme-previews/`` relative to the repo root.

    Returns:
        List of paths to the generated PNG files.
    """
    if output_dir is None:
        output_dir = _REPO_ROOT / "docs" / "theme-previews"
    output_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv)
    generated: list[Path] = []

    for theme_name, theme_colors in BUILTIN_THEMES.items():
        window = _build_sample_widget()
        stylesheet = generate_stylesheet(theme_colors)
        window.setStyleSheet(stylesheet)
        window.show()

        # Force layout/paint
        app.processEvents()

        pixmap = window.grab()
        safe_name = theme_name.lower().replace(" ", "_")
        out_path = output_dir / f"{safe_name}.png"
        pixmap.save(str(out_path))
        generated.append(out_path)
        print(f"  Saved: {out_path.name}")

        window.close()

    return generated


def main() -> int:
    """Entry point."""
    print(f"Generating theme previews for {len(BUILTIN_THEMES)} themes...")
    paths = generate_screenshots()
    print(f"\nDone. {len(paths)} screenshots saved to {paths[0].parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
