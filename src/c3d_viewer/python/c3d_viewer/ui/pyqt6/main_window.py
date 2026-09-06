# mypy: ignore-errors
# TRACKED_TASK: see #2310 — architecture debt extraction schedule

#!/usr/bin/env python3
"""C3D Motion Capture Viewer PyQt6 Main Window.

A PyQt6 GUI for viewing and analyzing C3D motion capture files.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shared.python.theme.catppuccin import CATPPUCCIN_MOCHA, get_stylesheet
from shared.python.theme.integration import ThemedWindowMixin


class C3DViewerWindow(ThemedWindowMixin, QMainWindow):
    """Main window for C3D Motion Capture Viewer application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self.setup_theme_support()
        self._current_file: Path | None = None
        self._metadata: dict | None = None
        self._notes_dock: Any | None = None
        self._setup_ui()

    # -- Notes integration (shared workspace) --
    def _toggle_notes(self) -> None:
        """Show/hide the shared notes dock widget."""
        try:
            from shared.python.notes.integration import attach_notes_dock
        except ImportError:
            return
        if self._notes_dock is None:
            project_dir = Path(__file__).resolve().parents[4]
            self._notes_dock = attach_notes_dock(self, project_dir=project_dir)
        self._notes_dock.setVisible(not self._notes_dock.isVisible())

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("C3D Motion Capture Viewer")
        self.setMinimumSize(750, 800)
        self.setStyleSheet(get_stylesheet())

        # Menu bar with Notes toggle
        menu_bar = self.menuBar()
        if menu_bar is not None:
            view_menu = menu_bar.addMenu("&View")
            if view_menu is not None:
                notes_action = view_menu.addAction("Toggle &Notes")
                if notes_action is not None:
                    notes_action.triggered.connect(self._toggle_notes)

        # Central widget with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCentralWidget(scroll_area)

        central_widget = QWidget()
        scroll_area.setWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Title
        title_label = QLabel("C3D Motion Capture Viewer")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['blue']};")
        main_layout.addWidget(title_label)

        # File loading
        main_layout.addWidget(self._create_file_group())

        # Tab widget for different views
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.tab_widget.addTab(self._create_metadata_tab(), "Metadata")
        self.tab_widget.addTab(self._create_markers_tab(), "Markers")
        self.tab_widget.addTab(self._create_analog_tab(), "Analog Channels")
        self.tab_widget.addTab(self._create_export_tab(), "Export")

        main_layout.addStretch()

    def _create_file_group(self) -> QGroupBox:
        """Create the file loading group."""
        group = QGroupBox("C3D File")
        layout = QGridLayout(group)
        layout.setSpacing(10)

        # File path display
        layout.addWidget(QLabel("File:"), 0, 0)
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['subtext0']};")
        layout.addWidget(self.file_label, 0, 1)

        # Load button
        load_btn = QPushButton("Load C3D File")
        load_btn.setObjectName("loadBtn")
        load_btn.clicked.connect(self._load_file)
        layout.addWidget(load_btn, 0, 2)

        return group

    def _create_metadata_tab(self) -> QWidget:
        """Create the metadata view tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # File info
        info_group = QGroupBox("Recording Information")
        info_layout = QGridLayout(info_group)
        info_layout.setSpacing(8)

        labels = [
            ("Markers:", "marker_count"),
            ("Frames:", "frame_count"),
            ("Frame Rate:", "frame_rate"),
            ("Duration:", "duration"),
            ("Units:", "units"),
            ("Analog Channels:", "analog_count"),
            ("Analog Rate:", "analog_rate"),
            ("Events:", "event_count"),
        ]

        self.info_labels: dict[str, QLabel] = {}
        for row, (label_text, key) in enumerate(labels):
            info_layout.addWidget(QLabel(label_text), row, 0)
            value_label = QLabel("-")
            value_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['sapphire']};")
            self.info_labels[key] = value_label
            info_layout.addWidget(value_label, row, 1)

        layout.addWidget(info_group)

        # Events list
        events_group = QGroupBox("Events")
        events_layout = QVBoxLayout(events_group)
        self.events_list = QListWidget()
        self.events_list.setMaximumHeight(150)
        events_layout.addWidget(self.events_list)
        layout.addWidget(events_group)

        layout.addStretch()
        return tab

    def _create_markers_tab(self) -> QWidget:
        """Create the markers view tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Marker list
        markers_group = QGroupBox("Marker Labels")
        markers_layout = QVBoxLayout(markers_group)
        self.marker_list = QListWidget()
        self.marker_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        markers_layout.addWidget(self.marker_list)
        layout.addWidget(markers_group)

        # Trajectory preview
        preview_group = QGroupBox("Trajectory Statistics")
        preview_layout = QVBoxLayout(preview_group)
        self.trajectory_text = QTextEdit()
        self.trajectory_text.setReadOnly(True)
        self.trajectory_text.setMaximumHeight(150)
        self.trajectory_text.setPlaceholderText(
            "Select markers to view trajectory statistics..."
        )
        preview_layout.addWidget(self.trajectory_text)

        preview_btn = QPushButton("Analyze Selected Markers")
        preview_btn.clicked.connect(self._analyze_markers)
        preview_layout.addWidget(preview_btn)

        layout.addWidget(preview_group)

        layout.addStretch()
        return tab

    def _create_analog_tab(self) -> QWidget:
        """Create the analog channels view tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Analog channels table
        analog_group = QGroupBox("Analog Channels")
        analog_layout = QVBoxLayout(analog_group)

        self.analog_table = QTableWidget()
        self.analog_table.setColumnCount(3)
        self.analog_table.setHorizontalHeaderLabels(["Channel", "Unit", "Samples"])
        header = self.analog_table.horizontalHeader()
        if header:
            header.setStretchLastSection(True)
        analog_layout.addWidget(self.analog_table)

        layout.addWidget(analog_group)

        # Force plates
        force_group = QGroupBox("Force Plates")
        force_layout = QVBoxLayout(force_group)
        self.force_text = QTextEdit()
        self.force_text.setReadOnly(True)
        self.force_text.setMaximumHeight(150)
        self.force_text.setPlaceholderText("Force plate data will appear here...")
        force_layout.addWidget(self.force_text)

        analyze_force_btn = QPushButton("Analyze Force Plates")
        analyze_force_btn.clicked.connect(self._analyze_force_plates)
        force_layout.addWidget(analyze_force_btn)

        layout.addWidget(force_group)

        layout.addStretch()
        return tab

    def _create_export_tab(self) -> QWidget:
        """Create the export options tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Export options
        options_group = QGroupBox("Export Options")
        options_layout = QGridLayout(options_group)
        options_layout.setSpacing(10)

        options_layout.addWidget(QLabel("Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["CSV", "JSON", "NPZ"])
        options_layout.addWidget(self.format_combo, 0, 1)

        options_layout.addWidget(QLabel("Target Units:"), 1, 0)
        self.units_combo = QComboBox()
        self.units_combo.addItems(["Original", "m", "mm", "cm"])
        options_layout.addWidget(self.units_combo, 1, 1)

        options_layout.addWidget(QLabel("Start Frame:"), 2, 0)
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, 999999)
        options_layout.addWidget(self.start_frame_spin, 2, 1)

        options_layout.addWidget(QLabel("End Frame:"), 3, 0)
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(0, 999999)
        self.end_frame_spin.setValue(999999)
        options_layout.addWidget(self.end_frame_spin, 3, 1)

        layout.addWidget(options_group)

        # Export buttons
        export_group = QGroupBox("Export Data")
        export_layout = QGridLayout(export_group)
        export_layout.setSpacing(10)

        export_points_btn = QPushButton("Export Marker Data")
        export_points_btn.clicked.connect(self._export_points)
        export_layout.addWidget(export_points_btn, 0, 0)

        export_analog_btn = QPushButton("Export Analog Data")
        export_analog_btn.clicked.connect(self._export_analog)
        export_layout.addWidget(export_analog_btn, 0, 1)

        export_force_btn = QPushButton("Export Force Plate Data")
        export_force_btn.clicked.connect(self._export_force)
        export_layout.addWidget(export_force_btn, 1, 0, 1, 2)

        layout.addWidget(export_group)

        # Status
        self.export_status = QTextEdit()
        self.export_status.setReadOnly(True)
        self.export_status.setMaximumHeight(100)
        self.export_status.setPlaceholderText("Export status will appear here...")
        layout.addWidget(self.export_status)

        layout.addStretch()
        return tab

    def _load_file(self) -> None:
        """Load a C3D file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open C3D File",
            "",
            "C3D Files (*.c3d);;All Files (*)",
        )

        if not file_path:
            return

        self._current_file = Path(file_path)

        try:
            self._load_c3d_data()
        except ImportError:
            self._show_demo_data()
        except (ValueError, RuntimeError, OSError) as exc:
            self._show_load_error(exc)
        else:
            # Paint "loaded" only after the data actually loaded (#3978).
            self.file_label.setText(self._current_file.name)
            self.file_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['green']};")
            self.export_status.setPlainText("")

    def _load_c3d_data(self) -> None:
        """Load actual C3D data using the reader."""
        from shared.python.sidekick.lab.bio.c3d_reader import C3DDataReader

        reader = C3DDataReader(self._current_file)
        metadata = reader.get_metadata()

        self._update_metadata_display(metadata)
        self._update_marker_list(metadata.marker_labels)
        self._update_analog_table(metadata.analog_labels, metadata.analog_units)

    def _show_demo_data(self) -> None:
        """Show demo data when ezc3d is not available."""
        # Annotate the file label: demo numbers must never be presented as
        # the contents of the user's chosen file (#3978).
        if self._current_file is not None:
            self.file_label.setText(
                f"{self._current_file.name} (demo data - library unavailable)"
            )
        else:
            self.file_label.setText("demo data - library unavailable")
        self.file_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['yellow']};")

        # Demo metadata
        self.info_labels["marker_count"].setText("12 (demo)")
        self.info_labels["frame_count"].setText("1000 (demo)")
        self.info_labels["frame_rate"].setText("100.0 Hz (demo)")
        self.info_labels["duration"].setText("10.0 s (demo)")
        self.info_labels["units"].setText("mm (demo)")
        self.info_labels["analog_count"].setText("6 (demo)")
        self.info_labels["analog_rate"].setText("1000.0 Hz (demo)")
        self.info_labels["event_count"].setText("2 (demo)")

        # Demo events
        self.events_list.clear()
        self.events_list.addItems(
            [
                "Event: Start @ 0.0s",
                "Event: End @ 10.0s",
            ]
        )

        # Demo markers
        demo_markers = [
            "LASI",
            "RASI",
            "LPSI",
            "RPSI",
            "LKNE",
            "RKNE",
            "LANK",
            "RANK",
            "LTOE",
            "RTOE",
            "LHEE",
            "RHEE",
        ]
        self.marker_list.clear()
        self.marker_list.addItems(demo_markers)

        # Demo analog
        self.analog_table.setRowCount(6)
        demo_analog = [
            ("Fx1", "N", "10000"),
            ("Fy1", "N", "10000"),
            ("Fz1", "N", "10000"),
            ("Mx1", "N·m", "10000"),
            ("My1", "N·m", "10000"),
            ("Mz1", "N·m", "10000"),
        ]
        for row, (channel, unit, samples) in enumerate(demo_analog):
            self.analog_table.setItem(row, 0, QTableWidgetItem(channel))
            self.analog_table.setItem(row, 1, QTableWidgetItem(unit))
            self.analog_table.setItem(row, 2, QTableWidgetItem(samples))

        self.export_status.setPlainText(
            "Note: ezc3d library not available. Showing demo data.\n"
            "Install with: pip install ezc3d"
        )
        self.export_status.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['yellow']};")

    def _show_load_error(self, exc: Exception) -> None:
        """Show a visible error state when the C3D file fails to load.

        Reader failures (ValueError/RuntimeError/OSError) must never escape
        the Qt slot uncaught, and a failure must never leave a stale
        "loaded" state on screen: the file label switches to the error color
        and panels populated by any previous load are cleared (#3978).
        """
        if self._current_file is None:
            raise ValueError("a file must be selected before reporting an error")
        self.file_label.setText(f"{self._current_file.name} (load failed)")
        self.file_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")
        self._reset_data_panels()
        self.export_status.setPlainText(
            f"Could not load '{self._current_file.name}': {exc}\n"
            "Choose a valid C3D file and try again."
        )
        self.export_status.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")

    def _reset_data_panels(self) -> None:
        """Clear data panels to their empty placeholders after a failed load."""
        for value_label in self.info_labels.values():
            value_label.setText("-")
        self.events_list.clear()
        self.marker_list.clear()
        self.analog_table.setRowCount(0)
        self.end_frame_spin.setValue(self.end_frame_spin.maximum())

    def _update_metadata_display(self, metadata: Any) -> None:
        """Update the metadata display from C3D metadata."""
        self.info_labels["marker_count"].setText(str(metadata.marker_count))
        self.info_labels["frame_count"].setText(str(metadata.frame_count))
        self.info_labels["frame_rate"].setText(f"{metadata.frame_rate:.1f} Hz")
        self.info_labels["duration"].setText(f"{metadata.duration:.2f} s")
        self.info_labels["units"].setText(metadata.units)
        self.info_labels["analog_count"].setText(str(metadata.analog_count))

        if metadata.analog_rate:
            self.info_labels["analog_rate"].setText(f"{metadata.analog_rate:.1f} Hz")
        else:
            self.info_labels["analog_rate"].setText("-")

        self.info_labels["event_count"].setText(str(len(metadata.events)))

        # Update events list
        self.events_list.clear()
        for event in metadata.events:
            self.events_list.addItem(f"{event.label} @ {event.time:.3f}s")

        # Update frame range
        self.end_frame_spin.setValue(metadata.frame_count - 1)

    def _update_marker_list(self, labels: list[str]) -> None:
        """Update the marker list widget."""
        if labels is None:
            raise ValueError("labels must be provided")
        self.marker_list.clear()
        self.marker_list.addItems(labels)

    def _update_analog_table(self, labels: list[str], units: list[str]) -> None:
        """Update the analog channels table."""
        if labels is None:
            raise ValueError("labels must be provided")
        self.analog_table.setRowCount(len(labels))
        for row, (label, unit) in enumerate(zip(labels, units, strict=True)):
            self.analog_table.setItem(row, 0, QTableWidgetItem(label))
            self.analog_table.setItem(row, 1, QTableWidgetItem(unit))
            self.analog_table.setItem(row, 2, QTableWidgetItem("-"))

    def _analyze_markers(self) -> None:
        """Analyze selected markers."""
        selected = self.marker_list.selectedItems()
        if not selected:
            self.trajectory_text.setPlainText("No markers selected.")
            return

        marker_names = [item.text() for item in selected]
        analysis = []
        analysis.append(f"Selected {len(marker_names)} markers:")
        analysis.extend([f"  - {name}" for name in marker_names])
        analysis.append("")
        analysis.append("Note: Full trajectory analysis requires ezc3d library.")
        analysis.append("Install with: pip install ezc3d")

        self.trajectory_text.setPlainText("\n".join(analysis))

    def _analyze_force_plates(self) -> None:
        """Analyze force plate data."""
        if self._current_file is None:
            self.force_text.setPlainText("No file loaded.")
            return

        try:
            from shared.python.sidekick.lab.bio.c3d_reader import C3DDataReader

            reader = C3DDataReader(self._current_file)
            plate_count = reader.get_force_plate_count()

            if plate_count == 0:
                self.force_text.setPlainText("No force plates detected in file.")
            else:
                channels = reader.get_force_plate_channels()
                analysis = [f"Detected {plate_count} force plate(s):"]
                for plate_num, ch in channels.items():
                    analysis.append(f"\nPlate {plate_num}:")
                    analysis.extend(
                        [f"  {key.upper()}: {label}" for (key, label) in ch.items()]
                    )
                self.force_text.setPlainText("\n".join(analysis))

        except ImportError:
            self.force_text.setPlainText(
                "Force plate analysis requires ezc3d library.\n"
                "Install with: pip install ezc3d"
            )

    def _export_points(self) -> None:
        """Export marker point data."""
        if self._current_file is None:
            self.export_status.setPlainText("No file loaded.")
            return

        self._do_export("points")

    def _export_analog(self) -> None:
        """Export analog channel data."""
        if self._current_file is None:
            self.export_status.setPlainText("No file loaded.")
            return

        self._do_export("analog")

    def _export_force(self) -> None:
        """Export force plate data."""
        if self._current_file is None:
            self.export_status.setPlainText("No file loaded.")
            return

        self._do_export("force")

    def _do_export(self, data_type: str) -> None:
        """Perform the export operation."""
        if data_type is None:
            raise ValueError("data_type must be provided")
        if self._current_file is None:
            self.export_status.setPlainText("No file loaded.")
            return

        file_format = self.format_combo.currentText().lower()
        suffix = f".{file_format}"

        default_name = f"{self._current_file.stem}_{data_type}{suffix}"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {data_type.title()} Data",
            default_name,
            f"{file_format.upper()} Files (*{suffix});;All Files (*)",
        )

        if not file_path:
            return

        try:
            from shared.python.sidekick.lab.bio.c3d_reader import C3DDataReader

            reader = C3DDataReader(self._current_file)

            units_text = self.units_combo.currentText()
            target_units: str | None = None if units_text == "Original" else units_text

            if data_type == "points":
                reader.export_points(
                    file_path,
                    target_units=target_units,
                    file_format=file_format,
                )
            elif data_type == "analog":
                reader.export_analog(file_path, file_format=file_format)
            else:  # force
                df = reader.force_plate_dataframe()
                if file_format == "csv":
                    df.to_csv(file_path, index=False)
                elif file_format == "json":
                    df.to_json(file_path, orient="records", indent=2)

            self.export_status.setPlainText(f"Exported {data_type} to:\n{file_path}")
            self.export_status.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['green']};")

        except ImportError:
            self.export_status.setPlainText(
                "Export requires ezc3d library.\nInstall with: pip install ezc3d"
            )
            self.export_status.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")


def main() -> int:
    """Run the C3D Motion Capture Viewer application."""
    app = QApplication(sys.argv)
    window = C3DViewerWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
