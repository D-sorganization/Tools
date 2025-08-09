#!/usr/bin/env python3
"""
Integration Example: CSV to Parquet Converter as a Subtab
This example shows how to integrate the CSV to Parquet converter
as a subtab in another PyQt6 application.

Author: AI Assistant
Date: 2025
"""

import sys

# Import our CSV to Parquet converter
from csv_to_parquet_converter import CSVToParquetConverter
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QLabel,
                             QMainWindow, QPushButton, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget)


class MainApplication(QMainWindow):
    """Example main application with multiple tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Processing Suite")
        self.setMinimumSize(1000, 800)

        # Create central widget with tab widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create main layout
        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)

        # Create header
        header = QLabel("Data Processing Suite")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            "color: #2c3e50; padding: 10px; background-color: #ecf0f1; border-radius: 5px;"
        )
        main_layout.addWidget(header)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Add tabs
        self._create_dashboard_tab()
        self._create_csv_converter_tab()
        self._create_data_analysis_tab()
        self._create_settings_tab()

        # Apply styling
        self._apply_styling()

    def _create_dashboard_tab(self):
        """Create the dashboard tab."""
        dashboard_widget = QWidget()
        layout = QVBoxLayout()

        # Welcome section
        welcome_group = QGroupBox("Welcome to Data Processing Suite")
        welcome_layout = QVBoxLayout()

        welcome_text = QLabel(
            "This application provides various data processing tools:\n\n"
            "• CSV to Parquet Converter - Bulk convert CSV files to Parquet format\n"
            "• Data Analysis Tools - Analyze and visualize your data\n"
            "• Settings - Configure application preferences\n\n"
            "Select a tab above to get started!"
        )
        welcome_text.setStyleSheet("font-size: 14px; line-height: 1.5;")
        welcome_layout.addWidget(welcome_text)

        welcome_group.setLayout(welcome_layout)
        layout.addWidget(welcome_group)

        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout()

        convert_btn = QPushButton("Open CSV Converter")
        convert_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))

        analyze_btn = QPushButton("Open Data Analysis")
        analyze_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))

        actions_layout.addWidget(convert_btn)
        actions_layout.addWidget(analyze_btn)
        actions_layout.addStretch()

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        layout.addStretch()
        dashboard_widget.setLayout(layout)

        self.tab_widget.addTab(dashboard_widget, "Dashboard")

    def _create_csv_converter_tab(self):
        """Create the CSV to Parquet converter tab."""
        # Create our converter widget
        converter_widget = CSVToParquetConverter()

        # Add it to the tab widget
        self.tab_widget.addTab(converter_widget, "CSV to Parquet Converter")

    def _create_data_analysis_tab(self):
        """Create a placeholder data analysis tab."""
        analysis_widget = QWidget()
        layout = QVBoxLayout()

        # Placeholder content
        placeholder_group = QGroupBox("Data Analysis Tools")
        placeholder_layout = QVBoxLayout()

        placeholder_text = QLabel(
            "Data analysis tools will be implemented here.\n\n"
            "This could include:\n"
            "• Statistical analysis\n"
            "• Data visualization\n"
            "• Report generation\n"
            "• Data cleaning tools"
        )
        placeholder_text.setStyleSheet("font-size: 14px; line-height: 1.5;")
        placeholder_layout.addWidget(placeholder_text)

        placeholder_group.setLayout(placeholder_layout)
        layout.addWidget(placeholder_group)
        layout.addStretch()

        analysis_widget.setLayout(layout)
        self.tab_widget.addTab(analysis_widget, "Data Analysis")

    def _create_settings_tab(self):
        """Create a settings tab."""
        settings_widget = QWidget()
        layout = QVBoxLayout()

        # Settings content
        settings_group = QGroupBox("Application Settings")
        settings_layout = QVBoxLayout()

        settings_text = QLabel(
            "Application settings and configuration options will be displayed here.\n\n"
            "This could include:\n"
            "• Default file paths\n"
            "• Processing preferences\n"
            "• UI customization options\n"
            "• Performance settings"
        )
        settings_text.setStyleSheet("font-size: 14px; line-height: 1.5;")
        settings_layout.addWidget(settings_text)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        layout.addStretch()

        settings_widget.setLayout(layout)
        self.tab_widget.addTab(settings_widget, "Settings")

    def _apply_styling(self):
        """Apply custom styling to the application."""
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f8f9fa;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #007bff;
            }
            QTabBar::tab:hover {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """
        )


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Data Processing Suite")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Data Processor")

    # Create and show main window
    window = MainApplication()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
