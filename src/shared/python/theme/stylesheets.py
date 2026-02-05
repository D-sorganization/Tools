"""Qt stylesheet generation for the fleet-wide theme system.

This module generates complete QSS stylesheets for PyQt6 applications
based on theme color definitions.
"""

from __future__ import annotations


def generate_stylesheet(theme: dict[str, str]) -> str:
    """Generate a complete QSS stylesheet for a theme.

    Args:
        theme: Theme dictionary with color values

    Returns:
        Complete QSS stylesheet string
    """
    return f"""
        /* ================================================================ */
        /* Main Window and Widgets */
        /* ================================================================ */
        QMainWindow, QWidget {{
            background-color: {theme["bg"]};
            color: {theme["text"]};
        }}

        /* ================================================================ */
        /* Group Boxes */
        /* ================================================================ */
        QGroupBox {{
            font-weight: bold;
            border: 2px solid {theme["border"]};
            border-radius: 5px;
            margin: 5px;
            padding-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }}

        /* ================================================================ */
        /* Scroll Areas */
        /* ================================================================ */
        QScrollArea {{
            border: none;
            background-color: {theme["bg"]};
        }}

        /* ================================================================ */
        /* Input Widgets */
        /* ================================================================ */
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            padding: 5px 8px;
            font-size: 12px;
            border: 1px solid {theme["border"]};
            border-radius: 3px;
            background-color: {theme["input_bg"]};
            color: {theme["text"]};
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {theme["focus"]};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
            background-color: {theme["input_bg"]};
        }}
        QComboBox::down-arrow {{
            width: 10px;
            height: 10px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {theme["input_bg"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
            selection-background-color: {theme["accent"]};
            selection-color: white;
        }}

        /* ================================================================ */
        /* Labels */
        /* ================================================================ */
        QLabel {{
            color: {theme["text_secondary"]};
            font-size: 11px;
            background: transparent;
        }}

        /* ================================================================ */
        /* Buttons */
        /* ================================================================ */
        QPushButton {{
            padding: 6px 14px;
            font-size: 11px;
            border: 1px solid {theme["border"]};
            border-radius: 4px;
            background-color: {theme["group_bg"]};
            color: {theme["text"]};
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {theme["accent"]};
            color: white;
            border-color: {theme["accent"]};
        }}
        QPushButton:pressed {{
            background-color: {theme["button_hover"]};
            border-color: {theme["button_hover"]};
        }}
        QPushButton:disabled {{
            background-color: {theme["border"]};
            color: {theme["label"]};
            border-color: {theme["border"]};
        }}
        QPushButton:checked {{
            background-color: {theme["accent"]};
            color: white;
            border-color: {theme["accent"]};
        }}

        /* ================================================================ */
        /* Menu Bar */
        /* ================================================================ */
        QMenuBar {{
            background-color: {theme["bg"]};
            border-bottom: 1px solid {theme["border"]};
            padding: 4px;
        }}
        QMenuBar::item {{
            padding: 6px 12px;
            background: transparent;
            color: {theme["text"]};
        }}
        QMenuBar::item:selected {{
            background: rgba(128, 189, 255, 0.2);
            color: {theme["text"]};
            border-radius: 3px;
        }}

        /* ================================================================ */
        /* Menus */
        /* ================================================================ */
        QMenu {{
            background-color: {theme["input_bg"]};
            border: 1px solid {theme["border"]};
            padding: 4px;
        }}
        QMenu::item {{
            padding: 8px 24px 8px 12px;
            color: {theme["text"]};
            border-radius: 2px;
        }}
        QMenu::item:selected {{
            background-color: rgba(128, 189, 255, 0.3);
            color: {theme["text"]};
        }}
        QMenu::separator {{
            height: 1px;
            background: {theme["border"]};
            margin: 4px 8px;
        }}

        /* ================================================================ */
        /* Tab Widget */
        /* ================================================================ */
        QTabWidget::pane {{
            border: 1px solid {theme["border"]};
            background-color: {theme["bg"]};
            border-radius: 2px;
        }}
        QTabBar::tab {{
            background-color: {theme["group_bg"]};
            color: {theme["text_secondary"]};
            border: 1px solid {theme["border"]};
            border-bottom: none;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 2px;
            border-top-right-radius: 2px;
            font-size: 11px;
            font-weight: 500;
        }}
        QTabBar::tab:selected {{
            background-color: {theme["bg"]};
            color: {theme["accent"]};
            border-bottom: 2px solid {theme["accent"]};
            font-weight: 600;
        }}
        QTabBar::tab:hover {{
            background-color: {theme["title_bg"]};
            color: {theme["text"]};
        }}

        /* ================================================================ */
        /* Table Widget */
        /* ================================================================ */
        QTableWidget, QTableView {{
            background-color: {theme["input_bg"]};
            alternate-background-color: {theme["table_alt"]};
            gridline-color: {theme["border"]};
            border: 1px solid {theme["border"]};
            selection-background-color: {theme["accent"]};
            selection-color: white;
        }}
        QHeaderView::section {{
            background-color: {theme["table_header"]};
            color: {theme["text_secondary"]};
            padding: 6px;
            border: 1px solid {theme["border"]};
            font-weight: 600;
            font-size: 11px;
        }}

        /* ================================================================ */
        /* Tree Widget */
        /* ================================================================ */
        QTreeWidget, QTreeView {{
            background-color: {theme["input_bg"]};
            alternate-background-color: {theme["table_alt"]};
            border: 1px solid {theme["border"]};
            selection-background-color: {theme["accent"]};
            selection-color: white;
        }}
        QTreeWidget::item, QTreeView::item {{
            padding: 4px;
        }}
        QTreeWidget::item:selected, QTreeView::item:selected {{
            background-color: {theme["accent"]};
            color: white;
        }}

        /* ================================================================ */
        /* List Widget */
        /* ================================================================ */
        QListWidget, QListView {{
            background-color: {theme["input_bg"]};
            alternate-background-color: {theme["table_alt"]};
            border: 1px solid {theme["border"]};
            selection-background-color: {theme["accent"]};
            selection-color: white;
        }}

        /* ================================================================ */
        /* Text Edit */
        /* ================================================================ */
        QTextEdit, QPlainTextEdit {{
            background-color: {theme["input_bg"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
            border-radius: 3px;
            padding: 6px;
            font-size: 11px;
        }}

        /* ================================================================ */
        /* Progress Bar */
        /* ================================================================ */
        QProgressBar {{
            border: 1px solid {theme["border"]};
            border-radius: 3px;
            text-align: center;
            background-color: {theme["group_bg"]};
            color: {theme["text"]};
        }}
        QProgressBar::chunk {{
            background-color: {theme["accent"]};
            border-radius: 2px;
        }}

        /* ================================================================ */
        /* Status Bar */
        /* ================================================================ */
        QStatusBar {{
            background-color: {theme["bg"]};
            color: {theme["text_secondary"]};
            border-top: 1px solid {theme["border"]};
        }}

        /* ================================================================ */
        /* Slider */
        /* ================================================================ */
        QSlider::groove:horizontal {{
            border: 1px solid {theme["border"]};
            height: 4px;
            background: {theme["group_bg"]};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {theme["accent"]};
            border: 1px solid {theme["accent"]};
            width: 14px;
            margin: -6px 0;
            border-radius: 7px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {theme["button_hover"]};
        }}
        QSlider::groove:vertical {{
            border: 1px solid {theme["border"]};
            width: 4px;
            background: {theme["group_bg"]};
            border-radius: 2px;
        }}
        QSlider::handle:vertical {{
            background: {theme["accent"]};
            border: 1px solid {theme["accent"]};
            height: 14px;
            margin: 0 -6px;
            border-radius: 7px;
        }}

        /* ================================================================ */
        /* Checkbox and Radio Button */
        /* ================================================================ */
        QCheckBox, QRadioButton {{
            color: {theme["text_secondary"]};
            spacing: 8px;
        }}
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {theme["border"]};
            background-color: {theme["input_bg"]};
        }}
        QCheckBox::indicator {{
            border-radius: 3px;
        }}
        QRadioButton::indicator {{
            border-radius: 8px;
        }}
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
            background-color: {theme["accent"]};
            border-color: {theme["accent"]};
        }}

        /* ================================================================ */
        /* Tooltips */
        /* ================================================================ */
        QToolTip {{
            background-color: {theme["title_bg"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
            padding: 4px;
            border-radius: 3px;
        }}

        /* ================================================================ */
        /* Toolbar */
        /* ================================================================ */
        QToolBar {{
            background-color: {theme["bg"]};
            border-bottom: 1px solid {theme["border"]};
            spacing: 4px;
            padding: 4px;
        }}
        QToolBar::separator {{
            background-color: {theme["border"]};
            width: 1px;
            margin: 4px 2px;
        }}
        QToolButton {{
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 4px;
            padding: 4px;
        }}
        QToolButton:hover {{
            background-color: {theme["group_bg"]};
            border-color: {theme["border"]};
        }}
        QToolButton:pressed {{
            background-color: {theme["accent"]};
        }}

        /* ================================================================ */
        /* Dock Widget */
        /* ================================================================ */
        QDockWidget {{
            color: {theme["text"]};
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }}
        QDockWidget::title {{
            background-color: {theme["title_bg"]};
            border: 1px solid {theme["title_border"]};
            padding: 6px;
            text-align: left;
        }}
        QDockWidget::close-button, QDockWidget::float-button {{
            border: none;
            background: transparent;
            padding: 2px;
        }}

        /* ================================================================ */
        /* Scroll Bars */
        /* ================================================================ */
        QScrollBar:vertical {{
            background-color: {theme["bg"]};
            width: 12px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background-color: {theme["border"]};
            border-radius: 6px;
            min-height: 20px;
            margin: 2px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {theme["accent"]};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar:horizontal {{
            background-color: {theme["bg"]};
            height: 12px;
            margin: 0;
        }}
        QScrollBar::handle:horizontal {{
            background-color: {theme["border"]};
            border-radius: 6px;
            min-width: 20px;
            margin: 2px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background-color: {theme["accent"]};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}

        /* ================================================================ */
        /* Splitter */
        /* ================================================================ */
        QSplitter::handle {{
            background-color: {theme["border"]};
        }}
        QSplitter::handle:horizontal {{
            width: 2px;
        }}
        QSplitter::handle:vertical {{
            height: 2px;
        }}
        QSplitter::handle:hover {{
            background-color: {theme["accent"]};
        }}

        /* ================================================================ */
        /* Dialogs */
        /* ================================================================ */
        QDialog {{
            background-color: {theme["bg"]};
            color: {theme["text"]};
        }}
        QDialogButtonBox {{
            background-color: transparent;
        }}

        /* ================================================================ */
        /* Message Box */
        /* ================================================================ */
        QMessageBox {{
            background-color: {theme["bg"]};
        }}
        QMessageBox QLabel {{
            color: {theme["text"]};
        }}

        /* ================================================================ */
        /* File Dialog */
        /* ================================================================ */
        QFileDialog {{
            background-color: {theme["bg"]};
        }}

        /* ================================================================ */
        /* Calendar Widget */
        /* ================================================================ */
        QCalendarWidget {{
            background-color: {theme["bg"]};
        }}
        QCalendarWidget QToolButton {{
            color: {theme["text"]};
            background-color: {theme["group_bg"]};
        }}
        QCalendarWidget QMenu {{
            background-color: {theme["input_bg"]};
        }}
        QCalendarWidget QSpinBox {{
            background-color: {theme["input_bg"]};
        }}

        /* ================================================================ */
        /* Frame */
        /* ================================================================ */
        QFrame {{
            border: none;
        }}
        QFrame[frameShape="4"], QFrame[frameShape="5"] {{
            border: 1px solid {theme["border"]};
        }}
    """


def generate_minimal_stylesheet(theme: dict[str, str]) -> str:
    """Generate a minimal stylesheet for embedding in parent applications.

    This is useful for applications that want to apply theme colors
    without overriding all widget styles.

    Args:
        theme: Theme dictionary with color values

    Returns:
        Minimal QSS stylesheet string
    """
    return f"""
        QWidget {{
            background-color: {theme["bg"]};
            color: {theme["text"]};
        }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: {theme["input_bg"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
        }}
        QPushButton {{
            background-color: {theme["group_bg"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
        }}
        QPushButton:hover {{
            background-color: {theme["accent"]};
            color: white;
        }}
    """


__all__ = [
    "generate_minimal_stylesheet",
    "generate_stylesheet",
]
