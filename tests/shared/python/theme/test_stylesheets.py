from __future__ import annotations

import pytest

from src.shared.python.theme import stylesheets


@pytest.fixture
def sentinel_theme() -> dict[str, str]:
    return {
        "bg": "#010101",
        "group_bg": "#020202",
        "border": "#030303",
        "text": "#040404",
        "text_secondary": "#050505",
        "label": "#060606",
        "focus": "#070707",
        "input_bg": "#080808",
        "accent": "#090909",
        "title_bg": "#0a0a0a",
        "title_border": "#0b0b0b",
        "table_header": "#0c0c0c",
        "table_alt": "#0d0d0d",
        "button_hover": "#0e0e0e",
    }


def test_generate_stylesheet_includes_all_widget_sections(
    sentinel_theme: dict[str, str],
) -> None:
    sheet = stylesheets.generate_stylesheet(sentinel_theme)

    expected_selectors = [
        "QMainWindow, QWidget",
        "QGroupBox",
        "QScrollArea",
        "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox",
        "QComboBox::drop-down",
        "QComboBox QAbstractItemView",
        "QLabel",
        "QPushButton",
        "QMenuBar",
        "QMenu::separator",
        "QTabWidget::pane",
        "QTableWidget, QTableView",
        "QTreeWidget, QTreeView",
        "QListWidget, QListView",
        "QTextEdit, QPlainTextEdit",
        "QProgressBar",
        "QStatusBar",
        "QSlider::groove:horizontal",
        "QSlider::handle:vertical",
        "QCheckBox, QRadioButton",
        "QToolTip",
        "QToolBar",
        "QToolButton:pressed",
        "QDockWidget",
        "QScrollBar:vertical",
        "QScrollBar:horizontal",
        "QSplitter::handle",
        "QDialogButtonBox",
        "QMessageBox QLabel",
        "QCalendarWidget QSpinBox",
        'QFrame[frameShape="4"], QFrame[frameShape="5"]',
        "ToolCard",
        "#toolCardTitle",
        "#launchButton:pressed",
        "#helpButton:hover",
    ]
    for selector in expected_selectors:
        assert selector in sheet

    for color in sentinel_theme.values():
        assert color in sheet


def test_generate_stylesheet_preserves_section_order(
    sentinel_theme: dict[str, str],
) -> None:
    sheet = stylesheets.generate_stylesheet(sentinel_theme)

    assert sheet.index("QMainWindow, QWidget") < sheet.index("QGroupBox")
    assert sheet.index("QGroupBox") < sheet.index("QScrollArea")
    assert sheet.index("QToolButton:pressed") < sheet.index("QDockWidget")
    assert sheet.index("QDockWidget") < sheet.index("QSplitter::handle")
    assert sheet.index("QSplitter::handle") < sheet.index("ToolCard")


def test_generate_minimal_stylesheet_limits_scope_to_embedding_defaults(
    sentinel_theme: dict[str, str],
) -> None:
    sheet = stylesheets.generate_minimal_stylesheet(sentinel_theme)

    assert "QWidget" in sheet
    assert "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox" in sheet
    assert "QPushButton:hover" in sheet
    assert sentinel_theme["bg"] in sheet
    assert sentinel_theme["input_bg"] in sheet
    assert sentinel_theme["accent"] in sheet
    assert "QMenuBar" not in sheet
    assert "ToolCard" not in sheet
    assert "QScrollBar" not in sheet


def test_generators_raise_key_error_for_missing_theme_color(
    sentinel_theme: dict[str, str],
) -> None:
    theme = dict(sentinel_theme)
    del theme["button_hover"]

    with pytest.raises(KeyError, match="button_hover"):
        stylesheets.generate_stylesheet(theme)


def test_public_exports_list_stylesheet_generators() -> None:
    assert stylesheets.__all__ == [
        "generate_minimal_stylesheet",
        "generate_stylesheet",
    ]
