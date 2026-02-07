"""
Widget Factory - Common UI widget creation utilities

This module provides factory functions to create commonly used PyQt widgets
with standardized configurations, reducing code duplication across the codebase.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
)


def create_double_spinbox(
    min_value: float = 0.0,
    max_value: float = 1000.0,
    default_value: float = 0.0,
    decimals: int = 1,
    suffix: str = "",
    prefix: str = "",
    no_buttons: bool = True,
    value_changed_callback: Callable | None = None,
) -> QDoubleSpinBox:
    """Create a QDoubleSpinBox with common default settings.

    Args:
        min_value: Minimum value
        max_value: Maximum value
        default_value: Default value
        decimals: Number of decimal places
        suffix: Suffix text (e.g., " in", " °C")
        prefix: Prefix text (e.g., "$", "€")
        no_buttons: Whether to hide spin buttons
        value_changed_callback: Optional callback for value changes

    Returns:
        Configured QDoubleSpinBox
    """
    spinbox = QDoubleSpinBox()
    spinbox.setRange(min_value, max_value)
    spinbox.setValue(default_value)
    spinbox.setDecimals(decimals)
    if suffix:
        spinbox.setSuffix(suffix)
    if prefix:
        spinbox.setPrefix(prefix)
    if no_buttons:
        spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    if value_changed_callback:
        spinbox.valueChanged.connect(value_changed_callback)
    return spinbox


def create_slider(
    min_value: int = 0,
    max_value: int = 100,
    default_value: int = 50,
    orientation: Qt.Orientation = Qt.Orientation.Horizontal,
    single_step: int = 1,
    tick_interval: int = 10,
    show_ticks: bool = False,
    value_changed_callback: Callable | None = None,
) -> QSlider:
    """Create a QSlider with common default settings.

    Args:
        min_value: Minimum value
        max_value: Maximum value
        default_value: Default value
        orientation: Horizontal or Vertical
        single_step: Step size
        tick_interval: Tick mark interval
        show_ticks: Whether to show tick marks
        value_changed_callback: Optional callback for value changes

    Returns:
        Configured QSlider
    """
    slider = QSlider(orientation)
    slider.setRange(min_value, max_value)
    slider.setValue(default_value)
    slider.setSingleStep(single_step)
    if show_ticks:
        slider.setTickInterval(tick_interval)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    if value_changed_callback:
        slider.valueChanged.connect(value_changed_callback)
    return slider


def create_checkbox(
    text: str = "",
    checked: bool = False,
    state_changed_callback: Callable | None = None,
) -> QCheckBox:
    """Create a QCheckBox with common default settings.

    Args:
        text: Checkbox label text
        checked: Initial checked state
        state_changed_callback: Optional callback for state changes

    Returns:
        Configured QCheckBox
    """
    checkbox = QCheckBox(text)
    checkbox.setChecked(checked)
    if state_changed_callback:
        checkbox.stateChanged.connect(state_changed_callback)
    return checkbox


def create_readonly_lineedit(
    default_text: str = "",
    background_color: str = "#f0f0f0",
) -> QLineEdit:
    """Create a read-only QLineEdit for displaying values.

    Args:
        default_text: Initial text
        background_color: Background color hex code

    Returns:
        Configured read-only QLineEdit
    """
    lineedit = QLineEdit(default_text)
    lineedit.setReadOnly(True)
    lineedit.setStyleSheet(f"background-color: {background_color};")
    return lineedit


def create_button(
    text: str,
    clicked_callback: Callable | None = None,
    tooltip: str = "",
    style: str | None = None,
) -> QPushButton:
    """Create a QPushButton with common default settings.

    Args:
        text: Button text
        clicked_callback: Optional callback for click events
        tooltip: Tooltip text
        style: Optional CSS style string

    Returns:
        Configured QPushButton
    """
    button = QPushButton(text)
    if tooltip:
        button.setToolTip(tooltip)
    if style:
        button.setStyleSheet(style)
    if clicked_callback:
        button.clicked.connect(clicked_callback)
    return button


def create_combobox(
    items: list[str],
    default_item: str | None = None,
    current_text_changed_callback: Callable | None = None,
) -> QComboBox:
    """Create a QComboBox with common default settings.

    Args:
        items: List of items to add
        default_item: Default selected item (uses first item if None)
        current_text_changed_callback: Optional callback for selection changes

    Returns:
        Configured QComboBox
    """
    combobox = QComboBox()
    combobox.addItems(items)
    if default_item:
        combobox.setCurrentText(default_item)
    elif items:
        combobox.setCurrentIndex(0)
    if current_text_changed_callback:
        combobox.currentTextChanged.connect(current_text_changed_callback)
    return combobox


def create_label(
    text: str = "",
    font_size: int | None = None,
    bold: bool = False,
    alignment: Qt.AlignmentFlag | None = None,
    color: str | None = None,
) -> QLabel:
    """Create a QLabel with common default settings.

    Args:
        text: Label text
        font_size: Font size in points
        bold: Whether text should be bold
        alignment: Text alignment
        color: Text color hex code

    Returns:
        Configured QLabel
    """
    label = QLabel(text)
    if font_size or bold:
        from PyQt6.QtGui import QFont

        font = QFont()
        if font_size:
            font.setPointSize(font_size)
        if bold:
            font.setBold(True)
        label.setFont(font)
    if alignment:
        label.setAlignment(alignment)
    if color:
        label.setStyleSheet(f"color: {color};")
    return label
