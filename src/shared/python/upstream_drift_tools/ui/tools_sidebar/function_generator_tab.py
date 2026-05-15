"""Function Generator Sidekick tab integration."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from . import design_tokens as theme
from .qt_compat import QT_API, QtWidgets

logger = logging.getLogger(__name__)

FUNCTION_GENERATOR_TAB_ID = "function_generator"


def build_function_generator_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Function Generator tab when its PyQt6 surface is available.
    
    This tab provides lazy-loaded access to the Function Generator tool
    within the Sidekick sidebar, following the same pattern as other
    optional tabs like Rotation Converter and Unit Converter.
    """
    if QT_API != "PyQt6":
        return placeholder(
            sidebar,
            "Function Generator",
            "Function Generator requires the PyQt6 UI backend.",
        )
    try:
        # Lazy-load the Function Generator widget to avoid import overhead
        # when the tab is not being used
        module = importlib.import_module(
            "function_generator.python.function_generator.ui.pyqt6.main_window"
        )
        window_type = module.FunctionGeneratorWidget
        widget = window_type(parent=sidebar)
    except Exception as exc:  # noqa: BLE001 - optional GUI surface
        logger.debug("Function Generator unavailable for Sidekick: %s", exc)
        return placeholder(
            sidebar,
            "Function Generator",
            "Function Generator is unavailable because optional UI dependencies "
            "could not be loaded.",
        )
    
    widget.setObjectName(theme.SIDEKICK_FUNCTION_GENERATOR_OBJECT_NAME)
    return widget


def placeholder(
    sidebar: Any,
    title: str,
    message: str | None = None,
) -> QtWidgets.QWidget:
    """Build a compact placeholder for optional tabs."""
    widget = QtWidgets.QWidget(sidebar)
    widget.setObjectName(theme.SIDEKICK_PLACEHOLDER_OBJECT_NAME)
    layout = QtWidgets.QVBoxLayout(widget)
    label = QtWidgets.QLabel(title, widget)
    label.setObjectName(theme.SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME)
    label.setWordWrap(True)
    layout.addWidget(label)
    if message:
        detail = QtWidgets.QLabel(message, widget)
        detail.setWordWrap(True)
        layout.addWidget(detail)
    layout.addStretch(1)
    return widget