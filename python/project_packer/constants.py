"""
Constants for Project Packer application.

This module contains all configuration constants used throughout the application
to avoid magic numbers and provide clear documentation of values and their sources.
"""

from typing import Final

# =============================================================================
# UI CONSTANTS
# =============================================================================

# Window dimensions
DEFAULT_WINDOW_WIDTH: Final[int] = 600  # [px] Default application window width
DEFAULT_WINDOW_HEIGHT: Final[int] = 500  # [px] Default application window height

# UI spacing and layout
DEFAULT_PADDING: Final[int] = 20  # [px] Default padding for UI elements
SMALL_PADDING: Final[int] = 10  # [px] Small padding for compact elements
TINY_PADDING: Final[int] = 5  # [px] Tiny padding for minimal spacing

# Font sizes
TITLE_FONT_SIZE: Final[int] = 16  # [pt] Title font size
HEADER_FONT_SIZE: Final[int] = 10  # [pt] Header font size
BOLD_HEADER_FONT_SIZE: Final[int] = 10  # [pt] Bold header font size

# Listbox dimensions
DEFAULT_LISTBOX_HEIGHT: Final[int] = 6  # [lines] Default listbox height
STATUS_TEXT_HEIGHT: Final[int] = 8  # [lines] Status text area height

# Grid weights
GRID_WEIGHT_MAIN: Final[int] = 1  # Main grid weight for responsive layout

# =============================================================================
# SOURCES AND REFERENCES
# =============================================================================

"""
Sources for constants:

1. UI Constants:
   - Window dimensions: Based on content requirements and common desktop resolutions
   - Padding values: Standard UI design guidelines for comfortable spacing
   - Font sizes: Accessibility guidelines and readability standards
   - Listbox heights: Balance between information display and screen real estate

2. Grid Layout:
   - Weight values: Standard tkinter grid weight system for responsive layouts
"""
