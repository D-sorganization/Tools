"""Shared base types for UI widgets."""

from dataclasses import dataclass


@dataclass
class PanelStyle:
    """Styling for UI panels."""

    background_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.7)
    text_color: tuple[int, int, int] = (220, 220, 220)
    title_color: tuple[int, int, int] = (255, 255, 100)
    border_color: tuple[float, float, float, float] = (0.3, 0.3, 0.4, 0.5)
    padding: int = 15
    line_height: int = 24
    font_size: int = 26  # Increased from 12
    title_font_size: int = 32  # Increased from 14
