"""Showcase application stylesheet (epic #4125, H6).

Applies the UpstreamDrift launcher's visual language across the PyQt6
app, pattern-matched to the shared theme package's
``stylesheets.generate_stylesheet`` conventions (studied from the
UpstreamDrift launcher ToolCard/launchButton styling: hover
highlighting toward the accent, 4-8 px radii, consistent paddings,
subtle depth): buttons gain hover/pressed states with a subtle
bottom-edge shadow, group boxes and tabs get the launcher's rounded
card treatment, and spacing is normalized.

Every color is DERIVED from the live :class:`~PyQt6.QtGui.QPalette`
(``palette(...)`` QSS references and low-alpha rgba tints of palette
roles) — nothing hard-coded — so the sheet layers cleanly on top of
whatever theme the shared launcher's ``setup_themed_app`` applied.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette

__all__ = ["showcase_stylesheet"]


def _rgba(color: QColor, alpha: int) -> str:
    """Low-alpha tint of a palette color for hover/selection washes."""
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha})"


def showcase_stylesheet(palette: QPalette) -> str:
    """The launcher-language QSS for the whole app, palette-derived.

    Includes the #4120 V4 selected-row styling (this sheet replaces
    ``selection_stylesheet`` at the window level) plus the H6 launcher
    treatment for buttons, group boxes, and tabs.
    """
    highlight = palette.color(QPalette.ColorRole.Highlight)
    shadow = palette.color(QPalette.ColorRole.Shadow)
    tint = _rgba(highlight, 44)
    hover_wash = _rgba(highlight, 28)
    # QSS has no box-shadow: the launcher's subtle depth is emulated
    # with a slightly heavier bottom edge in the palette shadow tone.
    edge = _rgba(shadow, 90)
    return f"""
        /* Selected result rows (epic 4120 V4, carried forward). */
        QFrame#resultRow {{ border-radius: 6px; }}
        QFrame#resultRow:hover {{ border: 1px solid palette(highlight); }}
        QFrame#resultRow[selected="true"] {{
            background-color: {tint};
            border: 1px solid palette(highlight);
        }}

        /* Launcher-style buttons (H6): hover highlight + subtle shadow. */
        QPushButton {{
            padding: 6px 14px;
            border: 1px solid palette(mid);
            border-bottom: 2px solid {edge};
            border-radius: 4px;
            background-color: palette(button);
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {hover_wash};
            border-color: palette(highlight);
        }}
        QPushButton:pressed {{
            background-color: {tint};
            border-bottom: 1px solid {edge};
            margin-top: 1px;
        }}
        QPushButton:checked {{
            background-color: {tint};
            border-color: palette(highlight);
        }}
        QPushButton:disabled {{
            color: palette(mid);
            border-bottom: 1px solid palette(mid);
        }}
        QToolButton {{
            border: 1px solid palette(mid);
            border-radius: 4px;
            padding: 2px 8px;
            font-weight: bold;
        }}
        QToolButton:hover {{
            background-color: {hover_wash};
            border-color: palette(highlight);
        }}

        /* Launcher-card group boxes (H6): rounded, consistent titles. */
        QGroupBox {{
            border: 1px solid palette(mid);
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 6px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            font-weight: bold;
        }}

        /* Tabs: hover highlighting in the launcher language (H6). */
        QTabBar::tab {{
            padding: 6px 14px;
            border: 1px solid palette(mid);
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 2px;
        }}
        QTabBar::tab:hover {{
            background-color: {hover_wash};
        }}
        QTabBar::tab:selected {{
            border-color: palette(highlight);
            background-color: {tint};
            font-weight: bold;
        }}
        QTabWidget::pane {{
            border: 1px solid palette(mid);
            border-radius: 6px;
            top: -1px;
        }}
    """
