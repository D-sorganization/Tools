"""Visual theme constants and helpers for the gasification calculator.

SRP: Styling only. No layout, no data, no logic.
"""

import matplotlib

COLORS = {
    "bg": "#0a0e17",
    "panel": "#131a2b",
    "accent": "#00d4ff",
    "accent2": "#ff6b35",
    "accent3": "#7c4dff",
    "text": "#e0e6ed",
    "text_dim": "#6b7a8d",
    "grid": "#1e2d42",
    "success": "#00e676",
    "warning": "#ffab00",
    "error": "#ff1744",
}

SPECIES_COLORS = {
    "H2": "#00d4ff",
    "CO": "#ff6b35",
    "CO2": "#7c4dff",
    "H2O": "#00e676",
    "CH4": "#ffab00",
    "N2": "#ff1744",
    "O2": "#e040fb",
    "C2H4": "#00bfa5",
    "C2H6": "#ff9100",
    "H2S": "#d50000",
    "NH3": "#2979ff",
    "SO2": "#f50057",
    "C3H8": "#64dd17",
    "Ar": "#90a4ae",
    "C_solid": "#78909c",
}

ELEMENT_COLORS = {
    "C": "#ff6b35",
    "H": "#00d4ff",
    "O": "#00e676",
    "N": "#7c4dff",
    "S": "#ffab00",
    "Ash": "#78909c",
}


def apply_theme():
    """Apply dark theme to matplotlib rcParams."""
    params = {
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["panel"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
        "xtick.color": COLORS["text_dim"],
        "ytick.color": COLORS["text_dim"],
        "text.color": COLORS["text"],
        "font.family": "monospace",
        "font.size": 9,
        "lines.linewidth": 2.0,
        "lines.antialiased": True,
    }
    for k, v in params.items():
        matplotlib.rcParams[k] = v


def style_slider(slider, color):
    """Apply consistent styling to a matplotlib Slider."""
    slider.label.set_color(COLORS["text"])
    slider.valtext.set_color(color)
