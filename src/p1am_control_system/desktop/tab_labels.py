"""Single source of truth for P1AM desktop tab labels."""

from __future__ import annotations

TAB_ORDER: tuple[str, ...] = (
    "mimic",
    "trends",
    "control",
    "routing",
    "history",
    "settings",
)

TAB_TITLES: dict[str, str] = {
    "mimic": "Plant Mimic Diagram",
    "trends": "Trends & Signal Filters",
    "control": "PID & MPC Control Loops",
    "routing": "DCS Routing Matrix",
    "history": "Event History",
    "settings": "Settings",
}

TOGGLEABLE_TAB_ORDER: tuple[str, ...] = TAB_ORDER[:-1]
