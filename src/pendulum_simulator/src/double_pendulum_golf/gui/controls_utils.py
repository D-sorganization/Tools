"""
Shared utilities for double and triple pendulum control widgets.

DRY: All parse helpers and playback state used by both
     ControlsWidget and ControlsWidgetTriple live here.

Design by Contract
------------------
- parse_float raises ValueError (not returns None) on bad input
- parse_coeffs raises ValueError on non-numeric tokens
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .controls_widget import LabeledInput

# ---------------------------------------------------------------------------
# Stylesheet tokens shared by both control panels
# ---------------------------------------------------------------------------

STYLE_GROUP = (
    "QGroupBox { color: #c8c8e0; border: 1px solid #404060;"
    "border-radius: 6px; margin-top: 10px; padding-top: 14px;"
    "font-weight: bold; font-size: 10px; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
)
STYLE_EDIT = (
    "background: #1e1e30; color: #e0e0f0; border: 1px solid #404060;"
    "border-radius: 3px; padding: 2px 5px; font-family: monospace; font-size: 10px;"
)
STYLE_LABEL = "color: #9090b0; font-size: 10px;"
STYLE_SPIN = (
    "background: #1e1e30; color: #e0e0f0; border: 1px solid #404060;"
    "border-radius: 3px; padding: 1px 4px; font-size: 10px;"
)
STYLE_CHECK = (
    "QCheckBox { color: #b8b8d0; font-size: 11px; spacing: 4px; }"
    "QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #484868;"
    "border-radius: 3px; background: #22223a; }"
    "QCheckBox::indicator:checked { background: #5060a0; border-color: #7080c0; }"
)
STYLE_SLIDER = (
    "QSlider::groove:horizontal { background: #252540; height: 6px; border-radius: 3px; }"
    "QSlider::sub-page:horizontal { background: #5060a0; border-radius: 3px; }"
    "QSlider::handle:horizontal { background: #8090d0; width: 14px;"
    "margin: -5px 0; border-radius: 7px; border: 1px solid #6070b0; }"
    "QSlider::handle:horizontal:hover { background: #a0b0f0; }"
)
STYLE_BTN = (
    "QPushButton { background: #282848; color: #c0c0e8; border: 1px solid #404068;"
    "border-radius: 5px; padding: 6px 14px; font-size: 11px; }"
    "QPushButton:hover { background: #30306a; color: #e0e0ff; }"
    "QPushButton:pressed { background: #20204a; }"
)
STYLE_BTN_IMPORT = (
    "QPushButton { background: #1e4a2a; color: #a0e8b0; border: 1px solid #285a38;"
    "border-radius: 5px; padding: 7px 14px; font-weight: bold; font-size: 11px; }"
    "QPushButton:hover { background: #286038; }"
    "QPushButton:pressed { background: #153820; }"
)


# ---------------------------------------------------------------------------
# Input parsing helpers
# ---------------------------------------------------------------------------


def parse_float(widget: LabeledInput, name: str) -> float:
    """Parse a single float from a LabeledInput; raise ValueError on failure.

    Parameters
    ----------
    widget:
        The LabeledInput whose .value is parsed.
    name:
        Human-readable field name used in the error message.

    Raises
    ------
    ValueError
        If the text cannot be converted to float.
    """
    try:
        return float(widget.value)
    except ValueError:
        raise ValueError(f"Cannot parse '{name}': '{widget.value}'") from None


def parse_coeffs(widget: LabeledInput, name: str) -> list[float]:
    """Parse comma-separated polynomial coefficients from a LabeledInput.

    Parameters
    ----------
    widget:
        The LabeledInput whose .value is parsed.
    name:
        Human-readable field name used in the error message.

    Returns
    -------
    list[float]
        Parsed coefficients; empty list converts to [0.0] at call site.

    Raises
    ------
    ValueError
        If any token cannot be converted to float.
    """
    try:
        parts = widget.value.split(",")
        return [float(p.strip()) for p in parts if p.strip()]
    except ValueError:
        raise ValueError(
            f"Cannot parse '{name}' coefficients: '{widget.value}'"
        ) from None


def parse_coeffs_lenient(widget: LabeledInput) -> list[float]:
    """Silently-tolerant version used for live torque preview rendering.

    Returns [0.0] on any parse failure rather than raising.
    """
    parts = widget.value.split(",")
    result: list[float] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            result.append(float(part))
        except ValueError:
            return []
    return result or [0.0]


def clamp_dt(raw: float) -> float:
    """Clamp a raw dt value to a sane simulation range [1e-5, 0.1].

    Precondition: raw is a finite float (already parsed).
    """
    assert isinstance(raw, float), "dt must be a float"
    return max(1e-5, min(0.1, raw))
