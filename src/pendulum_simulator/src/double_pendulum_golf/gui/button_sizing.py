"""Single-source-of-truth helper for sizing toolstrip buttons to their text.

Why a separate module
---------------------
Several toolstrip buttons (Equations of Motion, Pop-Out Chart,
Diagnostics, …) had no minimum-width hint and got truncated when the
header was crowded. The fix is one ``setMinimumWidth(text + chrome)``
call per button — but doing it inline once per button breeds copy-paste
drift the moment a label changes (the existing model dropdown already
had its own bespoke version of this code).

This module is the **single** place that computes "how wide should a
QPushButton be to display its current text?". Both the toolstrip and
any future widget that wants its buttons to fit call
``fit_button_to_text(btn)``.

Design by Contract
------------------
- Pre:  ``button`` is a ``QPushButton`` (or ``QAbstractButton`` with
        a ``text()`` method); ``padding >= 0``.
- Post: ``button.minimumWidth() >= fontMetrics.horizontalAdvance(text)
        + padding``.
- Inv:  the helper never *shrinks* a button below its existing minimum
        width — Qt layout managers may already have set a hint.

DRY
---
A single ``DEFAULT_BUTTON_PADDING_PX`` constant captures the chrome
allowance (border + internal margins). All call sites pick it up
automatically when it changes.
"""

from __future__ import annotations

from typing import Final, TypeVar

from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import QAbstractButton

DEFAULT_BUTTON_PADDING_PX: Final[int] = 24
"""Pixels of chrome (border + internal margin) to add past the text width.

Calibrated against the toolstrip ``_BTN_SMALL`` style which uses
``border:1px`` + ``padding:3px 10px`` (= 22 px combined). The default
of 24 px adds a 2-px buffer so anti-aliased glyphs and any future
border tweaks never clip.
"""


_B = TypeVar("_B", bound=QAbstractButton)


def fit_button_to_text(
    button: _B | None,
    padding: int = DEFAULT_BUTTON_PADDING_PX,
) -> _B:
    """Set ``button.minimumWidth`` so its current text always fits.

    Parameters
    ----------
    button : QAbstractButton
        The button to size. Must not be None.
    padding : int, optional
        Pixels of chrome (border + padding) to add to the text width.
        Defaults to ``DEFAULT_BUTTON_PADDING_PX``. Must be ≥ 0.

    Returns
    -------
    QAbstractButton
        The same button (chained), so call sites can write
        ``self.btn = fit_button_to_text(QPushButton("..."))``.

    Raises
    ------
    ValueError
        If ``button`` is None or ``padding`` is negative.
    """
    if button is None:
        raise ValueError("button must not be None")
    if padding < 0:
        raise ValueError(f"padding must be ≥ 0, got {padding}")

    fm = QFontMetrics(button.font())
    text_width = fm.horizontalAdvance(button.text())
    required = text_width + padding

    # Never shrink: respect any larger hint already set by the caller
    # or by a parent layout manager.
    current = button.minimumWidth()
    if required > current:
        button.setMinimumWidth(required)
    return button
