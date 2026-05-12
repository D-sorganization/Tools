"""Overlay-state synchronization between the toolstrip and pendulum widgets.

When the user switches between pendulum models (Double / Triple / Golfer),
the new model's pendulum widget needs to inherit every overlay toggle and
slider value the user has already configured on the toolstrip. Without
this, freshly-shown widgets render with their *constructor* defaults
(everything off) and the user has to cycle each checkbox to "wake up"
the new model.

This module is the single source of truth for that mapping. Both the
model-switch handler in ``panel_builders`` and any future code that
re-applies state (e.g. saved-session restore) call exactly one
function — ``apply_toolstrip_overlay_state`` — and any new overlay
toggle is added in exactly one place.

Design by Contract
------------------
- Pre:  ``toolstrip`` and ``pendulum`` are non-None Qt widgets.
- Post: every overlay setter on ``pendulum`` that has a corresponding
        toolstrip control has been invoked exactly once with the
        toolstrip's current value. Setters absent from the pendulum
        widget are silently skipped (LOD: the helper does not assume
        anything about the concrete widget class).

Law of Demeter
--------------
This helper only reaches into ``toolstrip``'s public-by-convention
attributes (``chk_*``, ``_sld_*``) and ``pendulum``'s public ``set_*``
setters. Callers are not aware of any of these names — they pass two
opaque widgets and trust the helper.

DRY
---
The mapping below is the *only* place that knows which toolstrip
control drives which pendulum setter. The toolstrip's signal-emit
handlers and this snapshot helper agree by construction (both walk
the same widgets), so adding a new overlay is a one-line change here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PyQt6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


# Each entry: (toolstrip attribute, pendulum setter, value extractor)
# - boolean overlays: extractor = lambda w: w.isChecked()
# - scale sliders:    extractor = lambda w: w.value() / 10.0


def _checked(w: Any) -> bool:
    return bool(w.isChecked())


def _scale_value(w: Any) -> float:
    """Read a slider's display scale, honouring its ``scale_divisor`` property.

    The slider stores its raw→display divisor as a Qt property when it
    is constructed by ``_make_scale_slider``. We honour that property
    so a slider that maps raw 1..1000 to 0.01..10× returns the right
    display value here, no matter what the divisor is.
    """
    divisor = w.property("scale_divisor")
    if not divisor:
        divisor = 10
    return float(w.value()) / float(divisor)


_OVERLAY_BINDINGS: tuple[tuple[str, str, Callable[[Any], object]], ...] = (
    # Boolean overlays
    ("chk_forces", "set_show_forces", _checked),
    ("chk_zero_torque", "set_show_zero_torque_forces", _checked),
    ("chk_mob", "set_show_mob_ellipsoids", _checked),
    ("chk_force_ell", "set_show_force_ellipsoids", _checked),
    ("chk_com", "set_show_com", _checked),
    ("chk_torque", "set_show_torque_vectors", _checked),
    ("chk_mof", "set_show_moment_of_force", _checked),
    ("chk_sum_moments", "set_show_sum_moments", _checked),
    ("chk_3d", "set_3d_mode", _checked),
    # Scale sliders (raw value /10 = display scale)
    ("_sld_force", "set_force_scale", _scale_value),
    ("_sld_mob", "set_mob_ellipsoid_scale", _scale_value),
    ("_sld_force_ell", "set_force_ellipsoid_scale", _scale_value),
)


def apply_toolstrip_overlay_state(
    toolstrip: QWidget,
    pendulum: QWidget,
) -> None:
    """Push every overlay toggle and scale value from ``toolstrip`` to ``pendulum``.

    Parameters
    ----------
    toolstrip : QWidget
        The application toolstrip (must expose the ``chk_*`` and ``_sld_*``
        attributes; in practice this is always ``ToolStrip``).
    pendulum : QWidget
        The pendulum visualization widget on the active panel
        (``PendulumWidget`` or ``GolferPendulumWidget``).

    Raises
    ------
    ValueError
        If either argument is None.
    """
    if toolstrip is None:
        raise ValueError("toolstrip must not be None")
    if pendulum is None:
        raise ValueError("pendulum must not be None")

    for src_attr, dst_setter, extract in _OVERLAY_BINDINGS:
        src = getattr(toolstrip, src_attr, None)
        if src is None:
            logger.debug(
                "toolstrip has no attribute %r; skipping %s", src_attr, dst_setter
            )
            continue
        setter = getattr(pendulum, dst_setter, None)
        if setter is None:
            # The pendulum widget doesn't implement this overlay (e.g.
            # a hypothetical read-only model). Silently skip — the LOD
            # contract says we don't assume anything about the widget.
            continue
        try:
            value = extract(src)
        except Exception as exc:  # noqa: BLE001
            logger.debug("extractor for %s failed (%s); skipping", src_attr, exc)
            continue
        try:
            setter(value)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "%s.%s(%r) raised (%s); skipping",
                type(pendulum).__name__,
                dst_setter,
                value,
                exc,
            )
