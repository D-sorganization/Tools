"""Strict cross-runtime numeric projection for retained launch-monitor scalars."""

from __future__ import annotations

import math
import re
from numbers import Real

_DECIMAL = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


def finite_launch_monitor_scalar(value: object) -> float | None:
    """Return a finite decimal number without accepting booleans or radix text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not _DECIMAL.fullmatch(stripped):
            return None
        parsed = float(stripped)
    elif isinstance(value, Real):
        try:
            parsed = float(value)
        except OverflowError:
            return None
    else:
        return None
    return parsed if math.isfinite(parsed) else None


__all__ = ["finite_launch_monitor_scalar"]
