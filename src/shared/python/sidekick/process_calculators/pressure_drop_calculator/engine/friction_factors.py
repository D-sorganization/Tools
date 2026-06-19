"""Friction factor correlations for the pressure drop engine.

.. deprecated::
    This module is a thin compatibility shim. The canonical implementation
    lives in :mod:`._friction_factors`, which the production engine
    (``pressure_drop_calculation_engine``) imports directly. This shim exists
    only so older import paths keep working; do not add logic here.

    Historically a byte-identical twin lived here that diverged from the
    canonical module on the ``Re <= 0`` laminar contract (it silently returned
    the 0.064 default instead of raising — the bug fixed in issue #3103). The
    duplicate has been collapsed into this re-export so there is exactly one
    implementation with the raising contract (issue #3659).
"""

from __future__ import annotations

from ._friction_factors import (
    friction_factor_churchill,
    friction_factor_colebrook,
    friction_factor_haaland,
    friction_factor_laminar,
    friction_factor_swamee_jain,
    select_friction_factor_method,
)

__all__ = [
    "friction_factor_churchill",
    "friction_factor_colebrook",
    "friction_factor_haaland",
    "friction_factor_laminar",
    "friction_factor_swamee_jain",
    "select_friction_factor_method",
]
