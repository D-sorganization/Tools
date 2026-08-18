"""Pipeline-mode category and derived-variable policy for the registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from shared.python.contracts import require


def mode_categories(
    delivery: str,
    swing: str,
    club: str,
    ball_setup: str,
    launch: str,
) -> Mapping[str, tuple[str, ...]]:
    """Return the immutable category order for every pipeline mode."""
    return MappingProxyType(
        {
            "delivery": (delivery, club, ball_setup),
            "swing": (swing, delivery, club, ball_setup),
            "launch": (launch,),
        }
    )


def swing_derived_keys(delivery: str) -> tuple[str, ...]:
    """Return delivery values produced by, rather than input to, a swing."""
    return (
        f"{delivery}.clubhead_speed_mps",
        f"{delivery}.club_path_deg",
        f"{delivery}.attack_angle_deg",
    )


def keys_for_mode(
    mode: str,
    categories: Mapping[str, tuple[str, ...]],
    variables_in_category: Callable[[str], tuple[Any, ...]],
    derived_swing_keys: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve ordered registry keys admitted by one pipeline mode."""
    require(mode in categories, "unknown mode", mode)
    keys = [
        definition.key
        for category in categories[mode]
        for definition in variables_in_category(category)
    ]
    if mode == "swing":
        keys = [key for key in keys if key not in derived_swing_keys]
    return tuple(keys)


__all__ = ["keys_for_mode", "mode_categories", "swing_derived_keys"]
