"""Versioned persistence contract for the primary PyQt workspace tabs."""

from __future__ import annotations

import json
import logging
from typing import Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TAB_IDS",
    "NAVIGATION_SETTINGS_APP",
    "NAVIGATION_SETTINGS_ORG",
    "NAVIGATION_STATE_KEY",
    "NavigationSettings",
    "TAB_HELP_KEYS",
    "decode_navigation_state",
    "encode_navigation_state",
]

#: Stable IDs for primary tabs in their first-run order.
DEFAULT_TAB_IDS: tuple[str, ...] = (
    "clubhead",
    "plots",
    "calculation_description",
    "simulation",
    "flight_explorer",
    "regional_surfaces",
    "regional_ground_execution",
    "ground_playback",
    "launch_monitor_analytics",
    "capability_optimization",
    "variation",
    "putting",
    "glossary",
)
TAB_HELP_KEYS = DEFAULT_TAB_IDS
NAVIGATION_SETTINGS_ORG = "D-sorganization"
NAVIGATION_SETTINGS_APP = "RateOfClosureImpactExplorer"
NAVIGATION_STATE_KEY = "ui/primary-tabs/v1"
_NAVIGATION_STATE_VERSION = 1


class NavigationSettings(Protocol):
    """Minimal settings seam required by primary-tab persistence."""

    def value(self, key: str, default_value: object = None) -> object:
        """Return a persisted value."""

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        """Persist a value."""


def decode_navigation_state(raw: object) -> tuple[list[str], str | None] | None:
    """Validate and normalize a persisted primary-navigation payload.

    Invalid versions and shapes fail closed. Partial legacy orders retain their
    valid unique IDs and append every missing current tab in default order.
    """
    if not isinstance(raw, str):
        return None
    try:
        state = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Ignoring corrupt primary-tab navigation state")
        return None
    if not isinstance(state, dict) or state.get("version") != _NAVIGATION_STATE_VERSION:
        return None
    supplied = state.get("order")
    if not isinstance(supplied, list):
        supplied = []
    order = list(
        dict.fromkeys(tab_id for tab_id in supplied if tab_id in DEFAULT_TAB_IDS)
    )
    order.extend(tab_id for tab_id in DEFAULT_TAB_IDS if tab_id not in order)
    active = state.get("active")
    return order, active if active in DEFAULT_TAB_IDS else None


def encode_navigation_state(order: list[str], active: str) -> str:
    """Serialize a validated primary-tab order and active stable ID."""
    return json.dumps(
        {
            "version": _NAVIGATION_STATE_VERSION,
            "order": order,
            "active": active,
        }
    )
