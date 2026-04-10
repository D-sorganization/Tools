"""
Historical Space Exploration Events
====================================

Database of significant events in space exploration, astronomy,
and planetary science.

Each event includes:
- Date (year, month, day)
- Title
- Description
- Category (mission, discovery, observation, etc.)

This package was originally a single module ``historical_events.py``. It has
been split by era/program into submodules for maintainability, while keeping
the same public API (``SPACE_EVENTS``, ``get_events_for_date``,
``get_events_by_year``, ``get_events_by_category``).
"""

from calendar import monthrange
from datetime import datetime
from typing import Any

from .ancient_and_early import ANCIENT_AND_EARLY_EVENTS
from .crewed_programs import CREWED_PROGRAM_EVENTS
from .international_missions import INTERNATIONAL_EVENTS
from .modern_era import MODERN_ERA_EVENTS
from .planetary_missions import PLANETARY_MISSION_EVENTS

# Combined list of historical space events. Order preserved to match the
# original monolithic module: Ancient/Early -> Planetary -> Modern Era ->
# Crewed Programs -> International/Recent.
SPACE_EVENTS: list[dict[str, Any]] = [
    *ANCIENT_AND_EARLY_EVENTS,
    *PLANETARY_MISSION_EVENTS,
    *MODERN_ERA_EVENTS,
    *CREWED_PROGRAM_EVENTS,
    *INTERNATIONAL_EVENTS,
]


def get_events_for_date(dt: datetime, window_days: int = 3) -> list[dict[str, Any]]:
    """
    Get historical events near a specific date.

    Args:
        dt: The date to search around
        window_days: Number of days before/after to include (default: 3)

    Returns:
        List of matching events
    """
    assert dt is not None, "dt must be provided"
    matching_events = []

    for event in SPACE_EVENTS:
        # Check if month and day match (within window)
        if event["month"] == dt.month:
            day_diff = abs(int(event["day"]) - dt.day)
            if day_diff <= window_days:
                matching_events.append(event)

        # Also check adjacent months if within window
        # Handle month wrapping (December <-> January)
        month_diff = abs(int(event["month"]) - dt.month)
        is_adjacent = (month_diff == 1) or (
            month_diff == 11
        )  # 11 handles Dec->Jan or Jan->Dec

        if is_adjacent:
            # Calculate day difference across month boundary
            # Use calendar module to get actual days in month
            if event["month"] == dt.month + 1 or (
                dt.month == 12 and event["month"] == 1
            ):
                # Event is in next month
                days_in_current = monthrange(dt.year, dt.month)[1]
                day_diff = (days_in_current - dt.day) + int(event["day"])
            else:
                # Event is in previous month
                days_in_event_month = monthrange(dt.year, int(event["month"]))[1]
                day_diff = (days_in_event_month - int(event["day"])) + dt.day

            if day_diff <= window_days:
                matching_events.append(event)

    return matching_events


def get_events_by_year(year: int) -> list[dict[str, Any]]:
    """
    Get all events from a specific year.

    Args:
        year: The year to search

    Returns:
        List of events from that year
    """
    return [event for event in SPACE_EVENTS if event["year"] == year]


def get_events_by_category(category: str) -> list[dict[str, Any]]:
    """
    Get all events of a specific category.

    Args:
        category: Category to filter by

    Returns:
        List of matching events
    """
    return [event for event in SPACE_EVENTS if event["category"] == category]


__all__ = [
    "SPACE_EVENTS",
    "get_events_for_date",
    "get_events_by_year",
    "get_events_by_category",
]
