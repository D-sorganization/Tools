"""
Historical Space Exploration Events — Query API
================================================

Query helpers over the :data:`SPACE_EVENTS` database of significant
events in space exploration, astronomy, and planetary science.

Each event in the database includes:

- ``year``, ``month``, ``day`` — calendar date
- ``title`` — short name
- ``description`` — one-sentence summary
- ``category`` — e.g. ``mission``, ``discovery``, ``observation``

The event data itself lives in :mod:`solar_system.data.space_events_data`
(extracted for file-size budget, issue #2152). ``SPACE_EVENTS`` is
re-exported from this module so existing callers keep working.
"""

from calendar import monthrange
from datetime import datetime
from typing import Any

from .space_events_data import SPACE_EVENTS

__all__ = [
    "SPACE_EVENTS",
    "get_events_by_category",
    "get_events_by_year",
    "get_events_for_date",
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
    if not (dt is not None):
        raise ValueError("dt must be provided")
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
