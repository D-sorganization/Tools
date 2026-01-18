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
"""

from calendar import monthrange
from datetime import datetime
from typing import Any

# List of historical space events
SPACE_EVENTS: list[dict[str, Any]] = [
    # Ancient Astronomy
    {
        "year": 1610,
        "month": 1,
        "day": 7,
        "title": "Galileo Discovers Jupiter's Moons",
        "description": (
            "Galileo Galilei observes four moons orbiting Jupiter, providing evidence"
            " for Copernican heliocentrism"
        ),
        "category": "discovery",
    },
    # Early Space Age
    {
        "year": 1957,
        "month": 10,
        "day": 4,
        "title": "Sputnik 1 Launched",
        "description": (
            "Soviet Union launches first artificial satellite, beginning the Space Age"
        ),
        "category": "mission",
    },
    {
        "year": 1961,
        "month": 4,
        "day": 12,
        "title": "First Human in Space",
        "description": (
            "Yuri Gagarin becomes the first human to orbit Earth aboard Vostok 1"
        ),
        "category": "mission",
    },
    # Apollo Program
    {
        "year": 1968,
        "month": 12,
        "day": 21,
        "title": "Apollo 8 Launch",
        "description": (
            "First crewed mission to orbit the Moon, capturing the famous 'Earthrise'"
            " photo"
        ),
        "category": "mission",
    },
    {
        "year": 1969,
        "month": 7,
        "day": 16,
        "title": "Apollo 11 Launch",
        "description": (
            "Saturn V rocket launches with Neil Armstrong, Buzz Aldrin, and Michael"
            " Collins"
        ),
        "category": "mission",
    },
    {
        "year": 1969,
        "month": 7,
        "day": 20,
        "title": "First Moon Landing",
        "description": (
            "Apollo 11 lands in the Sea of Tranquility. 'That's one small step for"
            " man...'"
        ),
        "category": "mission",
    },
    {
        "year": 1969,
        "month": 7,
        "day": 21,
        "title": "First Moonwalk",
        "description": (
            "Neil Armstrong and Buzz Aldrin walk on the lunar surface for 2.5 hours"
        ),
        "category": "mission",
    },
    {
        "year": 1969,
        "month": 7,
        "day": 24,
        "title": "Apollo 11 Returns",
        "description": (
            "Safe splashdown in Pacific Ocean, completing historic moon mission"
        ),
        "category": "mission",
    },
    {
        "year": 1970,
        "month": 4,
        "day": 11,
        "title": "Apollo 13 Launch",
        "description": (
            "Launch of Apollo 13, which would face a critical in-flight emergency"
        ),
        "category": "mission",
    },
    {
        "year": 1970,
        "month": 4,
        "day": 13,
        "title": "Apollo 13 Accident",
        "description": (
            "'Houston, we've had a problem' - oxygen tank explosion forces mission"
            " abort"
        ),
        "category": "mission",
    },
    {
        "year": 1970,
        "month": 4,
        "day": 17,
        "title": "Apollo 13 Safe Return",
        "description": (
            "Crew safely returns to Earth after using lunar module as 'lifeboat'"
        ),
        "category": "mission",
    },
    # Planetary Missions
    {
        "year": 1971,
        "month": 11,
        "day": 13,
        "title": "Mariner 9 Reaches Mars",
        "description": (
            "First spacecraft to orbit another planet, maps 85% of Mars surface"
        ),
        "category": "mission",
    },
    {
        "year": 1973,
        "month": 12,
        "day": 3,
        "title": "Pioneer 10 at Jupiter",
        "description": "First spacecraft flyby of Jupiter, returning close-up images",
        "category": "mission",
    },
    {
        "year": 1976,
        "month": 7,
        "day": 20,
        "title": "Viking 1 Lands on Mars",
        "description": "First successful Mars landing, begins search for life",
        "category": "mission",
    },
    {
        "year": 1977,
        "month": 8,
        "day": 20,
        "title": "Voyager 2 Launch",
        "description": "Launches on grand tour of outer planets",
        "category": "mission",
    },
    {
        "year": 1977,
        "month": 9,
        "day": 5,
        "title": "Voyager 1 Launch",
        "description": "Launches on fast trajectory to Jupiter and Saturn",
        "category": "mission",
    },
    {
        "year": 1979,
        "month": 3,
        "day": 5,
        "title": "Voyager 1 at Jupiter",
        "description": "Discovers active volcanoes on Io, first found beyond Earth",
        "category": "discovery",
    },
    {
        "year": 1980,
        "month": 11,
        "day": 12,
        "title": "Voyager 1 at Saturn",
        "description": "Close flyby reveals complex ring structure and moon details",
        "category": "mission",
    },
    {
        "year": 1981,
        "month": 4,
        "day": 12,
        "title": "First Space Shuttle Launch",
        "description": "Columbia launches on STS-1, beginning reusable spacecraft era",
        "category": "mission",
    },
    {
        "year": 1986,
        "month": 1,
        "day": 24,
        "title": "Voyager 2 at Uranus",
        "description": (
            "First and only spacecraft visit to Uranus, discovers 10 new moons"
        ),
        "category": "mission",
    },
    {
        "year": 1986,
        "month": 2,
        "day": 9,
        "title": "Halley's Comet Return",
        "description": (
            "Armada of spacecraft from multiple nations study the famous comet"
        ),
        "category": "observation",
    },
    {
        "year": 1989,
        "month": 8,
        "day": 25,
        "title": "Voyager 2 at Neptune",
        "description": (
            "Completes grand tour, discovers Great Dark Spot and active geysers on "
            "Triton"
        ),
        "category": "mission",
    },
    {
        "year": 1990,
        "month": 4,
        "day": 24,
        "title": "Hubble Space Telescope Launch",
        "description": (
            "Revolutionary space observatory deployed by Space Shuttle Discovery"
        ),
        "category": "mission",
    },
    {
        "year": 1995,
        "month": 12,
        "day": 7,
        "title": "Galileo Arrives at Jupiter",
        "description": "Begins multi-year study of Jupiter and its moons",
        "category": "mission",
    },
    {
        "year": 1997,
        "month": 7,
        "day": 4,
        "title": "Mars Pathfinder Lands",
        "description": "Delivers Sojourner rover, first wheeled vehicle on Mars",
        "category": "mission",
    },
    {
        "year": 1997,
        "month": 10,
        "day": 15,
        "title": "Cassini Launch",
        "description": "Begins journey to Saturn carrying Huygens probe",
        "category": "mission",
    },
    # Modern Era
    {
        "year": 2000,
        "month": 11,
        "day": 2,
        "title": "ISS Continuous Occupation Begins",
        "description": (
            "First crew arrives at International Space Station for permanent habitation"
        ),
        "category": "mission",
    },
    {
        "year": 2004,
        "month": 1,
        "day": 3,
        "title": "Spirit Rover Lands on Mars",
        "description": "First of twin rovers begins exploring Gusev Crater",
        "category": "mission",
    },
    {
        "year": 2004,
        "month": 1,
        "day": 24,
        "title": "Opportunity Rover Lands",
        "description": (
            "Second rover lands on opposite side of Mars, will operate for 15 years"
        ),
        "category": "mission",
    },
    {
        "year": 2004,
        "month": 7,
        "day": 1,
        "title": "Cassini Enters Saturn Orbit",
        "description": "Begins comprehensive study of Saturn system",
        "category": "mission",
    },
    {
        "year": 2005,
        "month": 1,
        "day": 14,
        "title": "Huygens Lands on Titan",
        "description": "First landing on a moon in the outer solar system",
        "category": "mission",
    },
    {
        "year": 2006,
        "month": 1,
        "day": 19,
        "title": "New Horizons Launch",
        "description": "Begins 9-year journey to Pluto and beyond",
        "category": "mission",
    },
    {
        "year": 2006,
        "month": 8,
        "day": 24,
        "title": "Pluto Reclassified",
        "description": "IAU defines 'planet', reclassifying Pluto as a dwarf planet",
        "category": "discovery",
    },
    {
        "year": 2011,
        "month": 7,
        "day": 21,
        "title": "Final Space Shuttle Mission",
        "description": "Atlantis completes STS-135, ending 30-year shuttle program",
        "category": "mission",
    },
    {
        "year": 2012,
        "month": 8,
        "day": 6,
        "title": "Curiosity Rover Lands on Mars",
        "description": "Car-sized rover successfully lands using sky crane technique",
        "category": "mission",
    },
    {
        "year": 2012,
        "month": 8,
        "day": 25,
        "title": "Voyager 1 Enters Interstellar Space",
        "description": "First human-made object to leave the solar system",
        "category": "mission",
    },
    {
        "year": 2014,
        "month": 11,
        "day": 12,
        "title": "Philae Lands on Comet",
        "description": "Rosetta's lander touches down on Comet 67P",
        "category": "mission",
    },
    {
        "year": 2015,
        "month": 7,
        "day": 14,
        "title": "New Horizons at Pluto",
        "description": "First close-up images reveal heart-shaped Tombaugh Regio",
        "category": "mission",
    },
    {
        "year": 2016,
        "month": 7,
        "day": 4,
        "title": "Juno Arrives at Jupiter",
        "description": "Begins detailed study of Jupiter's interior and atmosphere",
        "category": "mission",
    },
    {
        "year": 2018,
        "month": 11,
        "day": 26,
        "title": "InSight Lands on Mars",
        "description": "First mission to study Mars' deep interior with seismometer",
        "category": "mission",
    },
    {
        "year": 2019,
        "month": 1,
        "day": 1,
        "title": "New Horizons at Arrokoth",
        "description": (
            "Flyby of most distant object ever visited - pristine Kuiper Belt object"
        ),
        "category": "mission",
    },
    {
        "year": 2020,
        "month": 7,
        "day": 30,
        "title": "Mars 2020 Launch",
        "description": (
            "Perseverance rover and Ingenuity helicopter begin journey to Mars"
        ),
        "category": "mission",
    },
    {
        "year": 2021,
        "month": 2,
        "day": 18,
        "title": "Perseverance Lands on Mars",
        "description": "Most advanced Mars rover lands in Jezero Crater",
        "category": "mission",
    },
    {
        "year": 2021,
        "month": 4,
        "day": 19,
        "title": "First Mars Helicopter Flight",
        "description": "Ingenuity achieves first powered flight on another planet",
        "category": "mission",
    },
    {
        "year": 2021,
        "month": 12,
        "day": 25,
        "title": "James Webb Space Telescope Launch",
        "description": "Most powerful space telescope ever built begins journey to L2",
        "category": "mission",
    },
    {
        "year": 2022,
        "month": 7,
        "day": 12,
        "title": "First JWST Images Released",
        "description": (
            "Revolutionary infrared telescope reveals deepest view of universe"
        ),
        "category": "observation",
    },
    {
        "year": 2022,
        "month": 9,
        "day": 26,
        "title": "DART Impact Success",
        "description": (
            "First planetary defense test successfully alters asteroid orbit"
        ),
        "category": "mission",
    },
    {
        "year": 2023,
        "month": 9,
        "day": 24,
        "title": "OSIRIS-REx Sample Return",
        "description": "Returns first asteroid sample from Bennu to Earth",
        "category": "mission",
    },
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
