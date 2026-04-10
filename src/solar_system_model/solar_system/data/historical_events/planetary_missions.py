"""
Planetary Missions
===================

Events from the era of planetary exploration: Mariner, Pioneer, Viking,
Voyager, Shuttle era beginnings, through Cassini's launch in 1997.
"""

from typing import Any

PLANETARY_MISSION_EVENTS: list[dict[str, Any]] = [
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
]
