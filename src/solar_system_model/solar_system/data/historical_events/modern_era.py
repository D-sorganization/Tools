"""
Modern Era
===========

2000-present events: ISS habitation, Mars rovers, JWST, Artemis, and other
contemporary missions along with some additional historical entries that were
appended after the initial Modern Era section.
"""

from typing import Any

MODERN_ERA_EVENTS: list[dict[str, Any]] = [
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
        "month": 4,
        "day": 14,
        "title": "JUICE Mission Launch",
        "description": (
            "ESA mission launches to study Jupiter's icy moons: Ganymede,"
            " Callisto, and Europa"
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
    {
        "year": 2024,
        "month": 10,
        "day": 14,
        "title": "Europa Clipper Launch",
        "description": "NASA's major mission to determine if Europa could support life",
        "category": "mission",
    },
    {
        "year": 1958,
        "month": 3,
        "day": 17,
        "title": "Vanguard 1 Launch",
        "description": "Oldest artificial satellite still in Earth orbit today",
        "category": "mission",
    },
    {
        "year": 1990,
        "month": 2,
        "day": 14,
        "title": "Pale Blue Dot Photo",
        "description": (
            "Voyager 1 takes the famous photo of Earth from 6 billion km away"
        ),
        "category": "observation",
    },
    {
        "year": 1994,
        "month": 7,
        "day": 16,
        "title": "Comet Shoemaker-Levy 9 Impacts Jupiter",
        "description": (
            "First time scientists observe a collision between two solar system bodies"
        ),
        "category": "discovery",
    },
    {
        "year": 2005,
        "month": 7,
        "day": 4,
        "title": "Deep Impact Collides with Comet",
        "description": "Smart Impactor hits Comet Tempel 1 to study its composition",
        "category": "mission",
    },
    {
        "year": 2015,
        "month": 3,
        "day": 6,
        "title": "Dawn Arrives at Ceres",
        "description": "First spacecraft to visit and orbit a dwarf planet",
        "category": "mission",
    },
    {
        "year": 2018,
        "month": 2,
        "day": 6,
        "title": "Falcon Heavy Maiden Flight",
        "description": (
            "World's most powerful operational rocket launches Starman into solar orbit"
        ),
        "category": "mission",
    },
]
