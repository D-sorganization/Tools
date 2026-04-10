"""
Crewed Spaceflight Programs
============================

Events from Mercury, Gemini, Apollo (missions 7-17), Skylab, and Space
Shuttle firsts.
"""

from typing import Any

CREWED_PROGRAM_EVENTS: list[dict[str, Any]] = [
    # Mercury Program
    {
        "year": 1961,
        "month": 5,
        "day": 5,
        "title": "Freedom 7 - First American in Space",
        "description": (
            "Alan Shepard becomes the first American in space on a 15-minute "
            "suborbital flight"
        ),
        "category": "mission",
    },
    {
        "year": 1962,
        "month": 2,
        "day": 20,
        "title": "Friendship 7 - First American Orbit",
        "description": (
            "John Glenn orbits Earth three times aboard Mercury-Atlas 6, "
            "becoming a national hero"
        ),
        "category": "mission",
    },
    # Gemini Program
    {
        "year": 1965,
        "month": 3,
        "day": 23,
        "title": "Gemini 3 - First Crewed Gemini Flight",
        "description": (
            "Gus Grissom and John Young complete three orbits, testing the "
            "new two-person spacecraft"
        ),
        "category": "mission",
    },
    {
        "year": 1965,
        "month": 6,
        "day": 3,
        "title": "First American Spacewalk",
        "description": (
            "Ed White performs a 23-minute EVA during Gemini 4, first American "
            "to walk in space"
        ),
        "category": "mission",
    },
    {
        "year": 1965,
        "month": 12,
        "day": 15,
        "title": "First Space Rendezvous",
        "description": (
            "Gemini 6A and Gemini 7 rendezvous in orbit, coming within 1 foot "
            "of each other"
        ),
        "category": "mission",
    },
    {
        "year": 1966,
        "month": 3,
        "day": 16,
        "title": "First Space Docking",
        "description": (
            "Gemini 8 with Neil Armstrong performs first docking with an Agena "
            "target vehicle"
        ),
        "category": "mission",
    },
    # Apollo Missions (7-17)
    {
        "year": 1968,
        "month": 10,
        "day": 11,
        "title": "Apollo 7 - First Crewed Apollo",
        "description": (
            "First crewed Apollo mission tests the Command Module in Earth orbit "
            "for 11 days"
        ),
        "category": "mission",
    },
    {
        "year": 1969,
        "month": 3,
        "day": 3,
        "title": "Apollo 9 - Lunar Module Test",
        "description": (
            "First crewed flight of the Lunar Module in Earth orbit, testing "
            "rendezvous procedures"
        ),
        "category": "mission",
    },
    {
        "year": 1969,
        "month": 5,
        "day": 18,
        "title": "Apollo 10 - Lunar Dress Rehearsal",
        "description": (
            "Full dress rehearsal for the Moon landing, Lunar Module descends to "
            "within 9 miles of surface"
        ),
        "category": "mission",
    },
    {
        "year": 1969,
        "month": 11,
        "day": 14,
        "title": "Apollo 12 - Precision Landing",
        "description": (
            "Lands within walking distance of Surveyor 3 probe, demonstrating "
            "precision lunar landing"
        ),
        "category": "mission",
    },
    {
        "year": 1971,
        "month": 1,
        "day": 31,
        "title": "Apollo 14 Launch",
        "description": (
            "Alan Shepard returns to space, later hits golf balls on the Moon"
        ),
        "category": "mission",
    },
    {
        "year": 1971,
        "month": 7,
        "day": 26,
        "title": "Apollo 15 - First Lunar Rover",
        "description": (
            "First use of the Lunar Roving Vehicle, extending exploration range "
            "on the Moon"
        ),
        "category": "mission",
    },
    {
        "year": 1972,
        "month": 4,
        "day": 16,
        "title": "Apollo 16 Launch",
        "description": (
            "John Young and Charles Duke explore the lunar highlands at "
            "Descartes region"
        ),
        "category": "mission",
    },
    {
        "year": 1972,
        "month": 12,
        "day": 7,
        "title": "Apollo 17 - Last Moon Mission",
        "description": (
            "Final Apollo lunar mission; Gene Cernan becomes the last person "
            "to walk on the Moon"
        ),
        "category": "mission",
    },
    # Skylab
    {
        "year": 1973,
        "month": 5,
        "day": 14,
        "title": "Skylab Launch",
        "description": (
            "First American space station launched, hosting three crews over "
            "171 days of occupation"
        ),
        "category": "mission",
    },
    # Space Shuttle Firsts
    {
        "year": 1983,
        "month": 6,
        "day": 18,
        "title": "First American Woman in Space",
        "description": (
            "Sally Ride flies aboard STS-7, becoming the first American woman in space"
        ),
        "category": "mission",
    },
    {
        "year": 1984,
        "month": 2,
        "day": 7,
        "title": "First Untethered Spacewalk",
        "description": (
            "Bruce McCandless performs the first untethered EVA using the Manned "
            "Maneuvering Unit"
        ),
        "category": "mission",
    },
    {
        "year": 1990,
        "month": 12,
        "day": 2,
        "title": "First Crew on Mir from Shuttle",
        "description": (
            "STS-35 demonstrates long-duration shuttle mission capability "
            "with ASTRO-1 observatory"
        ),
        "category": "mission",
    },
]
