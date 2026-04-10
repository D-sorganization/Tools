"""
International and Recent Missions
==================================

Recent commercial missions and international (non-US/USSR) space programs:
SpaceX, Artemis, China, India, Japan, and ESA.
"""

from typing import Any

INTERNATIONAL_EVENTS: list[dict[str, Any]] = [
    # Modern missions
    {
        "year": 2020,
        "month": 5,
        "day": 30,
        "title": "SpaceX Crew Dragon Demo-2",
        "description": (
            "First crewed orbital spaceflight from US soil since 2011, "
            "beginning commercial crew era"
        ),
        "category": "mission",
    },
    {
        "year": 2022,
        "month": 11,
        "day": 16,
        "title": "Artemis I Launch",
        "description": (
            "Uncrewed Orion spacecraft orbits the Moon on SLS rocket, "
            "beginning NASA's Artemis program"
        ),
        "category": "mission",
    },
    {
        "year": 2024,
        "month": 2,
        "day": 22,
        "title": "Odysseus Moon Landing",
        "description": (
            "Intuitive Machines' Odysseus becomes first private spacecraft to "
            "land on the Moon"
        ),
        "category": "mission",
    },
    # International missions
    {
        "year": 2003,
        "month": 10,
        "day": 15,
        "title": "China's First Crewed Spaceflight",
        "description": (
            "Yang Liwei orbits Earth on Shenzhou 5, making China the third "
            "nation to independently launch humans"
        ),
        "category": "mission",
    },
    {
        "year": 2004,
        "month": 3,
        "day": 2,
        "title": "Rosetta Launch",
        "description": (
            "ESA launches Rosetta on a 10-year journey to comet 67P/"
            "Churyumov-Gerasimenko"
        ),
        "category": "mission",
    },
    {
        "year": 2010,
        "month": 6,
        "day": 13,
        "title": "Hayabusa Returns Asteroid Samples",
        "description": (
            "JAXA's Hayabusa returns first-ever samples from an asteroid "
            "(25143 Itokawa) to Earth"
        ),
        "category": "mission",
    },
    {
        "year": 2013,
        "month": 11,
        "day": 5,
        "title": "Mangalyaan (Mars Orbiter Mission) Launch",
        "description": (
            "India's ISRO launches its first interplanetary mission to Mars on "
            "a remarkably low budget"
        ),
        "category": "mission",
    },
    {
        "year": 2014,
        "month": 9,
        "day": 24,
        "title": "Mangalyaan Enters Mars Orbit",
        "description": (
            "India becomes the first Asian nation to reach Mars orbit, and the "
            "first to do so on its first attempt"
        ),
        "category": "mission",
    },
    {
        "year": 2019,
        "month": 1,
        "day": 3,
        "title": "Chang'e 4 Lands on Lunar Far Side",
        "description": (
            "China achieves first-ever landing on the far side of the Moon "
            "in Von Karman crater"
        ),
        "category": "mission",
    },
    {
        "year": 2020,
        "month": 12,
        "day": 6,
        "title": "Hayabusa2 Returns Ryugu Samples",
        "description": (
            "JAXA's Hayabusa2 returns 5.4 grams of material from asteroid "
            "Ryugu after a 6-year mission"
        ),
        "category": "mission",
    },
    {
        "year": 2023,
        "month": 8,
        "day": 23,
        "title": "Chandrayaan-3 Lands on Moon",
        "description": (
            "India becomes the fourth nation to land on the Moon and the first "
            "to land near the lunar south pole"
        ),
        "category": "mission",
    },
]
