"""
Educational Information about Planets
=====================================

Additional descriptive information about solar system bodies
for educational display purposes.
"""

PLANET_DESCRIPTIONS = {
    "Sun": {
        "type": "G-type Main Sequence Star",
        "age": "4.6 billion years",
        "composition": "73% Hydrogen, 25% Helium",
        "core_temperature": "15 million °C",
        "surface_temperature": "5,500 °C",
        "fun_facts": [
            "Contains 99.86% of the solar system's mass",
            "Light takes 8 minutes to reach Earth",
            "Could fit 1.3 million Earths inside",
            "Generates energy through nuclear fusion",
        ],
    },
    "Mercury": {
        "type": "Terrestrial Planet",
        "moons": 0,
        "atmosphere": "Virtually none (exosphere)",
        "surface_features": "Heavily cratered, similar to our Moon",
        "day_length": "176 Earth days",
        "fun_facts": [
            "Smallest planet in the solar system",
            "Has the most eccentric orbit of any planet",
            "Surface temperatures range from -180°C to 430°C",
            "Named after the Roman messenger god",
        ],
    },
    "Venus": {
        "type": "Terrestrial Planet",
        "moons": 0,
        "atmosphere": "96% CO2, extremely dense",
        "surface_features": "Volcanic plains, highland regions",
        "surface_pressure": "92 times Earth's",
        "fun_facts": [
            "Hottest planet due to greenhouse effect",
            "Rotates backwards (retrograde)",
            "Day is longer than its year",
            "Called Earth's 'sister planet' due to similar size",
        ],
    },
    "Earth": {
        "type": "Terrestrial Planet",
        "moons": 1,
        "atmosphere": "78% Nitrogen, 21% Oxygen",
        "surface_features": "70% water, continents",
        "magnetic_field": "Strong, protects from solar wind",
        "fun_facts": [
            "Only known planet with life",
            "Has plate tectonics",
            "Tilted 23.5° causing seasons",
            "Moon stabilizes Earth's rotation",
        ],
    },
    "Mars": {
        "type": "Terrestrial Planet",
        "moons": 2,
        "moon_names": ["Phobos", "Deimos"],
        "atmosphere": "95% CO2, very thin",
        "surface_features": "Largest volcano and canyon in solar system",
        "notable_features": ["Olympus Mons", "Valles Marineris"],
        "fun_facts": [
            "Called the Red Planet due to iron oxide",
            "Has seasons like Earth",
            "Olympus Mons is 3x taller than Everest",
            "Target for human colonization",
        ],
    },
    "Jupiter": {
        "type": "Gas Giant",
        "moons": 95,
        "notable_moons": ["Io", "Europa", "Ganymede", "Callisto"],
        "atmosphere": "90% Hydrogen, 10% Helium",
        "notable_features": ["Great Red Spot", "Bands and zones"],
        "ring_system": "Faint, made of dust",
        "fun_facts": [
            "Largest planet - could fit all others inside",
            "Great Red Spot is a 400-year-old storm",
            "Has the strongest magnetic field",
            "Ganymede is larger than Mercury",
        ],
    },
    "Saturn": {
        "type": "Gas Giant",
        "moons": 146,
        "notable_moons": ["Titan", "Enceladus", "Mimas", "Rhea"],
        "atmosphere": "96% Hydrogen, 3% Helium",
        "ring_system": "Most spectacular, made of ice and rock",
        "ring_span": "282,000 km wide, only 10m thick",
        "fun_facts": [
            "Could float in water (if you had a big enough bathtub)",
            "Rings would stretch from Earth to Moon",
            "Titan has lakes of liquid methane",
            "Hexagonal storm at north pole",
        ],
    },
    "Uranus": {
        "type": "Ice Giant",
        "moons": 28,
        "notable_moons": ["Miranda", "Ariel", "Umbriel", "Titania", "Oberon"],
        "atmosphere": "83% Hydrogen, 15% Helium, 2% Methane",
        "ring_system": "13 faint rings",
        "axial_tilt": "97.77° - rolls around the Sun",
        "fun_facts": [
            "Rotates on its side",
            "Coldest planetary atmosphere (-224°C)",
            "Methane gives it blue-green color",
            "First planet discovered with telescope (1781)",
        ],
    },
    "Neptune": {
        "type": "Ice Giant",
        "moons": 16,
        "notable_moons": ["Triton"],
        "atmosphere": "80% Hydrogen, 19% Helium, 1% Methane",
        "notable_features": ["Great Dark Spot", "Strongest winds"],
        "wind_speeds": "Up to 2,100 km/h",
        "fun_facts": [
            "Windiest planet with supersonic storms",
            "Triton orbits backwards - captured moon",
            "Takes 165 years to orbit the Sun",
            "Predicted mathematically before being seen",
        ],
    },
    "Pluto": {
        "type": "Dwarf Planet",
        "moons": 5,
        "notable_moons": ["Charon"],
        "surface_features": "Heart-shaped glacier (Tombaugh Regio)",
        "atmosphere": "Thin, nitrogen and methane",
        "location": "Kuiper Belt",
        "fun_facts": [
            "Reclassified as dwarf planet in 2006",
            "Charon is half its size - they orbit each other",
            "Has blue skies and red water ice",
            "New Horizons flew by in 2015",
        ],
    },
}

TRANSFER_INFO = {
    "Earth-Mars": {
        "typical_duration": "6-9 months",
        "launch_windows": "Every 26 months",
        "delta_v": "~3.6 km/s from LEO",
        "notable_missions": [
            "Mariner 4 (1964) - First successful flyby",
            "Viking 1 & 2 (1976) - First successful landers",
            "Mars Pathfinder (1997) - First rover",
            "Curiosity (2012) - Ongoing exploration",
            "Perseverance (2021) - Sample return preparation",
        ],
    },
    "Earth-Venus": {
        "typical_duration": "4-5 months",
        "launch_windows": "Every 19 months",
        "delta_v": "~3.5 km/s from LEO",
        "notable_missions": [
            "Venera 7 (1970) - First successful landing",
            "Magellan (1990) - Radar mapping",
            "Venus Express (2006) - Atmospheric study",
        ],
    },
    "Earth-Jupiter": {
        "typical_duration": "2-6 years",
        "delta_v": "~6 km/s (with gravity assists)",
        "notable_missions": [
            "Pioneer 10 (1973) - First flyby",
            "Voyager 1 & 2 (1979) - Detailed imaging",
            "Galileo (1995) - First orbiter",
            "Juno (2016) - Ongoing study",
        ],
    },
}

ORBITAL_MECHANICS_GLOSSARY = {
    "Apoapsis": "The point in an orbit farthest from the body being orbited",
    "Periapsis": "The point in an orbit closest to the body being orbited",
    "Aphelion": "Apoapsis when orbiting the Sun",
    "Perihelion": "Periapsis when orbiting the Sun",
    "Semi-major axis": "Half the longest diameter of an elliptical orbit",
    "Eccentricity": "Measure of how elliptical an orbit is (0=circle, 1=parabola)",
    "Inclination": "Angle between orbital plane and reference plane",
    "Ascending node": "Where orbit crosses reference plane going north",
    "Argument of periapsis": "Angle from ascending node to periapsis",
    "True anomaly": "Current angular position in orbit",
    "Mean anomaly": "Fraction of orbital period elapsed, as an angle",
    "Eccentric anomaly": "Auxiliary angle used in Kepler's equation",
    "Hohmann transfer": "Most fuel-efficient transfer between circular orbits",
    "Delta-v": "Change in velocity required for a maneuver",
    "Synodic period": "Time between successive alignments of two bodies",
    "Sphere of influence": "Region where a body's gravity dominates",
}
