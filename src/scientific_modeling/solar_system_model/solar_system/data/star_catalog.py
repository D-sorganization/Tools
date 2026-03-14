"""Bright star catalog for accurate sky rendering.

The data below is curated from the public-domain HYG Database 3.0 and
Hipparcos catalog entries so we can draw recognizable constellations in the
sky dome. Magnitudes are visual apparent magnitudes and coordinates use the
J2000.0 epoch.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class StarEntry:
    """Single bright star descriptor."""

    name: str
    constellation: str
    ra_hours: float
    dec_degrees: float
    magnitude: float
    bv_index: float


STAR_CATALOG: list[StarEntry] = [
    StarEntry("Sirius", "Canis Major", 6.752481, -16.716116, -1.46, 0.00),
    StarEntry("Canopus", "Carina", 6.399203, -52.695661, -0.72, 0.15),
    StarEntry("Arcturus", "Boötes", 14.261208, 19.182417, -0.05, 1.23),
    StarEntry("Alpha Centauri", "Centaurus", 14.659969, -60.833993, -0.27, 0.71),
    StarEntry("Vega", "Lyra", 18.615649, 38.783692, 0.03, 0.00),
    StarEntry("Capella", "Auriga", 5.278155, 45.997991, 0.08, 0.80),
    StarEntry("Rigel", "Orion", 5.242298, -8.201640, 0.12, -0.03),
    StarEntry("Procyon", "Canis Minor", 7.655033, 5.225016, 0.38, 0.42),
    StarEntry("Achernar", "Eridanus", 1.628571, -57.236757, 0.46, -0.16),
    StarEntry("Betelgeuse", "Orion", 5.919529, 7.407064, 0.50, 1.85),
    StarEntry("Hadar", "Centaurus", 14.063724, -60.372978, 0.61, -0.17),
    StarEntry("Altair", "Aquila", 19.846389, 8.868322, 0.77, 0.22),
    StarEntry("Aldebaran", "Taurus", 4.598667, 16.509302, 0.86, 1.54),
    StarEntry("Spica", "Virgo", 13.419883, -11.161310, 0.98, -0.23),
    StarEntry("Antares", "Scorpius", 16.490128, -26.431946, 1.06, 1.83),
    StarEntry("Pollux", "Gemini", 7.755263, 28.026199, 1.14, 1.02),
    StarEntry("Fomalhaut", "Piscis Austrinus", 22.960848, -29.622236, 1.16, 0.09),
    StarEntry("Deneb", "Cygnus", 20.690531, 45.280338, 1.25, 0.09),
    StarEntry("Regulus", "Leo", 10.139530, 11.967207, 1.35, -0.12),
    StarEntry("Adhara", "Canis Major", 6.977096, -28.972084, 1.50, -0.21),
    StarEntry("Shaula", "Scorpius", 17.560146, -37.103821, 1.62, -0.22),
    StarEntry("Castor", "Gemini", 7.576667, 31.888283, 1.58, 0.00),
    StarEntry("Gacrux", "Crux", 12.519434, -57.113214, 1.64, 1.60),
    StarEntry("Bellatrix", "Orion", 5.418850, 6.349702, 1.64, -0.03),
    StarEntry("Elnath", "Taurus", 5.438198, 28.607450, 1.65, -0.13),
    StarEntry("Alnair", "Grus", 22.137216, -46.960975, 1.73, -0.03),
    StarEntry("Dubhe", "Ursa Major", 11.062129, 61.750873, 1.79, 1.07),
    StarEntry("Menkalinan", "Auriga", 5.992155, 44.947432, 1.90, 0.03),
    StarEntry("Mirfak", "Perseus", 3.405374, 49.861220, 1.82, 0.48),
    StarEntry("Peacock", "Pavo", 20.427458, -56.735090, 1.94, -0.21),
    StarEntry("Polaris", "Ursa Minor", 2.530301, 89.264109, 1.98, 0.60),
    StarEntry("Saiph", "Orion", 5.795941, -9.669604, 2.07, -0.24),
    StarEntry("Algol", "Perseus", 3.136147, 40.955648, 2.12, 0.00),
    StarEntry("Mizar", "Ursa Major", 13.398725, 54.925361, 2.23, 0.04),
    StarEntry("Alphard", "Hydra", 9.459790, -8.658603, 1.98, 1.44),
    StarEntry("Sadr", "Cygnus", 20.370472, 40.256679, 2.23, 0.70),
    StarEntry("Hamal", "Aries", 2.119555, 23.462778, 2.00, 1.16),
    StarEntry("Rasalhague", "Ophiuchus", 17.582241, 12.560070, 2.08, 0.17),
    StarEntry("Kochab", "Ursa Minor", 14.845109, 74.155497, 2.07, 1.15),
    StarEntry("Acrux", "Crux", 12.443304, -63.099092, 0.77, -0.24),
    StarEntry("Mimosa", "Crux", 12.795355, -59.688764, 1.25, -0.23),
    StarEntry("Alioth", "Ursa Major", 12.900485, 55.959843, 1.76, 0.00),
    StarEntry("Alnitak", "Orion", 5.679313, -1.942572, 1.74, -0.20),
    StarEntry("Alnilam", "Orion", 5.603559, -1.201917, 1.69, -0.19),
    StarEntry("Mintaka", "Orion", 5.533444, -0.299091, 2.23, -0.17),
    StarEntry("Denebola", "Leo", 11.817763, 14.572063, 2.14, 0.09),
    StarEntry("Algieba", "Leo", 10.332873, 19.841489, 2.28, 1.18),
    StarEntry("Eltanin", "Draco", 17.943436, 51.488895, 2.23, 1.52),
    StarEntry("Rastaban", "Draco", 17.507284, 52.301389, 2.79, 1.02),
    StarEntry("Alderamin", "Cepheus", 21.309664, 62.585606, 2.45, 0.48),
    StarEntry("Alfirk", "Cepheus", 21.477661, 70.560716, 3.23, 0.16),
    StarEntry("Schedar", "Cassiopeia", 0.675122, 56.537331, 2.24, 1.18),
    StarEntry("Caph", "Cassiopeia", 0.152978, 59.149781, 2.28, 0.48),
    StarEntry("Alpheratz", "Andromeda", 0.139794, 29.090429, 2.06, -0.02),
    StarEntry("Mirach", "Andromeda", 1.162195, 35.620556, 2.07, 1.52),
    StarEntry("Almach", "Andromeda", 2.064984, 42.329725, 2.10, 0.89),
    StarEntry("Markab", "Pegasus", 23.079348, 15.205265, 2.48, -0.06),
    StarEntry("Enif", "Pegasus", 21.736429, 9.875008, 2.39, 1.52),
    StarEntry("Scheat", "Pegasus", 23.062879, 28.082802, 2.44, 1.56),
    StarEntry("Algenib", "Pegasus", 0.220598, 15.183608, 2.84, -0.20),
    StarEntry("Diphda", "Cetus", 0.726486, -17.986689, 2.04, 0.95),
    StarEntry("Menkent", "Centaurus", 14.111374, -36.369954, 2.06, 1.10),
    StarEntry("Zubenelgenubi", "Libra", 14.847982, -16.041783, 2.75, 0.10),
    StarEntry("Kaus Australis", "Sagittarius", 18.402869, -34.384616, 1.79, 0.03),
    StarEntry("Nunki", "Sagittarius", 18.921090, -26.296722, 2.05, -0.10),
    StarEntry("Sargas", "Scorpius", 17.621971, -42.997821, 1.86, 0.57),
    StarEntry("Gienah", "Cygnus", 20.770189, 33.970257, 2.48, -0.01),
    StarEntry("Albireo", "Cygnus", 19.512022, 27.959692, 3.05, 1.18),
    StarEntry("Zeta Ophiuchi", "Ophiuchus", 16.619320, -10.567101, 2.56, -0.26),
    StarEntry("Sabik", "Ophiuchus", 17.172966, -15.724919, 2.43, 0.30),
    StarEntry("Furud", "Canis Major", 6.338630, -17.955918, 3.02, -0.22),
    StarEntry("Wezen", "Canis Major", 7.139860, -26.393199, 1.83, 0.14),
    StarEntry("Avior", "Carina", 8.375236, -59.509492, 1.86, 1.24),
    StarEntry("Aspidiske", "Carina", 9.284840, -59.275269, 1.86, -0.13),
    StarEntry("Miaplacidus", "Carina", 9.220100, -69.717207, 1.67, -0.10),
    StarEntry("Atria", "Triangulum Australe", 16.811079, -69.027679, 1.92, 1.24),
    StarEntry("Alkaid", "Ursa Major", 13.792340, 49.313302, 1.86, -0.03),
    StarEntry("Phad", "Ursa Major", 11.897168, 53.694759, 2.44, -0.04),
    StarEntry("Merak", "Ursa Major", 11.062129, 56.382344, 2.37, 0.06),
    StarEntry("Phecda", "Ursa Major", 11.523388, 53.694758, 2.41, 0.00),
]


def star_count() -> int:
    """Return the number of catalog stars."""

    return len(STAR_CATALOG)


def iter_catalog() -> Iterable[StarEntry]:
    """Iterate over catalog entries."""

    return iter(STAR_CATALOG)


def equatorial_to_cartesian(ra_hours: float, dec_degrees: float) -> list[float]:
    """Convert right ascension/declination to a unit vector in J2000 frame."""

    assert ra_hours is not None, "ra_hours must be provided"
    ra_radians = math.radians(ra_hours * 15.0)
    dec_radians = math.radians(dec_degrees)

    x = math.cos(dec_radians) * math.cos(ra_radians)
    y = math.sin(dec_radians)
    z = math.cos(dec_radians) * math.sin(ra_radians)

    return [x, y, z]
