"""
Astronomical Constants and Planetary Data
==========================================

Contains scientifically accurate constants and orbital elements for the solar system.
Data sources: NASA JPL, IAU 2015 resolutions, and planetary fact sheets.

Orbital elements are given for J2000.0 epoch (January 1, 2000, 12:00 TT)
"""

from dataclasses import dataclass

# ==============================================================================
# FUNDAMENTAL CONSTANTS
# ==============================================================================

# Gravitational constant (m³ kg⁻¹ s⁻²)
G = 6.67430e-11

# Speed of light (m/s)
C = 299792458

# Astronomical Unit in meters (IAU 2012 definition)
AU = 149597870700

# Astronomical Unit in kilometers
AU_KM = AU / 1000

# Julian year in seconds
JULIAN_YEAR = 365.25 * 24 * 3600

# Julian century in seconds
JULIAN_CENTURY = 100 * JULIAN_YEAR

# J2000.0 epoch as Julian Date
J2000 = 2451545.0

# Seconds per day
SECONDS_PER_DAY = 86400

# Days per Julian year
DAYS_PER_YEAR = 365.25

# ==============================================================================
# SOLAR DATA
# ==============================================================================

SUN_MASS = 1.98892e30  # kg
SUN_RADIUS = 696340  # km
SUN_GM = 1.32712440018e20  # m³/s² (standard gravitational parameter)
SUN_LUMINOSITY = 3.828e26  # W
SUN_TEMPERATURE = 5778  # K (effective)

# ==============================================================================
# PLANETARY DATA
# ==============================================================================


@dataclass
class OrbitalElements:
    """
    Keplerian orbital elements for a celestial body.

    All angles are in degrees, distances in AU, and rates in per century.
    These are mean elements at J2000.0 with secular variations.
    """

    # Mean elements at J2000.0
    semi_major_axis: float  # a - AU
    eccentricity: float  # e - dimensionless
    inclination: float  # i - degrees
    longitude_ascending: float  # Ω (Omega) - degrees
    longitude_perihelion: float  # ϖ (omega bar) - degrees
    mean_longitude: float  # L - degrees

    # Secular rates (per Julian century)
    semi_major_axis_rate: float = 0.0
    eccentricity_rate: float = 0.0
    inclination_rate: float = 0.0
    longitude_ascending_rate: float = 0.0
    longitude_perihelion_rate: float = 0.0
    mean_longitude_rate: float = 0.0


@dataclass
class PhysicalProperties:
    """Physical properties of a celestial body."""

    mass: float  # kg
    radius: float  # km (mean radius)
    density: float  # kg/m³
    surface_gravity: float  # m/s²
    escape_velocity: float  # km/s
    rotation_period: float  # hours (sidereal)
    axial_tilt: float  # degrees
    albedo: float  # geometric albedo
    temperature: float  # K (mean surface or cloud-top)
    color: tuple  # RGB color for visualization (0-1 range)


# Orbital elements from NASA JPL - Keplerian Elements for Approximate Positions
# Source: https://ssd.jpl.nasa.gov/planets/approx_pos.html
# Valid for 1800 AD - 2050 AD

ORBITAL_ELEMENTS: dict[str, OrbitalElements] = {
    "Mercury": OrbitalElements(
        semi_major_axis=0.38709927,
        eccentricity=0.20563593,
        inclination=7.00497902,
        longitude_ascending=48.33076593,
        longitude_perihelion=77.45779628,
        mean_longitude=252.25032350,
        semi_major_axis_rate=0.00000037,
        eccentricity_rate=0.00001906,
        inclination_rate=-0.00594749,
        longitude_ascending_rate=-0.12534081,
        longitude_perihelion_rate=0.16047689,
        mean_longitude_rate=149472.67411175,
    ),
    "Venus": OrbitalElements(
        semi_major_axis=0.72333566,
        eccentricity=0.00677672,
        inclination=3.39467605,
        longitude_ascending=76.67984255,
        longitude_perihelion=131.60246718,
        mean_longitude=181.97909950,
        semi_major_axis_rate=0.00000390,
        eccentricity_rate=-0.00004107,
        inclination_rate=-0.00078890,
        longitude_ascending_rate=-0.27769418,
        longitude_perihelion_rate=0.00268329,
        mean_longitude_rate=58517.81538729,
    ),
    "Earth": OrbitalElements(
        semi_major_axis=1.00000261,
        eccentricity=0.01671123,
        inclination=-0.00001531,
        longitude_ascending=0.0,
        longitude_perihelion=102.93768193,
        mean_longitude=100.46457166,
        semi_major_axis_rate=0.00000562,
        eccentricity_rate=-0.00004392,
        inclination_rate=-0.01294668,
        longitude_ascending_rate=0.0,
        longitude_perihelion_rate=0.32327364,
        mean_longitude_rate=35999.37244981,
    ),
    "Mars": OrbitalElements(
        semi_major_axis=1.52371034,
        eccentricity=0.09339410,
        inclination=1.84969142,
        longitude_ascending=49.55953891,
        longitude_perihelion=-23.94362959,
        mean_longitude=-4.55343205,
        semi_major_axis_rate=0.00001847,
        eccentricity_rate=0.00007882,
        inclination_rate=-0.00813131,
        longitude_ascending_rate=-0.29257343,
        longitude_perihelion_rate=0.44441088,
        mean_longitude_rate=19140.30268499,
    ),
    "Jupiter": OrbitalElements(
        semi_major_axis=5.20288700,
        eccentricity=0.04838624,
        inclination=1.30439695,
        longitude_ascending=100.47390909,
        longitude_perihelion=14.72847983,
        mean_longitude=34.39644051,
        semi_major_axis_rate=-0.00011607,
        eccentricity_rate=-0.00013253,
        inclination_rate=-0.00183714,
        longitude_ascending_rate=0.20469106,
        longitude_perihelion_rate=0.21252668,
        mean_longitude_rate=3034.74612775,
    ),
    "Saturn": OrbitalElements(
        semi_major_axis=9.53667594,
        eccentricity=0.05386179,
        inclination=2.48599187,
        longitude_ascending=113.66242448,
        longitude_perihelion=92.59887831,
        mean_longitude=49.95424423,
        semi_major_axis_rate=-0.00125060,
        eccentricity_rate=-0.00050991,
        inclination_rate=0.00193609,
        longitude_ascending_rate=-0.28867794,
        longitude_perihelion_rate=-0.41897216,
        mean_longitude_rate=1222.49362201,
    ),
    "Uranus": OrbitalElements(
        semi_major_axis=19.18916464,
        eccentricity=0.04725744,
        inclination=0.77263783,
        longitude_ascending=74.01692503,
        longitude_perihelion=170.95427630,
        mean_longitude=313.23810451,
        semi_major_axis_rate=-0.00196176,
        eccentricity_rate=-0.00004397,
        inclination_rate=-0.00242939,
        longitude_ascending_rate=0.04240589,
        longitude_perihelion_rate=0.40805281,
        mean_longitude_rate=428.48202785,
    ),
    "Neptune": OrbitalElements(
        semi_major_axis=30.06992276,
        eccentricity=0.00859048,
        inclination=1.77004347,
        longitude_ascending=131.78422574,
        longitude_perihelion=44.96476227,
        mean_longitude=-55.12002969,
        semi_major_axis_rate=0.00026291,
        eccentricity_rate=0.00005105,
        inclination_rate=0.00035372,
        longitude_ascending_rate=-0.00508664,
        longitude_perihelion_rate=-0.32241464,
        mean_longitude_rate=218.45945325,
    ),
    "Pluto": OrbitalElements(
        semi_major_axis=39.48211675,
        eccentricity=0.24882730,
        inclination=17.14001206,
        longitude_ascending=110.30393684,
        longitude_perihelion=224.06891629,
        mean_longitude=238.92903833,
        semi_major_axis_rate=-0.00031596,
        eccentricity_rate=0.00005170,
        inclination_rate=0.00004818,
        longitude_ascending_rate=-0.01183482,
        longitude_perihelion_rate=-0.04062942,
        mean_longitude_rate=145.20780515,
    ),
}

# Physical properties from NASA Planetary Fact Sheets
PHYSICAL_PROPERTIES: dict[str, PhysicalProperties] = {
    "Sun": PhysicalProperties(
        mass=SUN_MASS,
        radius=SUN_RADIUS,
        density=1408,
        surface_gravity=274,
        escape_velocity=617.7,
        rotation_period=609.12,  # Equatorial, sidereal
        axial_tilt=7.25,
        albedo=0.0,
        temperature=5778,
        color=(1.0, 0.95, 0.8),
    ),
    "Mercury": PhysicalProperties(
        mass=3.3011e23,
        radius=2439.7,
        density=5427,
        surface_gravity=3.7,
        escape_velocity=4.3,
        rotation_period=1407.6,
        axial_tilt=0.034,
        albedo=0.142,
        temperature=440,
        color=(0.7, 0.6, 0.5),
    ),
    "Venus": PhysicalProperties(
        mass=4.8675e24,
        radius=6051.8,
        density=5243,
        surface_gravity=8.87,
        escape_velocity=10.36,
        rotation_period=-5832.5,  # Negative = retrograde
        axial_tilt=177.36,
        albedo=0.689,
        temperature=737,
        color=(0.9, 0.75, 0.5),
    ),
    "Earth": PhysicalProperties(
        mass=5.97237e24,
        radius=6371.0,
        density=5514,
        surface_gravity=9.807,
        escape_velocity=11.186,
        rotation_period=23.9345,
        axial_tilt=23.4393,
        albedo=0.367,
        temperature=288,
        color=(0.2, 0.5, 0.9),
    ),
    "Mars": PhysicalProperties(
        mass=6.4171e23,
        radius=3389.5,
        density=3933,
        surface_gravity=3.721,
        escape_velocity=5.03,
        rotation_period=24.6229,
        axial_tilt=25.19,
        albedo=0.170,
        temperature=210,
        color=(0.9, 0.4, 0.2),
    ),
    "Jupiter": PhysicalProperties(
        mass=1.8982e27,
        radius=69911,
        density=1326,
        surface_gravity=24.79,
        escape_velocity=59.5,
        rotation_period=9.925,
        axial_tilt=3.13,
        albedo=0.538,
        temperature=165,
        color=(0.9, 0.8, 0.6),
    ),
    "Saturn": PhysicalProperties(
        mass=5.6834e26,
        radius=58232,
        density=687,
        surface_gravity=10.44,
        escape_velocity=35.5,
        rotation_period=10.656,
        axial_tilt=26.73,
        albedo=0.499,
        temperature=134,
        color=(0.9, 0.85, 0.6),
    ),
    "Uranus": PhysicalProperties(
        mass=8.6810e25,
        radius=25362,
        density=1271,
        surface_gravity=8.87,
        escape_velocity=21.3,
        rotation_period=-17.24,  # Retrograde
        axial_tilt=97.77,
        albedo=0.488,
        temperature=76,
        color=(0.6, 0.85, 0.9),
    ),
    "Neptune": PhysicalProperties(
        mass=1.02413e26,
        radius=24622,
        density=1638,
        surface_gravity=11.15,
        escape_velocity=23.5,
        rotation_period=16.11,
        axial_tilt=28.32,
        albedo=0.442,
        temperature=72,
        color=(0.3, 0.5, 0.9),
    ),
    "Pluto": PhysicalProperties(
        mass=1.303e22,
        radius=1188.3,
        density=1854,
        surface_gravity=0.62,
        escape_velocity=1.21,
        rotation_period=-153.2928,
        axial_tilt=122.53,
        albedo=0.72,
        temperature=44,
        color=(0.8, 0.75, 0.7),
    ),
    "Moon": PhysicalProperties(
        mass=7.342e22,
        radius=1737.4,
        density=3344,
        surface_gravity=1.62,
        escape_velocity=2.38,
        rotation_period=655.728,  # Synchronous
        axial_tilt=6.687,
        albedo=0.136,
        temperature=250,
        color=(0.7, 0.7, 0.7),
    ),
}

# Standard gravitational parameters (GM) in m³/s²
GM: dict[str, float] = {
    "Sun": 1.32712440018e20,
    "Mercury": 2.2032e13,
    "Venus": 3.24859e14,
    "Earth": 3.986004418e14,
    "Mars": 4.282837e13,
    "Jupiter": 1.26686534e17,
    "Saturn": 3.7931187e16,
    "Uranus": 5.793939e15,
    "Neptune": 6.836529e15,
    "Pluto": 8.71e11,
    "Moon": 4.9048695e12,
}

# Orbital periods in Earth days
ORBITAL_PERIODS: dict[str, float] = {
    "Mercury": 87.969,
    "Venus": 224.701,
    "Earth": 365.256,
    "Mars": 686.980,
    "Jupiter": 4332.589,
    "Saturn": 10759.22,
    "Uranus": 30688.5,
    "Neptune": 60182.0,
    "Pluto": 90560.0,
}

# Planet display order (inner to outer)
PLANET_ORDER = [
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
]

# Inner and outer planet classifications
INNER_PLANETS = ["Mercury", "Venus", "Earth", "Mars"]
OUTER_PLANETS = ["Jupiter", "Saturn", "Uranus", "Neptune"]
DWARF_PLANETS = ["Pluto"]

# ==============================================================================
# VISUALIZATION CONSTANTS
# ==============================================================================

# Scale factors for visualization (to make distances viewable)
DISTANCE_SCALE = 1e-9  # Convert meters to viewable units
SIZE_SCALE_PLANETS = 1e-6  # Scale for planet sizes
SIZE_SCALE_SUN = 1e-7  # Smaller scale for the sun to fit in view

# Minimum and maximum visual sizes
MIN_PLANET_SIZE = 0.02
MAX_PLANET_SIZE = 0.5
SUN_VISUAL_SIZE = 1.0

# Orbit trail settings
ORBIT_TRAIL_POINTS = 360  # Points per complete orbit
ORBIT_TRAIL_OPACITY = 0.5

# Colors for various elements
ORBIT_COLOR = (0.3, 0.3, 0.5, 0.5)
TRAJECTORY_COLOR = (0.0, 1.0, 0.5, 0.8)
GRID_COLOR = (0.2, 0.2, 0.3, 0.3)
