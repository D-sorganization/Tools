"""A 16-club reference library in SI units.

Values are conservative representative numbers normalized to SI from
typical published manufacturer specs. The primary numeric source is the
15-club table in UpstreamDrift's MuJoCo humanoid-golf model
(``src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/
club_configurations.py``), which lists lengths in inches, masses in
grams, head MOI in g·cm², and CG depth in millimeters; this module
converts those to meters, kilograms, and kg·m² (1 in = 0.0254 m,
1 g = 1e-3 kg, 1 g·cm² = 1e-7 kg·m²).

Fields the source table does not carry are filled with typical
published values, cited per field:

* ``cg_height_m`` — typical published CG heights above the sole
  (drivers ~25-30 mm, irons and wedges progressively lower).
* Face bulge/roll radii — typical published fitting values: drivers
  and fairway woods about 0.25-0.33 m (10-13 in) in both directions,
  hybrids slightly tighter; irons, wedges, and putters are flat.

The driver appears at three lofts (9.5°, 10.5°, 12°) sharing one head
platform, the standard loft ladder on published driver spec sheets.
"""

from __future__ import annotations

from rate_of_closure._contracts import ensure, require

from .types import ClubSpec, ClubType, HeadStyle

__all__ = ["CLUB_LIBRARY", "club_names", "get_club"]

_IN = 0.0254  # meters per inch
_G = 1.0e-3  # kilograms per gram
_GCM2 = 1.0e-7  # kg·m² per g·cm²

#: Typical driver/wood face curvature radius [m] (~11 in), per
#: published fitting references (bulge and roll 10-13 in radius).
_WOOD_BULGE_M = 0.30
_WOOD_ROLL_M = 0.28
_HYBRID_CURVE_M = 0.25


def _spec(
    name: str,
    club_type: ClubType,
    length_in: float,
    head_g: float,
    loft_deg: float,
    lie_deg: float,
    moi_gcm2: float,
    cg_depth_mm: float,
    cg_height_mm: float,
    bulge_m: float | None = None,
    roll_m: float | None = None,
    head_style: HeadStyle = HeadStyle.AUTO,
) -> ClubSpec:
    """Normalize one imperial/CGS source row into an SI ``ClubSpec``."""
    return ClubSpec(
        name=name,
        club_type=club_type,
        length_m=length_in * _IN,
        head_mass_kg=head_g * _G,
        loft_deg=loft_deg,
        lie_deg=lie_deg,
        moi_about_shaft_kg_m2=moi_gcm2 * _GCM2,
        cg_depth_m=cg_depth_mm * 1.0e-3,
        cg_height_m=cg_height_mm * 1.0e-3,
        face_bulge_radius_m=bulge_m,
        face_roll_radius_m=roll_m,
        head_style=head_style,
    )


#: Ordered name -> spec mapping; exactly 16 clubs, driver through putters.
CLUB_LIBRARY: dict[str, ClubSpec] = {
    spec.name: spec
    for spec in (
        _spec(
            "Driver 9.5°",
            ClubType.DRIVER,
            45.5,
            200,
            9.5,
            56.0,
            5200,
            25,
            28,
            _WOOD_BULGE_M,
            _WOOD_ROLL_M,
        ),
        _spec(
            "Driver 10.5°",
            ClubType.DRIVER,
            45.5,
            200,
            10.5,
            56.0,
            5200,
            25,
            28,
            _WOOD_BULGE_M,
            _WOOD_ROLL_M,
        ),
        _spec(
            "Driver 12°",
            ClubType.DRIVER,
            45.5,
            200,
            12.0,
            56.0,
            5200,
            25,
            28,
            _WOOD_BULGE_M,
            _WOOD_ROLL_M,
        ),
        _spec(
            "3-Wood",
            ClubType.WOOD,
            43.0,
            210,
            15.0,
            57.0,
            4500,
            22,
            23,
            _WOOD_BULGE_M,
            _WOOD_ROLL_M,
        ),
        _spec(
            "5-Wood",
            ClubType.WOOD,
            42.0,
            215,
            18.0,
            58.0,
            4300,
            20,
            22,
            _WOOD_BULGE_M,
            _WOOD_ROLL_M,
        ),
        _spec(
            "3-Hybrid",
            ClubType.HYBRID,
            40.5,
            230,
            19.0,
            59.0,
            3800,
            18,
            21,
            _HYBRID_CURVE_M,
            _HYBRID_CURVE_M,
        ),
        _spec("3-Iron", ClubType.IRON, 39.0, 240, 21.0, 59.5, 2800, 15, 20),
        _spec("5-Iron", ClubType.IRON, 38.0, 245, 27.0, 61.0, 2600, 14, 19),
        _spec("7-Iron", ClubType.IRON, 37.0, 250, 34.0, 62.5, 2400, 13, 19),
        _spec("9-Iron", ClubType.IRON, 36.0, 255, 41.0, 64.0, 2200, 12, 18),
        _spec("Pitching Wedge", ClubType.WEDGE, 35.5, 290, 46.0, 64.0, 2100, 11, 17),
        _spec("Gap Wedge", ClubType.WEDGE, 35.25, 295, 52.0, 64.0, 2000, 10, 17),
        _spec("Sand Wedge", ClubType.WEDGE, 35.0, 300, 56.0, 64.0, 1900, 10, 16),
        _spec("Lob Wedge", ClubType.WEDGE, 35.0, 305, 60.0, 64.0, 1850, 9, 16),
        # Putters (H1, #4125): typical published values — ~34 in length,
        # 3° loft, 70° lie; blades ~350 g with a shallow CG close to the
        # face, mallets ~360 g with a deeper CG and higher head MOI
        # (typical published putter fitting references, SI-normalized).
        _spec(
            "Blade Putter",
            ClubType.PUTTER,
            34.0,
            350,
            3.0,
            70.0,
            2500,
            12,
            14,
            head_style=HeadStyle.BLADE,
        ),
        _spec(
            "Mallet Putter",
            ClubType.PUTTER,
            34.0,
            360,
            3.0,
            70.0,
            4500,
            35,
            14,
            head_style=HeadStyle.MALLET,
        ),
    )
}
ensure(len(CLUB_LIBRARY) == 16, "library must hold exactly 16 clubs")


def club_names() -> list[str]:
    """Club names in display order (driver first, putter last)."""
    return list(CLUB_LIBRARY)


def get_club(name: str) -> ClubSpec:
    """Look up a library club by display name.

    Raises:
        PreconditionError: If ``name`` is not in the library.
    """
    require(name in CLUB_LIBRARY, f"unknown club {name!r}")
    return CLUB_LIBRARY[name]
