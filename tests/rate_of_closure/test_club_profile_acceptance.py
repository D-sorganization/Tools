"""Profile-view acceptance gates for clubhead realism (#4799 G5).

"Looks like a real club" is pinned here as a **test**, not an opinion.
Every measurement is taken from the toe-side profile view — a camera on
the toe side looking along the head frame's z axis — of the mesh that
comes out of the public :func:`~rate_of_closure.club.parametric_head_mesh`
entry point, the same one both GUIs render (PyQt ``club_view`` and the
React ``clubMeshSource``). Nothing here reaches into the internal
builder, so a change that keeps ``build_parametric_head`` honest but
breaks the renderable mesh still fails.

The toe-view silhouette is the mesh's **mid-plane slice**: every vertex
with ``|z| <= 1e-6 m``. Because each superellipse ring carries an exact
crown vertex (theta = pi/2) and an exact sole vertex (theta = 3pi/2),
and both cap fan centers sit on ``z = 0``, that slice is precisely the
outline a toe-side orthographic camera draws — 30 points for a 4-station
profile, 36 for the 5-station mallet.

Criteria (from epic #4799), parametrized over the whole 16-club library:

* the leading edge is the head's forward-most point, sits within a few
  millimeters of the hosel station on blades, and legitimately sits
  20-40 mm ahead of it on a driver;
* the face slopes **back** from the leading edge — the toe-view front
  edge recedes strictly with height and the topline setback is
  ``H sin(loft)`` on every flat-faced blade;
* the authored face height becomes slant height: the profile stands
  ``H cos(loft)`` tall;
* the sole is flat and continuous, and a wedge sole is deeper than an
  iron sole;
* the silhouette is closed — watertight, positive volume inside the
  sane band, z extents symmetric about the face center at z = 0.

:class:`TestCenterPivotRegression` is the anti-regression gate: it
computes what the leading-edge station **would** be if loft were applied
by rotating the face about its center (the #4799 root cause) and proves
the shipped mesh does not match it. Reintroduce the center pivot and
that class goes red for all 16 clubs.

The vitest twin is ``web/src/model/clubProfileAcceptance.test.ts``; both
runtimes pin the same two rendered tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cache

import numpy as np
import pytest

from rate_of_closure.club import (
    CLUB_LIBRARY,
    ClubSpec,
    ClubType,
    face_center_point,
    face_sagitta,
    hosel_point,
    is_watertight,
    mesh_volume_centroid,
    parametric_head_mesh,
)
from rate_of_closure.club.head_profiles import mass_scale, profile_for
from rate_of_closure.club.volumetrics import HEAD_VOLUME_BOUNDS_M3

pytestmark = pytest.mark.unit

#: Half-width of the mid-plane slice that forms the toe-view silhouette.
#: Ring crown/sole vertices land within ~1e-9 m of z = 0; every other
#: ring vertex is at least 1e-2 m away, so the band is unambiguous.
_PROFILE_Z_TOL_M = 1.0e-6
#: Sole band: profile points within 3% of the head's profile height of
#: the leading-edge line. Wide enough to catch a wedge's bounce dip,
#: narrow enough to exclude the next station up.
_SOLE_BAND_FRACTION = 0.03
#: Heights (fractions of the profile height above the leading edge) at
#: which the toe-view front edge is sampled.
_FRONT_EDGE_STEPS = 5

_ALL_CLUBS = list(CLUB_LIBRARY)
_BLADE_TYPES = (ClubType.IRON, ClubType.WEDGE)
_BLADES = [n for n, s in CLUB_LIBRARY.items() if s.club_type in _BLADE_TYPES]
_IRONS = [n for n, s in CLUB_LIBRARY.items() if s.club_type is ClubType.IRON]
_WEDGES = [n for n, s in CLUB_LIBRARY.items() if s.club_type is ClubType.WEDGE]
_DRIVERS = [n for n, s in CLUB_LIBRARY.items() if s.club_type is ClubType.DRIVER]
#: Clubs whose face carries no bulge/roll — the analytic face normal
#: ``(cos loft, sin loft, 0)`` is realized exactly on these.
_FLAT_FACED = [
    n
    for n, s in CLUB_LIBRARY.items()
    if s.face_bulge_radius_m is None and s.face_roll_radius_m is None
]
#: Clubs whose topline *is* the face top (blades and the blade putter);
#: woods, hybrids, and the mallet top out on the crown instead.
_FACE_TOPPED = [*_BLADES, "Blade Putter"]


@dataclass(frozen=True)
class ProfileMetrics:
    """Toe-view geometric report for one club, millimeters unless noted."""

    name: str
    loft_deg: float
    leading_edge_x: float
    leading_edge_y: float
    authored_le_x: float
    authored_le_y: float
    hosel_x: float
    topline_setback: float
    expected_setback: float
    topline_height: float
    expected_face_height: float
    sole_depth: float
    sole_flatness: float
    sole_gap: float
    sole_front_x: float
    sole_points: int
    profile_points: int
    front_edge: tuple[float, ...]
    width: float
    z_symmetry: float
    volume_cm3: float
    watertight: bool
    face_normal_deviation: float
    center_pivot_le_x: float

    @property
    def offset(self) -> float:
        """Leading edge ahead of (+) or behind (-) the hosel station."""
        return self.leading_edge_x - self.hosel_x

    @property
    def center_pivot_offset(self) -> float:
        """What :attr:`offset` would be under center-pivot loft."""
        return self.center_pivot_le_x - self.hosel_x

    @property
    def forward_kick(self) -> float:
        """Center-pivot loft's forward throw of the authored leading
        edge — the onset the lean avoids."""
        return self.center_pivot_le_x - self.authored_le_x

    @property
    def face_cap_realized(self) -> bool:
        """Whether the rendered mesh caps its face on the published
        :func:`face_center_point`."""
        return math.isfinite(self.face_normal_deviation)


def _profile_slice(spec: ClubSpec) -> np.ndarray:
    """Toe-view silhouette vertices of the rendered head, meters.

    Built from :func:`parametric_head_mesh` — the public entry point —
    so the report measures exactly what the GUIs draw.
    """
    mesh = parametric_head_mesh(spec)
    flat = np.asarray(mesh.triangles).reshape(-1, 3)
    mid = flat[np.abs(flat[:, 2]) <= _PROFILE_Z_TOL_M]
    return np.unique(mid, axis=0)


def _authored_face_height_m(spec: ClubSpec) -> float:
    """The authored face height ``H`` at this club's mass scale."""
    return 2.0 * profile_for(spec).sections[0][1] * mass_scale(spec)


def _authored_leading_edge_m(spec: ClubSpec) -> tuple[float, float]:
    """The unlofted, mass-scaled leading edge ``(x, y)`` [m].

    The bottom of the authored face section, set back by its own
    curvature sagitta. This is the station the epic says a real club
    keeps — the fixed line of the leading-edge lean and the point the
    center-pivot counterfactual rotates.
    """
    scale = mass_scale(spec)
    x_face, half_height, _hw, y_center = (
        value * scale for value in profile_for(spec).sections[0]
    )
    return (
        x_face - face_sagitta(spec, 0.0, -half_height),
        y_center - half_height,
    )


@cache
def _metrics(name: str) -> ProfileMetrics:
    """Measure one library club's toe-view profile (cached, pure)."""
    spec = CLUB_LIBRARY[name]
    mesh = parametric_head_mesh(spec)
    triangles = np.asarray(mesh.triangles)
    flat = triangles.reshape(-1, 3)
    profile = _profile_slice(spec)

    lead = profile[int(np.argmax(profile[:, 0]))]
    le_x, le_y = float(lead[0]), float(lead[1])
    y_max = float(profile[:, 1].max())
    y_min = float(profile[:, 1].min())
    height = y_max - y_min
    topline = profile[np.abs(profile[:, 1] - y_max) <= 1.0e-9]
    topline_x = float(topline[:, 0].max())

    sole = profile[profile[:, 1] <= le_y + _SOLE_BAND_FRACTION * height]
    sole = sole[np.argsort(sole[:, 0])]
    gaps = np.diff(sole[:, 0]) if sole.shape[0] > 1 else np.zeros(1)

    front = []
    for step in range(_FRONT_EDGE_STEPS + 1):
        cut = le_y + (y_max - le_y) * step / _FRONT_EDGE_STEPS
        above = profile[profile[:, 1] >= cut - 1.0e-12]
        front.append(float(above[:, 0].max()) * 1.0e3)

    volume, _centroid = mesh_volume_centroid(triangles)

    lam = math.radians(spec.loft_deg)
    want = np.array([math.cos(lam), math.sin(lam), 0.0])
    center = np.asarray(face_center_point(spec))
    on_cap = (np.abs(triangles - center).sum(axis=2) < 1.0e-12).any(axis=1)
    # A mesh that no longer realizes its own published face center — the
    # pre-#4799 center-pivot generator, for one — reports an infinite
    # deviation rather than raising, so the gates below fail on the
    # geometry rather than on an empty reduction.
    deviation = (
        float(np.linalg.norm(np.asarray(mesh.normals)[on_cap] - want, axis=1).max())
        if bool(on_cap.any())
        else math.inf
    )

    # Center-pivot counterfactual: rotate the *authored* leading edge
    # about the unlofted face center — exactly what the pre-#4799
    # generator did. Anchored on the authored station, not on the
    # measured one, so the prediction is independent of however the
    # generator currently applies loft.
    authored_x, authored_y = _authored_leading_edge_m(spec)
    unlofted_center = face_center_point(replace(spec, loft_deg=0.0))
    pivot_x = (
        unlofted_center[0]
        + math.cos(lam) * (authored_x - unlofted_center[0])
        - math.sin(lam) * (authored_y - unlofted_center[1])
    )

    return ProfileMetrics(
        name=name,
        loft_deg=spec.loft_deg,
        leading_edge_x=le_x * 1.0e3,
        leading_edge_y=le_y * 1.0e3,
        authored_le_x=authored_x * 1.0e3,
        authored_le_y=authored_y * 1.0e3,
        hosel_x=hosel_point(spec)[0] * 1.0e3,
        topline_setback=(le_x - topline_x) * 1.0e3,
        expected_setback=_authored_face_height_m(spec) * math.sin(lam) * 1.0e3,
        topline_height=(y_max - le_y) * 1.0e3,
        expected_face_height=_authored_face_height_m(spec) * math.cos(lam) * 1.0e3,
        sole_depth=float(sole[:, 0].max() - sole[:, 0].min()) * 1.0e3,
        sole_flatness=float(sole[:, 1].max() - sole[:, 1].min()) * 1.0e3,
        sole_gap=float(gaps.max()) * 1.0e3,
        sole_front_x=float(sole[:, 0].max()) * 1.0e3,
        sole_points=int(sole.shape[0]),
        profile_points=int(profile.shape[0]),
        front_edge=tuple(front),
        width=float(flat[:, 2].max() - flat[:, 2].min()) * 1.0e3,
        z_symmetry=float(flat[:, 2].max() + flat[:, 2].min()) * 1.0e3,
        volume_cm3=volume * 1.0e6,
        watertight=is_watertight(triangles),
        face_normal_deviation=deviation,
        center_pivot_le_x=pivot_x * 1.0e3,
    )


_PROFILE_HEADER = (
    f"{'club':<15}{'loft':>6}{'le_x':>7}{'hos_x':>7}{'setbk':>7}"
    f"{'Hsin':>7}{'topH':>7}{'soleD':>7}{'soleF':>7}{'vol_cc':>8}"
)
_ONSET_HEADER = (
    f"{'club':<15}{'le_x':>7}{'cp_le_x':>9}{'kick':>7}"
    f"{'hos_x':>7}{'onset':>7}{'cp_onset':>10}"
)


def _profile_row(m: ProfileMetrics) -> str:
    """One rendered row of the toe-view geometric report."""
    return (
        f"{m.name:<15}{m.loft_deg:>6.1f}{m.leading_edge_x:>7.2f}"
        f"{m.hosel_x:>7.2f}{m.topline_setback:>7.2f}{m.expected_setback:>7.2f}"
        f"{m.topline_height:>7.2f}{m.sole_depth:>7.2f}"
        f"{m.sole_flatness:>7.3f}{m.volume_cm3:>8.1f}"
    )


def _onset_row(m: ProfileMetrics) -> str:
    """One rendered row of the center-pivot counterfactual report."""
    return (
        f"{m.name:<15}{m.leading_edge_x:>7.2f}{m.center_pivot_le_x:>9.2f}"
        f"{m.forward_kick:>7.2f}{m.hosel_x:>7.2f}{m.offset:>7.2f}"
        f"{m.center_pivot_offset:>10.2f}"
    )


def profile_report() -> str:
    """The toe-view geometric report for the whole library, as a table."""
    rows = [_PROFILE_HEADER, *(_profile_row(_metrics(n)) for n in _ALL_CLUBS)]
    return "\n".join(rows)


def onset_report() -> str:
    """The center-pivot counterfactual table for the whole library."""
    rows = [_ONSET_HEADER, *(_onset_row(_metrics(n)) for n in _ALL_CLUBS)]
    return "\n".join(rows)


#: Pinned toe-view geometric report (mm; volume cm^3). ``setbk`` is the
#: topline setback behind the leading edge, ``Hsin``/``topH`` the
#: authored face height times sin/cos(loft), ``soleD``/``soleF`` the
#: sole line's front-to-back depth and its flatness.
_EXPECTED_PROFILE_REPORT = """\
club             loft   le_x  hos_x  setbk   Hsin   topH  soleD  soleF  vol_cc
Driver 9.5°       9.5  53.60  20.43  53.33   9.24  58.19  73.87  4.603   572.2
Driver 10.5°     10.5  53.60  19.43  54.35  10.21  58.01  73.90  4.589   570.5
Driver 12°       12.0  53.60  17.94  55.86  11.64  57.71  73.94  4.565   567.5
3-Wood           15.0  54.45  15.23  59.81  14.73  57.92  75.22  4.582   588.2
5-Wood           18.0  54.87  12.37  63.30  17.73  57.48  75.88  4.547   592.8
3-Hybrid         19.0  36.85   6.05  45.12  15.63  47.28  38.85  1.891   259.4
3-Iron           21.0  10.85   5.92  17.68  17.68  46.05  20.72  0.000    49.7
5-Iron           27.0  10.93   5.96  22.55  22.55  44.25  20.86  0.000    48.5
7-Iron           34.0  11.00   6.00  27.96  27.96  41.45  21.00  0.000    46.0
9-Iron           41.0  11.07   6.04  33.02  33.02  37.99  21.14  0.000    42.7
Pitching Wedge   46.0  11.87   8.40  36.99  36.99  35.72  27.97  0.549    54.3
Gap Wedge        52.0  11.93   8.45  40.75  40.75  31.84  28.11  0.490    49.0
Sand Wedge       56.0  12.00   8.50  43.11  43.11  29.08  28.25  0.447    45.3
Lob Wedge        60.0  12.07   8.55  45.28  45.28  26.14  28.40  0.402    41.1
Blade Putter      3.0  12.00   1.19   1.31   1.31  24.97  26.00  0.000    46.3
Mallet Putter     3.0  20.00  12.53  26.49   1.47  28.46  63.37  1.165   215.8"""

#: Pinned center-pivot counterfactual (mm). ``cp_le_x`` is where the
#: leading edge would land if loft rotated the face about its center;
#: ``kick`` is the forward throw that costs (epic #4799 measured ~21.6 mm
#: on the sand wedge, ~14 mm on the 7-iron, ~5 mm on the driver), and
#: ``cp_onset`` the resulting onset past the shaft (~25 mm on a wedge).
_EXPECTED_ONSET_REPORT = """\
club              le_x  cp_le_x   kick  hos_x  onset  cp_onset
Driver 9.5°      53.60    58.24   4.64  20.43  33.17     37.81
Driver 10.5°     53.60    58.72   5.13  19.43  34.17     39.29
Driver 12°       53.60    59.45   5.85  17.94  35.66     41.51
3-Wood           54.45    61.87   7.42  15.23  39.22     46.63
5-Wood           54.87    63.80   8.94  12.37  42.50     51.43
3-Hybrid         36.85    44.72   7.88   6.05  30.80     38.67
3-Iron           10.85    19.69   8.84   5.92   4.93     13.77
5-Iron           10.93    22.20  11.27   5.96   4.97     16.24
7-Iron           11.00    24.98  13.98   6.00   5.00     18.98
9-Iron           11.07    27.58  16.51   6.04   5.03     21.54
Pitching Wedge   11.87    30.36  18.49   8.40   3.46     21.95
Gap Wedge        11.93    32.31  20.37   8.45   3.48     23.85
Sand Wedge       12.00    33.55  21.55   8.50   3.50     25.05
Lob Wedge        12.07    34.71  22.64   8.55   3.52     26.16
Blade Putter     12.00    12.65   0.65   1.19  10.81     11.46
Mallet Putter    20.00    20.73   0.73  12.53   7.47      8.20"""

#: Millimeter tolerance when a rendered row is compared with its pin.
_REPORT_TOL_MM = 0.02
#: Volume column tolerance, cm^3.
_REPORT_TOL_CM3 = 0.2


def _parse_report(text: str) -> dict[str, list[float]]:
    """Name -> numeric cells for a rendered report table (header skipped)."""
    parsed: dict[str, list[float]] = {}
    for line in text.splitlines()[1:]:
        parsed[line[:15].strip()] = [float(cell) for cell in line[15:].split()]
    return parsed


def _assert_report_matches(actual: str, expected: str, volume_col: int) -> None:
    """Compare a rendered table against its pin, column by column."""
    got, want = _parse_report(actual), _parse_report(expected)
    assert sorted(got) == sorted(want), f"club set changed\n{actual}"
    for name in want:
        for index, (a, b) in enumerate(zip(got[name], want[name], strict=True)):
            tol = _REPORT_TOL_CM3 if index == volume_col else _REPORT_TOL_MM
            assert abs(a - b) <= tol, (
                f"{name} column {index}: {a} != pinned {b}\n"
                f"--- measured ---\n{actual}\n--- pinned ---\n{expected}"
            )


class TestProfileReport:
    """The report itself: rendered, pinned, and deterministic."""

    def test_profile_report_matches_its_pin(self) -> None:
        report = profile_report()
        print("\n" + report)  # visible under `pytest -s`
        _assert_report_matches(report, _EXPECTED_PROFILE_REPORT, volume_col=8)

    def test_onset_report_matches_its_pin(self) -> None:
        report = onset_report()
        print("\n" + report)
        _assert_report_matches(report, _EXPECTED_ONSET_REPORT, volume_col=-1)

    def test_report_covers_every_library_club(self) -> None:
        rows = profile_report().splitlines()
        assert len(rows) == len(CLUB_LIBRARY) + 1
        assert all(len(row) == len(_PROFILE_HEADER) for row in rows)

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_profile_measurement_is_deterministic(self, name: str) -> None:
        spec = CLUB_LIBRARY[name]
        first, second = _profile_slice(spec), _profile_slice(spec)
        assert np.array_equal(first, second)
        assert _profile_row(_metrics(name)) == _profile_row(_metrics(name))


class TestLeadingEdgeStation:
    """The leading edge leads, and it sits by the shaft on a blade."""

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_leading_edge_is_the_heads_forward_most_point(self, name: str) -> None:
        """Nothing on the head reaches ahead of the leading edge, and the
        toe-view silhouette contains that forward-most point."""
        spec = CLUB_LIBRARY[name]
        flat = np.asarray(parametric_head_mesh(spec).triangles).reshape(-1, 3)
        measured = _metrics(name).leading_edge_x
        assert measured == pytest.approx(float(flat[:, 0].max()) * 1.0e3, abs=1.0e-9)

    @pytest.mark.parametrize("name", _BLADES)
    def test_blade_leading_edge_is_a_few_mm_from_the_hosel(self, name: str) -> None:
        """Onset is gone: the leading edge is even with the shaft station
        to within a real club's few millimeters of offset."""
        metrics = _metrics(name)
        assert 0.0 <= metrics.offset <= 6.0, metrics.offset

    @pytest.mark.parametrize("name", _BLADES)
    def test_blade_leading_edge_never_leads_the_authored_station(
        self, name: str
    ) -> None:
        """Loft never pushes the blade's leading edge forward: the lofted
        head reaches no further than the unlofted one."""
        spec = CLUB_LIBRARY[name]
        unlofted = _profile_slice(replace(spec, loft_deg=0.0))
        assert (
            _metrics(name).leading_edge_x
            <= float(unlofted[:, 0].max()) * 1.0e3 + 1.0e-9
        )

    @pytest.mark.parametrize("name", _DRIVERS)
    def test_driver_face_sits_legitimately_ahead_of_the_hosel(self, name: str) -> None:
        assert 20.0 <= _metrics(name).offset <= 40.0

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_hosel_is_never_ahead_of_the_leading_edge(self, name: str) -> None:
        assert _metrics(name).hosel_x <= _metrics(name).leading_edge_x


class TestFaceLean:
    """The face slopes back from the leading edge, by exactly the lean."""

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_front_edge_recedes_strictly_with_height(self, name: str) -> None:
        """Toe-view: every step up the head moves the front edge back —
        the face leans away from the ball, never toward it."""
        front = _metrics(name).front_edge
        for lower, upper in zip(front[:-1], front[1:], strict=True):
            assert upper < lower - 1.0e-6, front

    @pytest.mark.parametrize("name", _FACE_TOPPED)
    def test_topline_sets_back_by_face_height_times_sin_loft(self, name: str) -> None:
        metrics = _metrics(name)
        assert metrics.topline_setback == pytest.approx(
            metrics.expected_setback, rel=0.01, abs=1.0e-6
        )

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_top_of_head_never_sits_ahead_of_the_face_top(self, name: str) -> None:
        """Woods and mallets top out on the crown, which is further back
        still — never forward of where the face top lands."""
        metrics = _metrics(name)
        assert metrics.topline_setback >= metrics.expected_setback - 1.0e-6

    @pytest.mark.parametrize("name", _FACE_TOPPED)
    def test_face_height_compresses_to_slant_height(self, name: str) -> None:
        """The authored face height becomes slant height: the profile
        stands H cos(loft) tall — why a 52 mm wedge face is ~29 mm."""
        metrics = _metrics(name)
        assert metrics.topline_height == pytest.approx(
            metrics.expected_face_height, rel=0.01, abs=1.0e-6
        )

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_rendered_face_caps_on_the_published_face_center(self, name: str) -> None:
        """What the renderer draws and what ``face_center_point``
        advertises are the same point."""
        assert _metrics(name).face_cap_realized

    @pytest.mark.parametrize("name", _FLAT_FACED)
    def test_flat_face_realizes_the_analytic_loft_normal(self, name: str) -> None:
        """Rendered normals on the face cap are exactly
        ``(cos loft, sin loft, 0)`` for every flat-faced club."""
        assert _metrics(name).face_normal_deviation <= 1.0e-9

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_curved_face_realizes_the_loft_normal_to_first_order(
        self, name: str
    ) -> None:
        """Wood and hybrid faces are curved, so the sheared mesh agrees
        with the analytic rotated normal only to first order in the
        sagitta slope — bounded, and documented as the contract."""
        assert _metrics(name).face_normal_deviation <= 0.05


class TestSole:
    """A real sole: flat, continuous, and deeper on a wedge."""

    @pytest.mark.parametrize("name", _BLADES + ["Blade Putter"])
    def test_sole_line_is_continuous_from_the_leading_edge(self, name: str) -> None:
        """The sole runs back from the leading edge without a hole: it
        starts at the leading edge and no station gap exceeds a quarter
        of its depth."""
        metrics = _metrics(name)
        assert metrics.sole_points >= 8
        assert metrics.sole_front_x == pytest.approx(metrics.leading_edge_x, abs=1.0e-9)
        assert metrics.sole_gap <= 0.25 * metrics.sole_depth

    @pytest.mark.parametrize("name", _IRONS + ["Blade Putter"])
    def test_iron_and_blade_putter_soles_are_flat(self, name: str) -> None:
        assert _metrics(name).sole_flatness <= 1.0e-9

    @pytest.mark.parametrize("name", _WEDGES)
    def test_wedge_sole_is_flat_within_its_bounce_hint(self, name: str) -> None:
        """A wedge sole is flat to within its sub-millimeter bounce dip."""
        assert 0.2 <= _metrics(name).sole_flatness <= 0.8

    def test_wedge_soles_are_deeper_than_every_iron_sole(self) -> None:
        widest_iron = max(_metrics(n).sole_depth for n in _IRONS)
        narrowest_wedge = min(_metrics(n).sole_depth for n in _WEDGES)
        assert narrowest_wedge > widest_iron
        assert narrowest_wedge - widest_iron >= 5.0

    @pytest.mark.parametrize("name", _BLADES)
    def test_blade_sole_depth_stays_in_the_published_span(self, name: str) -> None:
        """Reference-scale sole depth: irons ~18-24 mm, wedges ~26-32 mm
        (typical published spans; no brand geometry is reproduced)."""
        spec = CLUB_LIBRARY[name]
        reference = _metrics(name).sole_depth / mass_scale(spec)
        low, high = (18.0, 24.0) if spec.club_type is ClubType.IRON else (26.0, 32.0)
        assert low <= reference <= high


class TestSilhouetteIntegrity:
    """The profile closes on a real solid, not a shell."""

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_rendered_head_is_watertight_with_positive_volume(self, name: str) -> None:
        metrics = _metrics(name)
        assert metrics.watertight
        low, high = HEAD_VOLUME_BOUNDS_M3
        assert low * 1.0e6 <= metrics.volume_cm3 <= high * 1.0e6

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_z_extents_are_symmetric_about_the_face_center(self, name: str) -> None:
        """The face center sits on z = 0 and the head straddles it."""
        metrics = _metrics(name)
        assert metrics.z_symmetry == pytest.approx(0.0, abs=1.0e-9)
        assert metrics.width > 0.0
        assert face_center_point(CLUB_LIBRARY[name])[2] == 0.0

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_toe_view_slice_is_a_closed_outline(self, name: str) -> None:
        """Two silhouette points per cross-section (crown and sole) plus
        the two cap centers — a complete outline, not a sparse scatter.
        The outermost face ring *is* the first body station, so the rings
        number ``stations - 1`` body plus five face."""
        spec = CLUB_LIBRARY[name]
        stations = 3 * (len(profile_for(spec).sections) - 1) + 1
        assert _metrics(name).profile_points == 2 * (stations - 1 + 5) + 2

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_rendered_normals_are_unit_and_one_per_triangle(self, name: str) -> None:
        mesh = parametric_head_mesh(CLUB_LIBRARY[name])
        normals = np.asarray(mesh.normals)
        assert normals.shape == (np.asarray(mesh.triangles).shape[0], 3)
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)


class TestCenterPivotRegression:
    """Anti-regression: center-pivot loft must never come back (#4799).

    Each gate computes the leading-edge station the pre-#4799 generator
    produced — the authored leading edge rotated about the face center —
    and proves the shipped mesh does not land there. Revert the lean and
    every test in this class fails.
    """

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_mesh_does_not_match_center_pivot_leading_edge(self, name: str) -> None:
        metrics = _metrics(name)
        assert metrics.leading_edge_x < metrics.center_pivot_le_x - 0.1, (
            f"{name}: leading edge at {metrics.leading_edge_x:.3f} mm matches "
            f"the center-pivot station {metrics.center_pivot_le_x:.3f} mm — "
            "center-pivot loft has been reintroduced"
        )

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_leading_edge_stays_on_the_authored_station(self, name: str) -> None:
        """The rendered leading edge is the authored one, untouched by
        loft — the lean's fixed line, measured through the mesh."""
        metrics = _metrics(name)
        assert metrics.leading_edge_x == pytest.approx(
            metrics.authored_le_x, abs=1.0e-9
        )
        assert metrics.leading_edge_y == pytest.approx(
            metrics.authored_le_y, abs=1.0e-9
        )

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_forward_kick_equals_the_epic_closed_form(self, name: str) -> None:
        """The avoided throw is ``sin(loft) * half face height`` (plus the
        sagitta term on curved faces) — epic #4799's measured root cause,
        and a pure function of the authored profile."""
        spec = CLUB_LIBRARY[name]
        lam = math.radians(spec.loft_deg)
        half_height = _authored_face_height_m(spec) * 0.5 * 1.0e3
        metrics = _metrics(name)
        unlofted_face_x = face_center_point(replace(spec, loft_deg=0.0))[0] * 1.0e3
        sagitta = unlofted_face_x - metrics.authored_le_x
        expected = half_height * math.sin(lam) + sagitta * (1.0 - math.cos(lam))
        assert metrics.forward_kick == pytest.approx(expected, abs=1.0e-6)
        assert metrics.forward_kick > 0.1

    @pytest.mark.parametrize("name", _BLADES)
    def test_center_pivot_would_reintroduce_blade_onset(self, name: str) -> None:
        """A center-pivot blade juts its leading edge ~14-26 mm past the
        shaft; the shipped head keeps it inside a few millimeters."""
        metrics = _metrics(name)
        assert metrics.center_pivot_offset >= 13.0
        assert metrics.offset <= 6.0
        assert metrics.center_pivot_offset >= 2.5 * metrics.offset

    def test_sand_wedge_reproduces_the_epics_measured_onset(self) -> None:
        """The epic's headline number: a center-pivot sand wedge lands its
        leading edge ~34 mm out with the hosel at ~8.5 mm — ~25 mm of
        onset. The shipped wedge sits at ~3.5 mm."""
        metrics = _metrics("Sand Wedge")
        assert metrics.center_pivot_le_x == pytest.approx(33.6, abs=0.2)
        assert metrics.center_pivot_offset == pytest.approx(25.1, abs=0.2)
        assert metrics.forward_kick == pytest.approx(21.6, abs=0.2)
        assert metrics.offset == pytest.approx(3.5, abs=0.2)
