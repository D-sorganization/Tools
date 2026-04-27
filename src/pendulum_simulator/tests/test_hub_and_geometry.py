"""Extended tests for hub_options.py and segment_geometry.py.

hub_options: effective_hub_mass, make_massless_hub_params,
             compute_system_com, hub_offset_for_com
segment_geometry: cylinder_cross_section, ellipsoid_cross_section,
                  tapered_cylinder_cross_section, project_3d_to_2d,
                  depth_sort_segments, auto_radius_from_mass, SegmentStyle
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.hub_options import (
    compute_system_com,
    effective_hub_mass,
    hub_offset_for_com,
    make_massless_hub_params,
)
from double_pendulum_golf.physics_golfer import GolferParams, N_DOF
from double_pendulum_golf.segment_geometry import (
    SegmentStyle,
    auto_radius_from_mass,
    cylinder_cross_section,
    depth_sort_segments,
    ellipsoid_cross_section,
    project_3d_to_2d,
    tapered_cylinder_cross_section,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def params() -> GolferParams:
    return GolferParams(
        m_hub=2.0,
        m_r_upper=3.0,
        m_r_fore=2.0,
        m_l_upper=3.0,
        m_l_fore=2.0,
        m_club=0.5,
        L_hub=0.15,
        L_r_upper=0.35,
        L_r_fore=0.30,
        L_l_upper=0.35,
        L_l_fore=0.30,
        L_club=1.10,
        d_rs=0.20,
        d_ls=0.20,
        grip_right=0.05,
        grip_left=0.25,
        m_clubhead=0.2,
    )


@pytest.fixture
def zero_q() -> np.ndarray:
    return np.zeros(N_DOF)


# ===========================================================================
# Tests for hub_options.py
# ===========================================================================


class TestEffectiveHubMass:
    def test_nominal_returned(self) -> None:
        assert effective_hub_mass(5.0) == pytest.approx(5.0)

    def test_massless_returns_epsilon(self) -> None:
        result = effective_hub_mass(5.0, massless=True)
        assert result > 0
        assert result < 1e-4

    def test_always_positive(self) -> None:
        assert effective_hub_mass(0.01) > 0
        assert effective_hub_mass(0.01, massless=True) > 0

    def test_nominal_not_massless(self) -> None:
        result = effective_hub_mass(2.0, massless=False)
        assert result == pytest.approx(2.0)


class TestMakeMasslessHubParams:
    def test_returns_golfer_params(self, params: GolferParams) -> None:
        result = make_massless_hub_params(params)
        assert isinstance(result, GolferParams)

    def test_m_hub_near_zero(self, params: GolferParams) -> None:
        result = make_massless_hub_params(params)
        assert result.m_hub < 1e-4
        assert result.m_hub > 0

    def test_other_params_unchanged(self, params: GolferParams) -> None:
        result = make_massless_hub_params(params)
        assert result.m_r_upper == params.m_r_upper
        assert result.m_club == params.m_club
        assert result.L_club == params.L_club

    def test_original_unchanged(self, params: GolferParams) -> None:
        original_m_hub = params.m_hub
        _ = make_massless_hub_params(params)
        assert params.m_hub == original_m_hub  # original not mutated


class TestComputeSystemCom:
    def test_shape(self, params: GolferParams, zero_q: np.ndarray) -> None:
        com = compute_system_com(zero_q, params)
        assert com.shape == (2,)

    def test_finite(self, params: GolferParams, zero_q: np.ndarray) -> None:
        com = compute_system_com(zero_q, params)
        assert np.all(np.isfinite(com))

    def test_extended_state_handled(self, params: GolferParams) -> None:
        """State with 16 elements (pos + vel) should be truncated to N_DOF."""
        state_full = np.zeros(2 * N_DOF)
        com = compute_system_com(state_full, params)
        assert com.shape == (2,)
        assert np.all(np.isfinite(com))

    def test_off_center_with_nonzero_q(self, params: GolferParams) -> None:
        q_angled = np.zeros(N_DOF)
        q_angled[0] = np.pi / 4  # hub angle
        com = compute_system_com(q_angled, params)
        assert np.all(np.isfinite(com))


class TestHubOffsetForCom:
    def test_returns_tuple(self, params: GolferParams, zero_q: np.ndarray) -> None:
        result = hub_offset_for_com(zero_q, params)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_finite(self, params: GolferParams, zero_q: np.ndarray) -> None:
        dx, dy = hub_offset_for_com(zero_q, params)
        assert np.isfinite(dx)
        assert np.isfinite(dy)

    def test_matches_com(self, params: GolferParams, zero_q: np.ndarray) -> None:
        com = compute_system_com(zero_q, params)
        dx, dy = hub_offset_for_com(zero_q, params)
        assert dx == pytest.approx(float(com[0]))
        assert dy == pytest.approx(float(com[1]))


# ===========================================================================
# Tests for segment_geometry.py
# ===========================================================================


class TestSegmentStyle:
    def test_enum_values(self) -> None:
        assert SegmentStyle.LINE.value == "line"
        assert SegmentStyle.CYLINDER.value == "cylinder"
        assert SegmentStyle.ELLIPSOID.value == "ellipsoid"
        assert SegmentStyle.TAPERED.value == "tapered"


class TestCylinderCrossSection:
    def test_shape(self) -> None:
        start = np.array([0.0, 0.0])
        end = np.array([1.0, 0.0])
        corners = cylinder_cross_section(start, end, radius=0.1)
        assert corners.shape == (4, 2)

    def test_finite(self) -> None:
        start = np.array([0.0, 0.0])
        end = np.array([0.5, 0.5])
        corners = cylinder_cross_section(start, end, radius=0.05)
        assert np.all(np.isfinite(corners))

    def test_negative_radius_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            cylinder_cross_section(
                np.array([0.0, 0.0]), np.array([1.0, 0.0]), radius=-0.1
            )

    def test_degenerate_segment(self) -> None:
        """Zero-length segment should not crash."""
        start = np.array([0.5, 0.5])
        end = np.array([0.5, 0.5])
        corners = cylinder_cross_section(start, end, radius=0.1)
        assert corners.shape == (4, 2)

    def test_symmetric_about_axis(self) -> None:
        """Corners should be symmetric about the segment axis."""
        start = np.array([0.0, 0.0])
        end = np.array([1.0, 0.0])
        r = 0.1
        corners = cylinder_cross_section(start, end, radius=r)
        # Top and bottom corners should be at +r and -r in y
        y_coords = corners[:, 1]
        assert max(y_coords) == pytest.approx(r, abs=1e-10)
        assert min(y_coords) == pytest.approx(-r, abs=1e-10)


class TestEllipsoidCrossSection:
    def test_shape_default(self) -> None:
        centre = np.array([0.0, 0.0])
        pts = ellipsoid_cross_section(centre, semi_a=0.5, semi_b=0.2)
        assert pts.shape == (32, 2)

    def test_shape_custom_n_points(self) -> None:
        centre = np.array([1.0, 2.0])
        pts = ellipsoid_cross_section(centre, semi_a=0.3, semi_b=0.1, n_points=16)
        assert pts.shape == (16, 2)

    def test_finite(self) -> None:
        centre = np.array([0.0, 0.0])
        pts = ellipsoid_cross_section(centre, semi_a=0.4, semi_b=0.2, angle=np.pi / 4)
        assert np.all(np.isfinite(pts))

    def test_negative_semi_axis_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            ellipsoid_cross_section(np.zeros(2), semi_a=-0.1, semi_b=0.2)

    def test_centred_at_origin(self) -> None:
        centre = np.array([0.0, 0.0])
        pts = ellipsoid_cross_section(centre, semi_a=1.0, semi_b=0.5)
        # x should range from -1 to 1, y from -0.5 to 0.5
        assert pts[:, 0].min() == pytest.approx(-1.0, abs=1e-10)
        assert pts[:, 0].max() == pytest.approx(1.0, abs=0.01)

    def test_too_few_points_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            ellipsoid_cross_section(np.zeros(2), 0.5, 0.3, n_points=2)


class TestTaperedCylinderCrossSection:
    def test_shape(self) -> None:
        start = np.array([0.0, 0.0])
        end = np.array([0.0, 1.0])
        corners = tapered_cylinder_cross_section(
            start, end, radius_start=0.2, radius_end=0.05
        )
        assert corners.shape == (4, 2)

    def test_finite(self) -> None:
        start = np.array([0.0, 0.0])
        end = np.array([1.0, 0.5])
        corners = tapered_cylinder_cross_section(start, end, 0.1, 0.05)
        assert np.all(np.isfinite(corners))

    def test_degenerate_falls_back(self) -> None:
        """Zero-length segment should return 4 corners without crashing."""
        pt = np.array([0.0, 0.0])
        corners = tapered_cylinder_cross_section(pt, pt, 0.1, 0.05)
        assert corners.shape == (4, 2)

    def test_negative_radius_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            tapered_cylinder_cross_section(
                np.array([0.0, 0.0]),
                np.array([1.0, 0.0]),
                radius_start=-0.1,
                radius_end=0.1,
            )


class TestProject3dTo2d:
    def test_identity_no_rotation(self) -> None:
        """With zero tilt and azimuth, x and y should be preserved."""
        pt = np.array([1.0, 2.0, 0.0])
        proj = project_3d_to_2d(pt, tilt=0.0, azimuth=0.0)
        assert proj.shape == (2,)
        assert proj[0] == pytest.approx(1.0, abs=1e-10)
        assert proj[1] == pytest.approx(2.0, abs=1e-10)

    def test_returns_array_shape(self) -> None:
        pt = np.array([0.5, 1.0, 0.3])
        proj = project_3d_to_2d(pt, tilt=0.1, azimuth=0.2)
        assert proj.shape == (2,)

    def test_return_depth(self) -> None:
        pt = np.array([0.5, 1.0, 2.0])
        result = project_3d_to_2d(pt, return_depth=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        proj, depth = result
        assert proj.shape == (2,)
        assert isinstance(depth, float)

    def test_finite(self) -> None:
        pt = np.array([1.0, 2.0, 3.0])
        proj = project_3d_to_2d(pt, tilt=np.pi / 6, azimuth=np.pi / 4)
        assert np.all(np.isfinite(proj))


class TestDepthSortSegments:
    def test_sorted_descending(self) -> None:
        segs = [{"depth": 1.0}, {"depth": 3.0}, {"depth": 2.0}]
        result = depth_sort_segments(segs)
        depths = [s["depth"] for s in result]
        assert depths == sorted(depths, reverse=True)

    def test_empty_list(self) -> None:
        assert depth_sort_segments([]) == []

    def test_preserves_other_keys(self) -> None:
        segs = [{"depth": 2.0, "color": "red"}, {"depth": 5.0, "color": "blue"}]
        result = depth_sort_segments(segs)
        assert result[0]["color"] == "blue"
        assert result[1]["color"] == "red"


class TestAutoRadiusFromMass:
    def test_positive_result(self) -> None:
        r = auto_radius_from_mass(2.0, 1.0)
        assert r > 0

    def test_scales_with_mass(self) -> None:
        r1 = auto_radius_from_mass(1.0, 1.0)
        r2 = auto_radius_from_mass(4.0, 1.0)
        assert r2 == pytest.approx(2 * r1)  # sqrt(4) = 2

    def test_finite(self) -> None:
        assert np.isfinite(auto_radius_from_mass(0.5, 0.3))

    def test_negative_mass_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            auto_radius_from_mass(-1.0, 1.0)

    def test_negative_length_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            auto_radius_from_mass(1.0, -1.0)
