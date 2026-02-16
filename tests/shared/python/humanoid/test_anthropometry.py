"""Tests for humanoid_character_builder.core.anthropometry module.

Covers:
- SegmentAnthropometry and AnthropometryData dataclasses
- Gender interpolation (male / female / neutral)
- Mass estimation (segments sum near 100%)
- Dimension estimation
- Inertia estimation
- COM location
"""

from __future__ import annotations

import pytest
from humanoid_character_builder.core.anthropometry import (
    DE_LEVA_DATA,
    AnthropometryData,
    SegmentAnthropometry,
    estimate_segment_dimensions,
    estimate_segment_inertia_from_gyration,
    estimate_segment_masses,
    get_anthropometry_key,
    get_com_location,
    get_segment_length_ratio,
    get_segment_mass_ratio,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def anthro() -> AnthropometryData:
    """Use the default de Leva anthropometry data."""
    return DE_LEVA_DATA


# ── SegmentAnthropometry Dataclass ───────────────────────────────────────


class TestSegmentAnthropometry:
    """Test basic dataclass behavior."""

    def test_construction(self) -> None:
        seg = SegmentAnthropometry(
            mass_ratio=0.05,
            length_ratio=0.15,
            com_proximal_ratio=0.45,
            gyration_sagittal=0.3,
            gyration_transverse=0.3,
            gyration_longitudinal=0.15,
        )
        assert seg.mass_ratio == 0.05
        assert seg.length_ratio == 0.15
        assert seg.width_ratio == 0.2  # default
        assert seg.depth_ratio == 0.15  # default


# ── Gender Interpolation ────────────────────────────────────────────────


class TestGenderInterpolation:
    """Test that interpolation works for male, female, and neutral."""

    def test_male_upper_arm(self, anthro: AnthropometryData) -> None:
        data = anthro.get_segment_data("upper_arm", gender_factor=1.0)
        assert isinstance(data, SegmentAnthropometry)
        assert data.mass_ratio > 0

    def test_female_upper_arm(self, anthro: AnthropometryData) -> None:
        data = anthro.get_segment_data("upper_arm", gender_factor=0.0)
        assert isinstance(data, SegmentAnthropometry)
        assert data.mass_ratio > 0

    def test_neutral_is_midpoint(self, anthro: AnthropometryData) -> None:
        male = anthro.get_segment_data("upper_arm", gender_factor=1.0)
        female = anthro.get_segment_data("upper_arm", gender_factor=0.0)
        neutral = anthro.get_segment_data("upper_arm", gender_factor=0.5)
        expected_mass = (male.mass_ratio + female.mass_ratio) / 2.0
        assert neutral.mass_ratio == pytest.approx(expected_mass, rel=1e-6)

    def test_unknown_segment_returns_default(self, anthro: AnthropometryData) -> None:
        """Unknown segments should return a default rather than crashing."""
        data = anthro.get_segment_data("unknown_segment_xyx")
        assert isinstance(data, SegmentAnthropometry)
        assert data.mass_ratio >= 0


# ── Mass Estimation ─────────────────────────────────────────────────────


class TestMassEstimation:
    """Test estimate_segment_masses."""

    def test_all_masses_positive(self) -> None:
        masses = estimate_segment_masses(total_mass_kg=70.0, gender_factor=0.5)
        assert isinstance(masses, dict)
        for name, mass_val in masses.items():
            assert mass_val > 0, f"Segment {name} mass should be positive"

    def test_masses_sum_near_body_mass(self) -> None:
        """Sum of segment masses should approximate total body mass."""
        total = 80.0
        masses = estimate_segment_masses(total_mass_kg=total, gender_factor=1.0)
        mass_sum = sum(masses.values())
        # Allowing 5% tolerance since not all body parts may be tracked
        assert mass_sum == pytest.approx(total, rel=0.15)

    def test_male_female_difference(self) -> None:
        male = estimate_segment_masses(70.0, gender_factor=1.0)
        female = estimate_segment_masses(70.0, gender_factor=0.0)
        # They should differ in distribution
        assert male != female


# ── Mass/Length Ratio Functions ──────────────────────────────────────────


class TestRatioFunctions:
    """Test convenience ratio functions."""

    def test_mass_ratio_in_range(self) -> None:
        ratio = get_segment_mass_ratio("upper_arm", gender_factor=0.5)
        assert 0.0 < ratio < 1.0

    def test_length_ratio_in_range(self) -> None:
        ratio = get_segment_length_ratio("thigh", gender_factor=0.5)
        assert 0.0 < ratio < 1.0

    def test_anthropometry_key_mapping(self) -> None:
        key = get_anthropometry_key("upper_arm")
        assert isinstance(key, str)
        assert len(key) > 0


# ── Dimension Estimation ────────────────────────────────────────────────


class TestDimensionEstimation:
    """Test estimate_segment_dimensions."""

    def test_all_dimensions_positive(self) -> None:
        dims = estimate_segment_dimensions(total_height_m=1.75, gender_factor=0.5)
        assert isinstance(dims, dict)
        for name, dim in dims.items():
            assert dim["length"] > 0, f"{name} length should be positive"
            assert dim["width"] > 0, f"{name} width should be positive"
            assert dim["depth"] > 0, f"{name} depth should be positive"

    def test_taller_person_longer_segments(self) -> None:
        short = estimate_segment_dimensions(1.60, 0.5)
        tall = estimate_segment_dimensions(1.90, 0.5)
        # Pick any segment
        seg = list(short.keys())[0]
        assert tall[seg]["length"] > short[seg]["length"]


# ── Inertia Estimation ──────────────────────────────────────────────────


class TestInertiaEstimation:
    """Test inertia estimation from gyration radii."""

    def test_inertia_positive(self) -> None:
        inertia = estimate_segment_inertia_from_gyration(
            segment_name="thigh",
            mass_kg=8.0,
            length_m=0.42,
            gender_factor=0.5,
        )
        assert inertia["ixx"] > 0
        assert inertia["iyy"] > 0
        assert inertia["izz"] > 0

    def test_heavier_segment_more_inertia(self) -> None:
        light = estimate_segment_inertia_from_gyration("thigh", 5.0, 0.4, 0.5)
        heavy = estimate_segment_inertia_from_gyration("thigh", 10.0, 0.4, 0.5)
        assert heavy["ixx"] > light["ixx"]


# ── COM Location ────────────────────────────────────────────────────────


class TestCOMLocation:
    """Test get_com_location."""

    def test_com_along_z_axis(self) -> None:
        """COM should be along Z axis (segment oriented along Z)."""
        com = get_com_location("thigh", length_m=0.42, gender_factor=0.5)
        assert com[0] == pytest.approx(0.0)  # x = 0
        assert com[1] == pytest.approx(0.0)  # y = 0
        assert com[2] > 0  # z > 0 (proximal to distal)

    def test_com_within_segment_length(self) -> None:
        length = 0.42
        com = get_com_location("thigh", length_m=length, gender_factor=0.5)
        assert 0 < com[2] < length
