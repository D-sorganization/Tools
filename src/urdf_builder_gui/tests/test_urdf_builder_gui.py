"""TDD tests for the extracted URDF builder modules.

Tests cover:
  - anthropometric_model: HEIGHT_RATIOS, MASS_RATIOS, inertia computations,
    segment dimensions, template definitions, DbC contract enforcement.
  - urdf_generator: generate_urdf_xml, validate_urdf_structure, template
    dispatch, config propagation, XML validity.

Replaces the old mock-heavy tests with real functional tests.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from urdf_builder_gui.anthropometric_model import (
    HEIGHT_RATIOS,
    MASS_RATIOS,
    URDFConfig,
    compute_box_inertia,
    compute_cylinder_inertia,
    compute_segment_length,
    compute_segment_mass,
    compute_sphere_inertia,
    get_template_segments,
    interpolate_gender_factor,
)
from urdf_builder_gui.contracts import PreconditionError
from urdf_builder_gui.urdf_generator import (
    generate_urdf_xml,
    validate_urdf_structure,
)

# ═══════════════════════════════════════════════════════════════════════
# Anthropometric Model Tests
# ═══════════════════════════════════════════════════════════════════════


class TestHeightRatios:
    """Tests for HEIGHT_RATIOS constants."""

    def test_ratios_sum_reasonable(self) -> None:
        """Core ratios should be in a reasonable range."""
        core = (
            HEIGHT_RATIOS["pelvis"]
            + HEIGHT_RATIOS["torso"]
            + HEIGHT_RATIOS["head"]
            + HEIGHT_RATIOS["thigh"]
            + HEIGHT_RATIOS["shin"]
        )
        # Head + torso + pelvis + thigh + shin ≈ full body
        assert 0.85 < core < 1.05

    def test_all_ratios_positive(self) -> None:
        """Every ratio must be positive."""
        for key, val in HEIGHT_RATIOS.items():
            assert val > 0, f"Ratio for {key} must be positive"


class TestMassRatios:
    """Tests for MASS_RATIOS constants."""

    def test_total_mass_approximately_one(self) -> None:
        """Total mass with bilateral segments should ≈ 1.0."""
        bilateral = {"upper_arm", "forearm", "hand", "thigh", "shin", "foot"}
        # Exclude the combined "torso" entry to avoid double-counting
        skip = {"torso"}
        total = 0.0
        for key, val in MASS_RATIOS.items():
            if key in skip:
                continue
            total += val * 2 if key in bilateral else val
        assert 0.9 < total < 1.1, f"Total mass ratio = {total}"

    def test_all_ratios_positive(self) -> None:
        for key, val in MASS_RATIOS.items():
            assert val > 0, f"Mass ratio for {key} must be positive"


class TestComputeSegmentLength:
    """Tests for compute_segment_length."""

    def test_default_proportion(self) -> None:
        """Default factor = 1.0 should return height × ratio."""
        length = compute_segment_length(1.80, "thigh")
        expected = 1.80 * HEIGHT_RATIOS["thigh"]
        assert length == pytest.approx(expected)

    def test_scaled_proportion(self) -> None:
        """Factor = 1.5 should scale the result by 1.5."""
        base = compute_segment_length(1.80, "thigh", 1.0)
        scaled = compute_segment_length(1.80, "thigh", 1.5)
        assert scaled == pytest.approx(base * 1.5)

    def test_negative_height_raises(self) -> None:
        with pytest.raises((PreconditionError, AssertionError)):
            compute_segment_length(-1.0, "thigh")

    def test_unknown_segment_raises(self) -> None:
        with pytest.raises((PreconditionError, AssertionError)):
            compute_segment_length(1.80, "nonexistent_segment")


class TestComputeSegmentMass:
    """Tests for compute_segment_mass."""

    def test_single_segment(self) -> None:
        mass = compute_segment_mass(70.0, "pelvis")
        assert mass == pytest.approx(70.0 * MASS_RATIOS["pelvis"])

    def test_bilateral_count(self) -> None:
        single = compute_segment_mass(70.0, "thigh", count=1)
        double = compute_segment_mass(70.0, "thigh", count=2)
        assert double == pytest.approx(single * 2)

    def test_zero_mass_raises(self) -> None:
        with pytest.raises((PreconditionError, AssertionError)):
            compute_segment_mass(0.0, "pelvis")


class TestBoxInertia:
    """Tests for compute_box_inertia."""

    def test_unit_cube(self) -> None:
        """Unit cube with unit mass: I = 1/6 on each axis."""
        ixx, iyy, izz = compute_box_inertia(1.0, 1.0, 1.0, 1.0)
        assert ixx == pytest.approx(1.0 / 6.0)
        assert iyy == pytest.approx(1.0 / 6.0)
        assert izz == pytest.approx(1.0 / 6.0)

    def test_all_positive(self) -> None:
        ixx, iyy, izz = compute_box_inertia(10.0, 0.2, 0.5, 0.1)
        assert ixx > 0
        assert iyy > 0
        assert izz > 0

    def test_negative_mass_raises(self) -> None:
        with pytest.raises((PreconditionError, AssertionError)):
            compute_box_inertia(-1.0, 0.1, 0.1, 0.1)


class TestCylinderInertia:
    """Tests for compute_cylinder_inertia."""

    def test_all_positive(self) -> None:
        ixx, iyy, izz = compute_cylinder_inertia(5.0, 0.05, 0.4)
        assert ixx > 0
        assert iyy > 0
        assert izz > 0

    def test_transverse_equals(self) -> None:
        """Transverse axes should be equal for a cylinder."""
        ixx, iyy, _ = compute_cylinder_inertia(5.0, 0.05, 0.4)
        assert ixx == pytest.approx(iyy)


class TestSphereInertia:
    """Tests for compute_sphere_inertia."""

    def test_all_equal(self) -> None:
        """All axes should be equal for a sphere."""
        ixx, iyy, izz = compute_sphere_inertia(5.0, 0.1)
        assert ixx == pytest.approx(iyy)
        assert iyy == pytest.approx(izz)

    def test_known_value(self) -> None:
        ixx, _, _ = compute_sphere_inertia(1.0, 1.0)
        assert ixx == pytest.approx(2.0 / 5.0)


class TestInterpolateGender:
    """Tests for interpolate_gender_factor."""

    def test_endpoints(self) -> None:
        assert interpolate_gender_factor(0.0, 10.0, 20.0) == pytest.approx(10.0)
        assert interpolate_gender_factor(1.0, 10.0, 20.0) == pytest.approx(20.0)

    def test_midpoint(self) -> None:
        assert interpolate_gender_factor(0.5, 10.0, 20.0) == pytest.approx(15.0)

    def test_clamping(self) -> None:
        """Out-of-range values should be clamped."""
        assert interpolate_gender_factor(-1.0, 10.0, 20.0) == pytest.approx(10.0)
        assert interpolate_gender_factor(2.0, 10.0, 20.0) == pytest.approx(20.0)


class TestTemplateSegments:
    """Tests for TEMPLATE_SEGMENTS and get_template_segments."""

    def test_full_humanoid_has_all_body_parts(self) -> None:
        segs = get_template_segments("Full Humanoid")
        assert "pelvis" in segs
        assert "torso" in segs
        assert "head" in segs
        assert "thigh_l" in segs
        assert "upper_arm_r" in segs

    def test_upper_body_no_legs(self) -> None:
        segs = get_template_segments("Upper Body Only")
        assert "upper_arm_l" in segs
        assert "thigh_l" not in segs

    def test_lower_body_no_arms(self) -> None:
        segs = get_template_segments("Lower Body Only")
        assert "thigh_l" in segs
        assert "upper_arm_l" not in segs

    def test_unknown_template_raises(self) -> None:
        with pytest.raises((PreconditionError, AssertionError)):
            get_template_segments("Nonexistent Template")


# ═══════════════════════════════════════════════════════════════════════
# URDF Generator Tests
# ═══════════════════════════════════════════════════════════════════════


class TestGenerateURDF:
    """Tests for generate_urdf_xml."""

    def test_valid_xml_output(self) -> None:
        """Output must be well-formed XML."""
        config = URDFConfig(robot_name="test_robot", height_m=1.75, mass_kg=70.0)
        xml = generate_urdf_xml(config)
        # Should parse without error
        root = ET.fromstring(xml)
        assert root.tag == "robot"
        assert root.attrib["name"] == "test_robot"

    def test_contains_expected_links(self) -> None:
        """Full Humanoid template should produce expected links."""
        config = URDFConfig(template="Full Humanoid")
        xml = generate_urdf_xml(config)
        root = ET.fromstring(xml)
        link_names = {link.get("name") for link in root.findall("link")}
        assert "pelvis" in link_names
        assert "torso" in link_names
        assert "head" in link_names
        assert "thigh_l" in link_names

    def test_upper_body_only_excludes_legs(self) -> None:
        """Upper Body Only template should NOT include leg links."""
        config = URDFConfig(template="Upper Body Only")
        xml = generate_urdf_xml(config)
        root = ET.fromstring(xml)
        link_names = {link.get("name") for link in root.findall("link")}
        assert "upper_arm_l" in link_names
        assert "thigh_l" not in link_names

    def test_inertia_not_placeholder(self) -> None:
        """Inertia values must be computed, not hardcoded 0.01."""
        config = URDFConfig(height_m=1.80, mass_kg=80.0)
        xml = generate_urdf_xml(config)
        # None of the inertia values should be the old placeholder
        assert 'ixx="0.01"' not in xml
        assert 'iyy="0.01"' not in xml

    def test_damping_and_friction_in_joints(self) -> None:
        """Damping and friction from config should appear in joint dynamics."""
        config = URDFConfig(damping=5.0, friction=2.0)
        xml = generate_urdf_xml(config)
        assert 'damping="5.00"' in xml
        assert 'friction="2.00"' in xml

    def test_invalid_robot_name_raises(self) -> None:
        """Invalid XML names should be rejected."""
        with pytest.raises((PreconditionError, AssertionError)):
            generate_urdf_xml(URDFConfig(robot_name="has spaces"))

    def test_empty_robot_name_raises(self) -> None:
        with pytest.raises((PreconditionError, AssertionError)):
            generate_urdf_xml(URDFConfig(robot_name=""))

    def test_zero_height_raises(self) -> None:
        with pytest.raises((PreconditionError, AssertionError)):
            generate_urdf_xml(URDFConfig(height_m=0.0))


class TestValidateURDF:
    """Tests for validate_urdf_structure."""

    def test_valid_urdf_passes(self) -> None:
        config = URDFConfig()
        xml = generate_urdf_xml(config)
        is_valid, errors = validate_urdf_structure(xml)
        assert is_valid, f"Validation errors: {errors}"

    def test_malformed_xml_fails(self) -> None:
        is_valid, errors = validate_urdf_structure("<robot><link name='x'>")
        assert not is_valid
        assert len(errors) > 0

    def test_wrong_root_element(self) -> None:
        is_valid, errors = validate_urdf_structure('<model name="x"/>')
        assert not is_valid
        assert any("root element" in e.lower() for e in errors)

    def test_duplicate_link_names(self) -> None:
        xml = """<?xml version="1.0"?>
<robot name="test">
  <link name="dup"/>
  <link name="dup"/>
</robot>"""
        is_valid, errors = validate_urdf_structure(xml)
        assert not is_valid
        assert any("duplicate" in e.lower() for e in errors)
