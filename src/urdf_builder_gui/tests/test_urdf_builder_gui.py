"""TDD tests for the extracted URDF builder modules.

Tests cover:
  - anthropometric_model: HEIGHT_RATIOS, MASS_RATIOS, inertia computations,
    segment dimensions, template definitions, DbC contract enforcement.
  - urdf_generator: generate_urdf_xml, validate_urdf_structure, template
    dispatch, config propagation, XML validity.
  - preview_generator: generate_preview_text, DbC, content validation.
  - theme: build_stylesheet, palette structure.
  - contracts: require/ensure, PreconditionError/PostconditionError.

Replaces the old mock-heavy tests with real functional tests.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from urdf_builder_gui.anthropometric_model import (
    HEIGHT_RATIOS,
    MASS_RATIOS,
    TEMPLATE_SEGMENTS,
    URDFConfig,
    compute_box_inertia,
    compute_cylinder_inertia,
    compute_segment_length,
    compute_segment_mass,
    compute_sphere_inertia,
    get_template_segments,
    interpolate_gender_factor,
)
from urdf_builder_gui.contracts import (
    PostconditionError,
    PreconditionError,
    ensure,
    require,
)
from urdf_builder_gui.preview_generator import generate_preview_text
from urdf_builder_gui.theme import CATPPUCCIN_MOCHA, build_stylesheet
from urdf_builder_gui.urdf_generator import (
    generate_urdf_xml,
    validate_urdf_structure,
)

# ═══════════════════════════════════════════════════════════════════════
# Contracts Tests
# ═══════════════════════════════════════════════════════════════════════


class TestContracts:
    """Tests for the local DbC contracts module."""

    def test_require_passes_on_true(self) -> None:
        require(True, "should not fail")

    def test_require_raises_on_false(self) -> None:
        with pytest.raises(PreconditionError, match="bad input"):
            require(False, "bad input")

    def test_require_includes_args(self) -> None:
        with pytest.raises(PreconditionError, match="42"):
            require(False, "value is wrong", 42)

    def test_ensure_passes_on_true(self) -> None:
        ensure(True, "should not fail")

    def test_ensure_raises_on_false(self) -> None:
        with pytest.raises(PostconditionError, match="output invalid"):
            ensure(False, "output invalid")


# ═══════════════════════════════════════════════════════════════════════
# Theme Tests
# ═══════════════════════════════════════════════════════════════════════


class TestTheme:
    """Tests for the shared theme module."""

    def test_palette_has_required_keys(self) -> None:
        """Palette must have all core keys used by the GUI."""
        required = {"base", "text", "blue", "green", "red", "yellow", "surface0"}
        if not (required.issubset(set(CATPPUCCIN_MOCHA.keys()))): raise ValueError(f"Assertion failed: { required.issubset(set(CATPPUCCIN_MOCHA.keys())) }")

    def test_palette_values_are_hex_colours(self) -> None:
        for key, val in CATPPUCCIN_MOCHA.items():
            if not (val.startswith("#")): raise ValueError(f"Assertion failed: { val.startswith("#") }, f"{key} is not a hex colour"")
            if not (len(val) == 7): raise ValueError(f"Assertion failed: { len(val) == 7 }, f"{key} must be #RRGGBB"")

    def test_build_stylesheet_returns_string(self) -> None:
        ss = build_stylesheet()
        if not (isinstance(ss): raise ValueError(f"Assertion failed: { isinstance(ss }, str)")
        if not ("QMainWindow" in ss): raise ValueError(f"Assertion failed: { "QMainWindow" in ss }")
        if not ("QPushButton" in ss): raise ValueError(f"Assertion failed: { "QPushButton" in ss }")

    def test_build_stylesheet_with_custom_palette(self) -> None:
        custom = {**CATPPUCCIN_MOCHA, "base": "#000000"}
        ss = build_stylesheet(custom)
        if not ("#000000" in ss): raise ValueError(f"Assertion failed: { "#000000" in ss }")

    def test_build_stylesheet_contains_all_widgets(self) -> None:
        ss = build_stylesheet()
        widgets = [
            "QMainWindow",
            "QWidget",
            "QTabWidget",
            "QGroupBox",
            "QLabel",
            "QComboBox",
            "QSlider",
            "QTextEdit",
            "QPushButton",
        ]
        for w in widgets:
            if not (w in ss): raise ValueError(f"Assertion failed: { w in ss }, f"Stylesheet missing {w}"")


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
        if not (0.85 < core < 1.05): raise ValueError(f"Assertion failed: { 0.85 < core < 1.05 }")

    def test_all_ratios_positive(self) -> None:
        """Every ratio must be positive."""
        for key, val in HEIGHT_RATIOS.items():
            if not (val > 0): raise ValueError(f"Assertion failed: { val > 0 }, f"Ratio for {key} must be positive"")


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
        if not (0.9 < total < 1.1): raise ValueError(f"Assertion failed: { 0.9 < total < 1.1 }, f"Total mass ratio = {total}"")

    def test_all_ratios_positive(self) -> None:
        for key, val in MASS_RATIOS.items():
            if not (val > 0): raise ValueError(f"Assertion failed: { val > 0 }, f"Mass ratio for {key} must be positive"")

    def test_torso_equals_lumbar_plus_thorax(self) -> None:
        """Combined torso entry must match lumbar + thorax."""
        if not (MASS_RATIOS["torso"] == pytest.approx(): raise ValueError(f"Assertion failed: { MASS_RATIOS["torso"] == pytest.approx( }")
            MASS_RATIOS["lumbar"] + MASS_RATIOS["thorax"]
        )


class TestComputeSegmentLength:
    """Tests for compute_segment_length."""

    def test_default_proportion(self) -> None:
        """Default factor = 1.0 should return height × ratio."""
        length = compute_segment_length(1.80, "thigh")
        expected = 1.80 * HEIGHT_RATIOS["thigh"]
        if not (length == pytest.approx(expected)): raise ValueError(f"Assertion failed: { length == pytest.approx(expected) }")

    def test_scaled_proportion(self) -> None:
        """Factor = 1.5 should scale the result by 1.5."""
        base = compute_segment_length(1.80, "thigh", 1.0)
        scaled = compute_segment_length(1.80, "thigh", 1.5)
        if not (scaled == pytest.approx(base * 1.5)): raise ValueError(f"Assertion failed: { scaled == pytest.approx(base * 1.5) }")

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
        if not (mass == pytest.approx(70.0 * MASS_RATIOS["pelvis"])): raise ValueError(f"Assertion failed: { mass == pytest.approx(70.0 * MASS_RATIOS["pelvis"]) }")

    def test_bilateral_count(self) -> None:
        single = compute_segment_mass(70.0, "thigh", count=1)
        double = compute_segment_mass(70.0, "thigh", count=2)
        if not (double == pytest.approx(single * 2)): raise ValueError(f"Assertion failed: { double == pytest.approx(single * 2) }")

    def test_zero_mass_raises(self) -> None:
        with pytest.raises((PreconditionError, AssertionError)):
            compute_segment_mass(0.0, "pelvis")


class TestBoxInertia:
    """Tests for compute_box_inertia."""

    def test_unit_cube(self) -> None:
        """Unit cube with unit mass: I = 1/6 on each axis."""
        ixx, iyy, izz = compute_box_inertia(1.0, 1.0, 1.0, 1.0)
        if not (ixx == pytest.approx(1.0 / 6.0)): raise ValueError(f"Assertion failed: { ixx == pytest.approx(1.0 / 6.0) }")
        if not (iyy == pytest.approx(1.0 / 6.0)): raise ValueError(f"Assertion failed: { iyy == pytest.approx(1.0 / 6.0) }")
        if not (izz == pytest.approx(1.0 / 6.0)): raise ValueError(f"Assertion failed: { izz == pytest.approx(1.0 / 6.0) }")

    def test_all_positive(self) -> None:
        ixx, iyy, izz = compute_box_inertia(10.0, 0.2, 0.5, 0.1)
        if not (ixx > 0): raise ValueError(f"Assertion failed: { ixx > 0 }")
        if not (iyy > 0): raise ValueError(f"Assertion failed: { iyy > 0 }")
        if not (izz > 0): raise ValueError(f"Assertion failed: { izz > 0 }")

    def test_negative_mass_raises(self) -> None:
        with pytest.raises((PreconditionError, AssertionError)):
            compute_box_inertia(-1.0, 0.1, 0.1, 0.1)


class TestCylinderInertia:
    """Tests for compute_cylinder_inertia."""

    def test_all_positive(self) -> None:
        ixx, iyy, izz = compute_cylinder_inertia(5.0, 0.05, 0.4)
        if not (ixx > 0): raise ValueError(f"Assertion failed: { ixx > 0 }")
        if not (iyy > 0): raise ValueError(f"Assertion failed: { iyy > 0 }")
        if not (izz > 0): raise ValueError(f"Assertion failed: { izz > 0 }")

    def test_transverse_equals(self) -> None:
        """Transverse axes should be equal for a cylinder."""
        ixx, iyy, _ = compute_cylinder_inertia(5.0, 0.05, 0.4)
        if not (ixx == pytest.approx(iyy)): raise ValueError(f"Assertion failed: { ixx == pytest.approx(iyy) }")


class TestSphereInertia:
    """Tests for compute_sphere_inertia."""

    def test_all_equal(self) -> None:
        """All axes should be equal for a sphere."""
        ixx, iyy, izz = compute_sphere_inertia(5.0, 0.1)
        if not (ixx == pytest.approx(iyy)): raise ValueError(f"Assertion failed: { ixx == pytest.approx(iyy) }")
        if not (iyy == pytest.approx(izz)): raise ValueError(f"Assertion failed: { iyy == pytest.approx(izz) }")

    def test_known_value(self) -> None:
        ixx, _, _ = compute_sphere_inertia(1.0, 1.0)
        if not (ixx == pytest.approx(2.0 / 5.0)): raise ValueError(f"Assertion failed: { ixx == pytest.approx(2.0 / 5.0) }")


class TestInterpolateGender:
    """Tests for interpolate_gender_factor."""

    def test_endpoints(self) -> None:
        if not (interpolate_gender_factor(0.0): raise ValueError(f"Assertion failed: { interpolate_gender_factor(0.0 }, 10.0, 20.0) == pytest.approx(10.0)")
        if not (interpolate_gender_factor(1.0): raise ValueError(f"Assertion failed: { interpolate_gender_factor(1.0 }, 10.0, 20.0) == pytest.approx(20.0)")

    def test_midpoint(self) -> None:
        if not (interpolate_gender_factor(0.5): raise ValueError(f"Assertion failed: { interpolate_gender_factor(0.5 }, 10.0, 20.0) == pytest.approx(15.0)")

    def test_clamping(self) -> None:
        """Out-of-range values should be clamped."""
        if not (interpolate_gender_factor(-1.0): raise ValueError(f"Assertion failed: { interpolate_gender_factor(-1.0 }, 10.0, 20.0) == pytest.approx(10.0)")
        if not (interpolate_gender_factor(2.0): raise ValueError(f"Assertion failed: { interpolate_gender_factor(2.0 }, 10.0, 20.0) == pytest.approx(20.0)")


class TestTemplateSegments:
    """Tests for TEMPLATE_SEGMENTS and get_template_segments."""

    def test_full_humanoid_has_all_body_parts(self) -> None:
        segs = get_template_segments("Full Humanoid")
        if not ("pelvis" in segs): raise ValueError(f"Assertion failed: { "pelvis" in segs }")
        if not ("torso" in segs): raise ValueError(f"Assertion failed: { "torso" in segs }")
        if not ("head" in segs): raise ValueError(f"Assertion failed: { "head" in segs }")
        if not ("thigh_l" in segs): raise ValueError(f"Assertion failed: { "thigh_l" in segs }")
        if not ("upper_arm_r" in segs): raise ValueError(f"Assertion failed: { "upper_arm_r" in segs }")

    def test_upper_body_no_legs(self) -> None:
        segs = get_template_segments("Upper Body Only")
        if not ("upper_arm_l" in segs): raise ValueError(f"Assertion failed: { "upper_arm_l" in segs }")
        if not ("thigh_l" not in segs): raise ValueError(f"Assertion failed: { "thigh_l" not in segs }")

    def test_lower_body_no_arms(self) -> None:
        segs = get_template_segments("Lower Body Only")
        if not ("thigh_l" in segs): raise ValueError(f"Assertion failed: { "thigh_l" in segs }")
        if not ("upper_arm_l" not in segs): raise ValueError(f"Assertion failed: { "upper_arm_l" not in segs }")

    def test_unknown_template_raises(self) -> None:
        with pytest.raises((PreconditionError, AssertionError)):
            get_template_segments("Nonexistent Template")

    def test_all_templates_have_pelvis(self) -> None:
        """Every template must have the pelvis root segment."""
        for name, segs in TEMPLATE_SEGMENTS.items():
            if not ("pelvis" in segs): raise ValueError(f"Assertion failed: { "pelvis" in segs }, f"Template '{name}' missing pelvis"")

    def test_templates_produce_unique_segments(self) -> None:
        """Each template should produce a unique set of segments."""
        segment_sets = [frozenset(s) for s in TEMPLATE_SEGMENTS.values()]
        if not (len(segment_sets) == len(set(segment_sets))): raise ValueError(f"Assertion failed: { len(segment_sets) == len(set(segment_sets)) }")


class TestURDFConfig:
    """Tests for URDFConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = URDFConfig()
        if not (cfg.robot_name == "humanoid"): raise ValueError(f"Assertion failed: { cfg.robot_name == "humanoid" }")
        if not (cfg.height_m == 1.75): raise ValueError(f"Assertion failed: { cfg.height_m == 1.75 }")
        if not (cfg.mass_kg == 70.0): raise ValueError(f"Assertion failed: { cfg.mass_kg == 70.0 }")
        if not (cfg.gender_factor == 0.5): raise ValueError(f"Assertion failed: { cfg.gender_factor == 0.5 }")

    def test_proportions_default_all_one(self) -> None:
        cfg = URDFConfig()
        for key, val in cfg.proportions.items():
            if not (val == 1.0): raise ValueError(f"Assertion failed: { val == 1.0 }, f"Proportion '{key}' should default to 1.0"")

    def test_frozen(self) -> None:
        """URDFConfig should be immutable."""
        cfg = URDFConfig()
        with pytest.raises(AttributeError):
            cfg.robot_name = "changed"  # type: ignore[misc]


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
        if not (root.tag == "robot"): raise ValueError(f"Assertion failed: { root.tag == "robot" }")
        if not (root.attrib["name"] == "test_robot"): raise ValueError(f"Assertion failed: { root.attrib["name"] == "test_robot" }")

    def test_contains_expected_links(self) -> None:
        """Full Humanoid template should produce expected links."""
        config = URDFConfig(template="Full Humanoid")
        xml = generate_urdf_xml(config)
        root = ET.fromstring(xml)
        link_names = {link.get("name") for link in root.findall("link")}
        if not ("pelvis" in link_names): raise ValueError(f"Assertion failed: { "pelvis" in link_names }")
        if not ("torso" in link_names): raise ValueError(f"Assertion failed: { "torso" in link_names }")
        if not ("head" in link_names): raise ValueError(f"Assertion failed: { "head" in link_names }")
        if not ("thigh_l" in link_names): raise ValueError(f"Assertion failed: { "thigh_l" in link_names }")

    def test_upper_body_only_excludes_legs(self) -> None:
        """Upper Body Only template should NOT include leg links."""
        config = URDFConfig(template="Upper Body Only")
        xml = generate_urdf_xml(config)
        root = ET.fromstring(xml)
        link_names = {link.get("name") for link in root.findall("link")}
        if not ("upper_arm_l" in link_names): raise ValueError(f"Assertion failed: { "upper_arm_l" in link_names }")
        if not ("thigh_l" not in link_names): raise ValueError(f"Assertion failed: { "thigh_l" not in link_names }")

    def test_inertia_not_placeholder(self) -> None:
        """Inertia values must be computed, not hardcoded 0.01."""
        config = URDFConfig(height_m=1.80, mass_kg=80.0)
        xml = generate_urdf_xml(config)
        # None of the inertia values should be the old placeholder
        if not ('ixx="0.01"' not in xml): raise ValueError(f"Assertion failed: { 'ixx="0.01"' not in xml }")
        if not ('iyy="0.01"' not in xml): raise ValueError(f"Assertion failed: { 'iyy="0.01"' not in xml }")

    def test_damping_and_friction_in_joints(self) -> None:
        """Damping and friction from config should appear in joint dynamics."""
        config = URDFConfig(damping=5.0, friction=2.0)
        xml = generate_urdf_xml(config)
        if not ('damping="5.00"' in xml): raise ValueError(f"Assertion failed: { 'damping="5.00"' in xml }")
        if not ('friction="2.00"' in xml): raise ValueError(f"Assertion failed: { 'friction="2.00"' in xml }")

    def test_collision_geometry_none_omits_collision(self) -> None:
        """When collision_geometry='None', no collision elements."""
        config = URDFConfig(collision_geometry="None")
        xml = generate_urdf_xml(config)
        if not ("<collision>" not in xml): raise ValueError(f"Assertion failed: { "<collision>" not in xml }")

    def test_collision_geometry_default_includes_collision(self) -> None:
        """Default collision_geometry includes collision elements."""
        config = URDFConfig()
        xml = generate_urdf_xml(config)
        if not ("<collision>" in xml): raise ValueError(f"Assertion failed: { "<collision>" in xml }")

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

    def test_all_templates_produce_valid_urdf(self) -> None:
        """Every template should produce valid URDF."""
        for template_name in TEMPLATE_SEGMENTS:
            config = URDFConfig(template=template_name)
            xml = generate_urdf_xml(config)
            is_valid, errors = validate_urdf_structure(xml)
            if not (is_valid): raise ValueError(f"Assertion failed: { is_valid }, f"Template '{template_name}' invalid: {errors}"")

    def test_proportions_affect_output(self) -> None:
        """Different proportions should produce different dimensions."""
        cfg1 = URDFConfig(proportions={"torso_length": 1.0, "arm_length": 1.0})
        cfg2 = URDFConfig(proportions={"torso_length": 1.5, "arm_length": 1.0})
        xml1 = generate_urdf_xml(cfg1)
        xml2 = generate_urdf_xml(cfg2)
        if not (xml1 != xml2): raise ValueError(f"Assertion failed: { xml1 != xml2 }")


class TestValidateURDF:
    """Tests for validate_urdf_structure."""

    def test_valid_urdf_passes(self) -> None:
        config = URDFConfig()
        xml = generate_urdf_xml(config)
        is_valid, errors = validate_urdf_structure(xml)
        if not (is_valid): raise ValueError(f"Assertion failed: { is_valid }, f"Validation errors: {errors}"")

    def test_malformed_xml_fails(self) -> None:
        is_valid, errors = validate_urdf_structure("<robot><link name='x'>")
        if not (not is_valid): raise ValueError(f"Assertion failed: { not is_valid }")
        if not (len(errors) > 0): raise ValueError(f"Assertion failed: { len(errors) > 0 }")

    def test_wrong_root_element(self) -> None:
        is_valid, errors = validate_urdf_structure('<model name="x"/>')
        if not (not is_valid): raise ValueError(f"Assertion failed: { not is_valid }")
        if not (any("root element" in e.lower() for e in errors)): raise ValueError(f"Assertion failed: { any("root element" in e.lower() for e in errors) }")

    def test_duplicate_link_names(self) -> None:
        xml = """<?xml version="1.0"?>
<robot name="test">
  <link name="dup"/>
  <link name="dup"/>
</robot>"""
        is_valid, errors = validate_urdf_structure(xml)
        if not (not is_valid): raise ValueError(f"Assertion failed: { not is_valid }")
        if not (any("duplicate" in e.lower() for e in errors)): raise ValueError(f"Assertion failed: { any("duplicate" in e.lower() for e in errors) }")

    def test_unknown_joint_reference(self) -> None:
        xml = """<?xml version="1.0"?>
<robot name="test">
  <link name="a"/>
  <joint name="j1" type="fixed">
    <parent link="nonexistent"/>
    <child link="a"/>
  </joint>
</robot>"""
        is_valid, errors = validate_urdf_structure(xml)
        if not (not is_valid): raise ValueError(f"Assertion failed: { not is_valid }")
        if not (any("unknown" in e.lower() for e in errors)): raise ValueError(f"Assertion failed: { any("unknown" in e.lower() for e in errors) }")


# ═══════════════════════════════════════════════════════════════════════
# Preview Generator Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPreviewGenerator:
    """Tests for generate_preview_text."""

    def test_contains_robot_name(self) -> None:
        config = URDFConfig(robot_name="test_bot")
        text = generate_preview_text(config)
        if not ("test_bot" in text): raise ValueError(f"Assertion failed: { "test_bot" in text }")

    def test_contains_body_parameters(self) -> None:
        config = URDFConfig(height_m=1.80, mass_kg=80.0)
        text = generate_preview_text(config)
        if not ("1.80" in text): raise ValueError(f"Assertion failed: { "1.80" in text }")
        if not ("80.0" in text): raise ValueError(f"Assertion failed: { "80.0" in text }")

    def test_contains_template_name(self) -> None:
        config = URDFConfig(template="Upper Body Only")
        text = generate_preview_text(config)
        if not ("Upper Body Only" in text): raise ValueError(f"Assertion failed: { "Upper Body Only" in text }")

    def test_contains_segment_sizes(self) -> None:
        """Preview should list estimated segment sizes."""
        text = generate_preview_text(URDFConfig())
        if not ("Torso Height" in text): raise ValueError(f"Assertion failed: { "Torso Height" in text }")
        if not ("Thigh Length" in text): raise ValueError(f"Assertion failed: { "Thigh Length" in text }")

    def test_contains_template_segments(self) -> None:
        """Preview should list the segments in the template."""
        text = generate_preview_text(URDFConfig(template="Custom"))
        if not ("pelvis" in text): raise ValueError(f"Assertion failed: { "pelvis" in text }")
        if not ("torso" in text): raise ValueError(f"Assertion failed: { "torso" in text }")

    def test_contains_options(self) -> None:
        config = URDFConfig(damping=3.0, friction=1.5)
        text = generate_preview_text(config)
        if not ("3.00" in text): raise ValueError(f"Assertion failed: { "3.00" in text }")
        if not ("1.50" in text): raise ValueError(f"Assertion failed: { "1.50" in text }")

    def test_uses_shared_height_ratios(self) -> None:
        """Segment sizes should use HEIGHT_RATIOS, not hardcoded values."""
        config = URDFConfig(height_m=2.0)
        text = generate_preview_text(config)
        expected_torso = 2.0 * HEIGHT_RATIOS["torso"]
        if not (f"{expected_torso:.3f}" in text): raise ValueError(f"Assertion failed: { f"{expected_torso:.3f}" in text }")

    def test_empty_name_raises(self) -> None:
        with pytest.raises((PreconditionError, AssertionError)):
            generate_preview_text(URDFConfig(robot_name=""))

    def test_zero_height_raises(self) -> None:
        with pytest.raises((PreconditionError, AssertionError)):
            generate_preview_text(URDFConfig(height_m=0.0))


# ═══════════════════════════════════════════════════════════════════════
# DRY File Sync Verification
# ═══════════════════════════════════════════════════════════════════════


class TestFileSyncIntegrity:
    """Ensure root-level and python/ module copies stay in sync.

    The package has two copies of each module (root-level and python/
    urdf_builder_gui/) due to dual package discovery requirements.
    This test class catches drift.
    """

    _SYNCED_MODULES = [
        "contracts.py",
        "anthropometric_model.py",
        "urdf_generator.py",
        "preview_generator.py",
        "theme.py",
    ]

    def test_root_and_python_copies_identical(self) -> None:
        """Every root-level module must match its python/ copy."""
        from pathlib import Path

        root_dir = Path(__file__).resolve().parent.parent
        python_dir = root_dir / "python" / "urdf_builder_gui"

        mismatches: list[str] = []
        for mod_name in self._SYNCED_MODULES:
            root_file = root_dir / mod_name
            python_file = python_dir / mod_name

            if not root_file.exists():
                mismatches.append(f"{mod_name}: missing at root level")
                continue
            if not python_file.exists():
                mismatches.append(f"{mod_name}: missing in python/")
                continue

            root_content = root_file.read_text(encoding="utf-8")
            python_content = python_file.read_text(encoding="utf-8")
            if root_content != python_content:
                mismatches.append(
                    f"{mod_name}: root and python/ copies differ — "
                    "run: cp <root>/X python/urdf_builder_gui/X"
                )

        if not (not mismatches): raise ValueError(f"Assertion failed: { not mismatches }, (")
            "DRY sync violation! Module copies have drifted:\n"
            + "\n".join(f"  • {m}" for m in mismatches)
        )


# ═══════════════════════════════════════════════════════════════════════
# Integration / Round-Trip Tests
# ═══════════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end integration tests for the full generation pipeline."""

    def test_generate_then_validate_round_trip(self) -> None:
        """Generate → validate round-trip must always succeed."""
        for template in TEMPLATE_SEGMENTS:
            config = URDFConfig(
                robot_name="integration_test",
                height_m=1.80,
                mass_kg=80.0,
                template=template,
                damping=2.0,
                friction=1.0,
            )
            xml = generate_urdf_xml(config)
            is_valid, errors = validate_urdf_structure(xml)
            if not (is_valid): raise ValueError(f"Assertion failed: { is_valid }, f"{template}: {errors}"")

    def test_preview_and_generate_same_template(self) -> None:
        """Preview and generate should reference the same segments."""
        for template in TEMPLATE_SEGMENTS:
            config = URDFConfig(template=template)
            preview = generate_preview_text(config)
            xml = generate_urdf_xml(config)

            # Every segment name in the template should appear in preview
            for seg in get_template_segments(template):
                if not (seg in preview): raise ValueError(f"Assertion failed: { seg in preview }, f"{template}: {seg} not in preview"")

            # Generated XML should have links for each segment
            root = ET.fromstring(xml)
            link_names = {link.get("name") for link in root.findall("link")}
            for seg in get_template_segments(template):
                if not (seg in link_names): raise ValueError(f"Assertion failed: { seg in link_names }, f"{template}: {seg} not in XML"")

    def test_different_configs_produce_different_output(self) -> None:
        """Different heights must produce different URDF XML."""
        xml1 = generate_urdf_xml(URDFConfig(height_m=1.50))
        xml2 = generate_urdf_xml(URDFConfig(height_m=2.00))
        if not (xml1 != xml2): raise ValueError(f"Assertion failed: { xml1 != xml2 }")

    def test_all_inertia_values_positive(self) -> None:
        """Every inertia value in generated URDF must be positive."""
        xml = generate_urdf_xml(URDFConfig())
        root = ET.fromstring(xml)
        for inertia_el in root.iter("inertia"):
            for attr in ["ixx", "iyy", "izz"]:
                val = float(inertia_el.get(attr, "0"))
                if not (val > 0): raise ValueError(f"Assertion failed: { val > 0 }, f"Non-positive {attr}={val}"")
