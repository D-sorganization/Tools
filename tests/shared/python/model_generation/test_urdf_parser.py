"""Tests for the URDF parser.

Covers structural parsing (links, joints, materials, geometry, inertial),
the kinematic-tree query helpers on ParsedModel, the parse/round-trip
contract, and the mesh-filename security validation that guards against
path traversal and unsafe URI schemes.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from model_generation.converters.urdf_parser import ParsedModel, URDFParser
from model_generation.core.types import GeometryType, JointType

_ARM_URDF = """<?xml version="1.0"?>
<robot name="arm">
  <material name="red"><color rgba="1 0 0 1"/></material>
  <link name="base">
    <inertial>
      <origin xyz="0 0 0.1"/>
      <mass value="2.0"/>
      <inertia ixx="0.1" iyy="0.2" izz="0.3" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry><box size="0.1 0.2 0.3"/></geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry><cylinder radius="0.05" length="0.4"/></geometry>
    </collision>
  </link>
  <link name="forearm">
    <visual><geometry><sphere radius="0.02"/></geometry></visual>
  </link>
  <link name="hand">
    <visual>
      <geometry><mesh filename="meshes/hand.stl" scale="1 1 1"/></geometry>
    </visual>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="forearm"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="100" velocity="2"/>
    <dynamics damping="0.7" friction="0.1"/>
  </joint>
  <joint name="wrist" type="fixed">
    <parent link="forearm"/>
    <child link="hand"/>
  </joint>
</robot>
"""


@pytest.fixture
def model() -> ParsedModel:
    return URDFParser().parse(_ARM_URDF)


class TestParseStructure:
    def test_robot_name(self, model: ParsedModel) -> None:
        assert model.name == "arm"

    def test_links_and_joints_counts(self, model: ParsedModel) -> None:
        assert [link.name for link in model.links] == ["base", "forearm", "hand"]
        assert [j.name for j in model.joints] == ["shoulder", "wrist"]

    def test_no_warnings_on_valid_input(self, model: ParsedModel) -> None:
        assert model.warnings == []

    def test_material_parsed(self, model: ParsedModel) -> None:
        assert "red" in model.materials
        assert model.materials["red"].color == (1.0, 0.0, 0.0, 1.0)

    def test_inertial_parsed(self, model: ParsedModel) -> None:
        base = model.get_link("base")
        assert base is not None
        assert base.inertia.mass == pytest.approx(2.0)
        assert base.inertia.center_of_mass == (0.0, 0.0, 0.1)
        assert base.inertia.ixx == pytest.approx(0.1)
        assert base.inertia.izz == pytest.approx(0.3)

    def test_link_without_inertial_gets_default(self, model: ParsedModel) -> None:
        forearm = model.get_link("forearm")
        assert forearm is not None
        # Default inertial when none provided.
        assert forearm.inertia.mass == pytest.approx(1.0)

    def test_box_geometry(self, model: ParsedModel) -> None:
        base = model.get_link("base")
        assert base.visual_geometry.geometry_type == GeometryType.BOX
        assert base.visual_geometry.dimensions == (0.1, 0.2, 0.3)

    def test_cylinder_collision_geometry(self, model: ParsedModel) -> None:
        base = model.get_link("base")
        assert base.collision_geometry.geometry_type == GeometryType.CYLINDER
        assert base.collision_geometry.dimensions == (0.05, 0.4)

    def test_sphere_geometry(self, model: ParsedModel) -> None:
        forearm = model.get_link("forearm")
        assert forearm.visual_geometry.geometry_type == GeometryType.SPHERE
        assert forearm.visual_geometry.dimensions == (0.02,)

    def test_mesh_geometry(self, model: ParsedModel) -> None:
        hand = model.get_link("hand")
        assert hand.visual_geometry.geometry_type == GeometryType.MESH
        # File does not exist on disk -> resolution leaves the original name.
        assert "hand.stl" in hand.visual_geometry.mesh_filename


class TestParseJoints:
    def test_revolute_joint(self, model: ParsedModel) -> None:
        j = model.get_joint("shoulder")
        assert j is not None
        assert j.joint_type == JointType.REVOLUTE
        assert j.parent == "base"
        assert j.child == "forearm"
        assert j.axis == (0.0, 1.0, 0.0)

    def test_joint_limits(self, model: ParsedModel) -> None:
        j = model.get_joint("shoulder")
        assert j.limits is not None
        assert j.limits.lower == pytest.approx(-1.5)
        assert j.limits.upper == pytest.approx(1.5)
        assert j.limits.effort == pytest.approx(100.0)

    def test_joint_dynamics(self, model: ParsedModel) -> None:
        j = model.get_joint("shoulder")
        assert j.dynamics.damping == pytest.approx(0.7)
        assert j.dynamics.friction == pytest.approx(0.1)

    def test_fixed_joint_has_default_axis_and_no_limits(
        self, model: ParsedModel
    ) -> None:
        j = model.get_joint("wrist")
        assert j.joint_type == JointType.FIXED
        assert j.axis == (0.0, 0.0, 1.0)
        assert j.limits is None

    def test_unknown_joint_type_falls_back_to_fixed(self) -> None:
        urdf = """<robot name="r">
          <link name="a"/><link name="b"/>
          <joint name="j" type="warp-drive">
            <parent link="a"/><child link="b"/>
          </joint>
        </robot>"""
        m = URDFParser().parse(urdf)
        assert m.get_joint("j").joint_type == JointType.FIXED

    def test_joint_missing_parent_recorded_as_warning(self) -> None:
        urdf = """<robot name="r">
          <link name="a"/><link name="b"/>
          <joint name="j" type="fixed"><child link="b"/></joint>
        </robot>"""
        m = URDFParser().parse(urdf)
        assert any("parent or child" in w for w in m.warnings)
        assert m.joints == []


class TestParsedModelQueries:
    def test_get_root_link(self, model: ParsedModel) -> None:
        # base is never a child -> it is the root.
        assert model.get_root_link().name == "base"

    def test_get_children(self, model: ParsedModel) -> None:
        assert model.get_children("base") == ["forearm"]
        assert model.get_children("forearm") == ["hand"]
        assert model.get_children("hand") == []

    def test_get_parent(self, model: ParsedModel) -> None:
        assert model.get_parent("forearm") == "base"
        assert model.get_parent("base") is None

    def test_get_subtree(self, model: ParsedModel) -> None:
        assert model.get_subtree("base") == ["base", "forearm", "hand"]
        assert model.get_subtree("forearm") == ["forearm", "hand"]

    def test_get_link_missing_returns_none(self, model: ParsedModel) -> None:
        assert model.get_link("ghost") is None

    def test_get_joint_missing_returns_none(self, model: ParsedModel) -> None:
        assert model.get_joint("ghost") is None

    def test_query_helpers_reject_none(self, model: ParsedModel) -> None:
        with pytest.raises(ValueError):
            model.get_link(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            model.get_parent(None)  # type: ignore[arg-type]

    def test_empty_model_root_is_none(self) -> None:
        empty = ParsedModel(name="empty")
        assert empty.get_root_link() is None

    def test_copy_is_independent_and_editable(self, model: ParsedModel) -> None:
        clone = model.copy()
        assert clone.read_only is False
        assert [link.name for link in clone.links] == [
            link.name for link in model.links
        ]
        # Mutating the clone's link list does not affect the original.
        clone.links.pop()
        assert len(clone.links) == len(model.links) - 1


class TestParseRoundTrip:
    def test_to_urdf_then_reparse(self, model: ParsedModel) -> None:
        xml = model.to_urdf()
        assert "<robot" in xml
        reparsed = URDFParser().parse(xml)
        assert reparsed.name == model.name
        assert [link.name for link in reparsed.links] == [
            link.name for link in model.links
        ]
        assert [j.name for j in reparsed.joints] == [j.name for j in model.joints]

    def test_parse_string_alias(self) -> None:
        m = URDFParser().parse_string(_ARM_URDF, read_only=True)
        assert m.read_only is True
        assert m.name == "arm"


class TestParseErrors:
    def test_invalid_xml_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid URDF XML"):
            URDFParser().parse("<robot><link></robot>")

    def test_wrong_root_element_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected 'robot'"):
            URDFParser().parse("<mujoco></mujoco>")

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            URDFParser().parse(Path("/nonexistent/robot.urdf"))

    def test_file_source_parsed(self, tmp_path: Path) -> None:
        f = tmp_path / "robot.urdf"
        f.write_text(_ARM_URDF)
        m = URDFParser().parse(f)
        assert m.name == "arm"
        assert m.source_path == f

    def test_unnamed_robot_defaults(self) -> None:
        m = URDFParser().parse("<robot><link name='a'/></robot>")
        assert m.name == "unnamed_robot"

    def test_link_missing_name_recorded_as_warning(self) -> None:
        m = URDFParser().parse("<robot name='r'><link/></robot>")
        assert any("name" in w.lower() for w in m.warnings)


class TestMeshFilenameValidation:
    @pytest.mark.parametrize(
        "bad",
        [
            "../etc/passwd",
            "/absolute/mesh.stl",
            "http://evil.example/mesh.stl",
            "package://../escape.stl",
            "C:/windows/mesh.stl",
            "sub/../../escape.stl",
            "",
            "   ",
        ],
    )
    def test_unsafe_filenames_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError):
            URDFParser._validate_mesh_filename(bad)

    def test_relative_filename_normalized(self) -> None:
        assert URDFParser._validate_mesh_filename("meshes/arm.stl") == "meshes/arm.stl"

    def test_backslashes_normalized_to_posix(self) -> None:
        assert URDFParser._validate_mesh_filename("meshes\\arm.stl") == "meshes/arm.stl"

    def test_package_uri_preserved(self) -> None:
        assert (
            URDFParser._validate_mesh_filename("package://robot/arm.stl")
            == "package://robot/arm.stl"
        )

    def test_none_filename_raises(self) -> None:
        with pytest.raises(ValueError):
            URDFParser._validate_mesh_filename(None)  # type: ignore[arg-type]

    def test_windows_drive_prefix_detection(self) -> None:
        assert URDFParser._has_windows_drive_prefix("C:/x") is True
        assert URDFParser._has_windows_drive_prefix("meshes/x") is False


class TestMeshResolution:
    def test_relative_mesh_resolved_when_present(self, tmp_path: Path) -> None:
        # Lay out: tmp/model.urdf referencing meshes/part.stl on disk.
        mesh_dir = tmp_path / "meshes"
        mesh_dir.mkdir()
        (mesh_dir / "part.stl").write_text("solid x endsolid x")
        urdf = tmp_path / "model.urdf"
        urdf.write_text(
            '<robot name="r"><link name="a"><visual><geometry>'
            '<mesh filename="meshes/part.stl"/></geometry></visual></link></robot>'
        )
        m = URDFParser(resolve_meshes=True).parse(urdf)
        resolved = m.get_link("a").visual_geometry.mesh_filename
        assert resolved.endswith("part.stl")
        assert Path(resolved).exists()

    def test_resolution_disabled_keeps_original(self, tmp_path: Path) -> None:
        urdf = tmp_path / "model.urdf"
        urdf.write_text(
            '<robot name="r"><link name="a"><visual><geometry>'
            '<mesh filename="meshes/part.stl"/></geometry></visual></link></robot>'
        )
        m = URDFParser(resolve_meshes=False).parse(urdf)
        assert m.get_link("a").visual_geometry.mesh_filename == "meshes/part.stl"


def test_default_joint_limits_use_pi_bounds() -> None:
    # A <limit/> with no attributes falls back to +/- pi default bounds.
    urdf = """<robot name="r">
      <link name="a"/><link name="b"/>
      <joint name="j" type="revolute">
        <parent link="a"/><child link="b"/>
        <limit/>
      </joint>
    </robot>"""
    m = URDFParser().parse(urdf)
    limits = m.get_joint("j").limits
    assert limits.lower == pytest.approx(-math.pi)
    assert limits.upper == pytest.approx(math.pi)
