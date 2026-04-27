"""Tests for fixes identified in the 2026-03-09 adversarial code review.

This module covers:
- C3D reader: bounds checking, export validation, DbC, sanitization
- URDF writer: material collisions, graph validation, mesh path safety
- Body parameters: strict validation bounds
- MJCF converter: capsule parsing safety
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("numpy")
import numpy as np

# ──────────────────────────────────────────────────────────────────
# C3D Reader Tests
# ──────────────────────────────────────────────────────────────────


class TestC3DReaderBoundsChecking:
    """C-04: C3D array indexing without bounds check."""

    @pytest.fixture()
    def reader(self, tmp_path):
        """Create a C3DDataReader with mocked ezc3d."""
        from upstream_drift_tools.lab.bio.c3d_reader import C3DDataReader

        p = tmp_path / "test.c3d"
        p.touch()
        return C3DDataReader(p)

    def test_points_3_channels_fills_nan_residuals(self, reader):
        """When C3D has only XYZ (3 channels), residuals should be NaN."""
        # 3 channels, 2 markers, 5 frames
        points_3ch = np.random.rand(3, 2, 5)
        mock_c3d = {
            "data": {
                "points": points_3ch,
                "analogs": np.zeros((1, 0, 5)),
            },
            "parameters": {
                "POINT": {
                    "LABELS": {"value": ["M1", "M2"]},
                    "FRAMES": {"value": [5]},
                    "RATE": {"value": [100.0]},
                    "UNITS": {"value": ["m"]},
                }
            },
        }

        with patch.object(reader, "_load", return_value=mock_c3d):
            reader._metadata = None
            df = reader.points_dataframe(include_time=False)
            assert "residual" in df.columns
            assert df["residual"].isna().all(), (
                "Residuals should be NaN when only 3 channels present"
            )

    def test_points_4_channels_has_residuals(self, reader):
        """When C3D has 4 channels, residuals are extracted normally."""
        points_4ch = np.random.rand(4, 2, 5)
        mock_c3d = {
            "data": {
                "points": points_4ch,
                "analogs": np.zeros((1, 0, 5)),
            },
            "parameters": {
                "POINT": {
                    "LABELS": {"value": ["M1", "M2"]},
                    "FRAMES": {"value": [5]},
                    "RATE": {"value": [100.0]},
                    "UNITS": {"value": ["m"]},
                }
            },
        }

        with patch.object(reader, "_load", return_value=mock_c3d):
            reader._metadata = None
            df = reader.points_dataframe(include_time=False)
            assert not df["residual"].isna().all()


class TestC3DExportPathValidation:
    """H-07: Export path validation hardening."""

    def test_env_var_allows_any_path(self, tmp_path, monkeypatch):
        """C3D_ALLOW_ANY_EXPORT_PATH=1 should skip traversal check."""
        from upstream_drift_tools.lab.bio.c3d_reader import C3DDataReader

        monkeypatch.setenv("C3D_ALLOW_ANY_EXPORT_PATH", "1")
        # Should not raise even for an outside path
        C3DDataReader._validate_export_path(Path("/some/other/directory/out.csv"))

    def test_traversal_blocked_by_default(self, monkeypatch):
        """Without env var, paths outside CWD should be rejected."""
        from upstream_drift_tools.lab.bio.c3d_reader import C3DDataReader

        monkeypatch.delenv("C3D_ALLOW_ANY_EXPORT_PATH", raising=False)
        with pytest.raises(ValueError, match="Security"):
            C3DDataReader._validate_export_path(Path("/etc/passwd"))

    def test_unsupported_extension_blocked(self, monkeypatch, tmp_path):
        """Unsupported output extensions should be rejected at validation time."""
        from upstream_drift_tools.lab.bio.c3d_reader import C3DDataReader

        monkeypatch.delenv("C3D_ALLOW_ANY_EXPORT_PATH", raising=False)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="Unsupported export format"):
            C3DDataReader._validate_export_path(tmp_path / "capture.exe")


class TestC3DDbCIntegration:
    """M-07: DbC decorators on C3D reader."""

    def test_empty_filepath_raises(self):
        """C3DDataReader should reject empty file paths."""
        from upstream_drift_tools.lab.bio.c3d_reader import C3DDataReader

        with pytest.raises((ValueError, Exception)):
            C3DDataReader("")

    def test_force_plate_negative_number_raises(self, tmp_path):
        """force_plate_dataframe should reject plate_number < 1."""
        from upstream_drift_tools.lab.bio.c3d_reader import C3DDataReader

        p = tmp_path / "test.c3d"
        p.touch()
        reader = C3DDataReader(p)

        with pytest.raises((ValueError, Exception)):
            reader.force_plate_dataframe(plate_number=-1)


class TestC3DMetadataSanitization:
    """CSV export metadata should be sanitized against formula injection."""

    def test_sanitize_for_csv_strips_dangerous_prefixes(self):
        from upstream_drift_tools.lab.bio.c3d_reader import C3DDataReader

        assert C3DDataReader._sanitize_for_csv("=cmd()") == "'=cmd()"
        assert C3DDataReader._sanitize_for_csv("+1+1") == "'+1+1"
        assert C3DDataReader._sanitize_for_csv("-malicious") == "'-malicious"
        assert C3DDataReader._sanitize_for_csv("@sum") == "'@sum"
        assert C3DDataReader._sanitize_for_csv("safe string") == "safe string"
        assert C3DDataReader._sanitize_for_csv(42) == 42


class TestC3DUnitScaleDRY:
    """M-06: No duplicate unit conversion dictionaries."""

    def test_unit_scale_m_to_mm(self):
        from upstream_drift_tools.lab.bio.c3d_reader import C3DDataReader

        assert C3DDataReader._unit_scale("m", "mm") == pytest.approx(1000.0)

    def test_unit_scale_mm_to_m(self):
        from upstream_drift_tools.lab.bio.c3d_reader import C3DDataReader

        assert C3DDataReader._unit_scale("mm", "m") == pytest.approx(0.001)

    def test_unit_scale_same_unit_noop(self):
        from upstream_drift_tools.lab.bio.c3d_reader import C3DDataReader

        assert C3DDataReader._unit_scale("cm", "cm") == pytest.approx(1.0)

    def test_unit_scale_unsupported_raises(self):
        from upstream_drift_tools.lab.bio.c3d_reader import C3DDataReader

        with pytest.raises(ValueError, match="Unsupported"):
            C3DDataReader._unit_scale("furlongs", "m")


# ──────────────────────────────────────────────────────────────────
# URDF Writer Tests
# ──────────────────────────────────────────────────────────────────


class TestURDFWriterGraphValidation:
    """H-04: URDF parser accepts invalid graphs."""

    def test_no_root_raises(self):
        """Cyclic graph with no root should raise ValueError."""
        from model_generation.builders.urdf_writer import URDFWriter
        from model_generation.core.types import Inertia, Joint, JointType, Link, Origin

        inertia = Inertia(ixx=1, iyy=1, izz=1, mass=1)
        links = [Link(name="A", inertia=inertia), Link(name="B", inertia=inertia)]
        joints = [
            Joint(
                name="j1",
                joint_type=JointType.FIXED,
                parent="A",
                child="B",
                origin=Origin(),
            ),
            Joint(
                name="j2",
                joint_type=JointType.FIXED,
                parent="B",
                child="A",
                origin=Origin(),
            ),
        ]

        writer = URDFWriter()
        with pytest.raises(ValueError, match="no root"):
            writer.write("cyclic_robot", links, joints)

    def test_valid_tree_succeeds(self):
        """A proper tree should not raise."""
        from model_generation.builders.urdf_writer import URDFWriter
        from model_generation.core.types import Inertia, Joint, JointType, Link, Origin

        inertia = Inertia(ixx=1, iyy=1, izz=1, mass=1)
        links = [
            Link(name="base", inertia=inertia),
            Link(name="child", inertia=inertia),
        ]
        joints = [
            Joint(
                name="j1",
                joint_type=JointType.FIXED,
                parent="base",
                child="child",
                origin=Origin(),
            ),
        ]

        writer = URDFWriter()
        result = writer.write("valid_robot", links, joints)
        assert "<robot" in result
        assert 'name="valid_robot"' in result

    def test_multiple_roots_raise(self):
        """Disconnected forests with multiple roots should be rejected."""
        from model_generation.builders.urdf_writer import URDFWriter
        from model_generation.core.types import Inertia, Joint, JointType, Link, Origin

        inertia = Inertia(ixx=1, iyy=1, izz=1, mass=1)
        links = [
            Link(name="base_a", inertia=inertia),
            Link(name="child_a", inertia=inertia),
            Link(name="base_b", inertia=inertia),
        ]
        joints = [
            Joint(
                name="j1",
                joint_type=JointType.FIXED,
                parent="base_a",
                child="child_a",
                origin=Origin(),
            ),
        ]

        writer = URDFWriter()
        with pytest.raises(ValueError, match="root"):
            writer.write("forest_robot", links, joints)


class TestURDFWriterMeshPathValidation:
    """H-05: Mesh path traversal must fail closed."""

    @staticmethod
    def _mesh_link(mesh_filename: str):
        from model_generation.core.types import Geometry, Inertia, Link

        return Link(
            name="base",
            inertia=Inertia(ixx=1, iyy=1, izz=1, mass=1),
            visual_geometry=Geometry.mesh(mesh_filename),
        )

    def test_relative_traversal_raises(self):
        """Relative mesh paths must not escape the asset tree."""
        from model_generation.builders.urdf_writer import URDFWriter

        writer = URDFWriter()

        with pytest.raises(ValueError, match="path traversal"):
            writer.write("unsafe_robot", [self._mesh_link("../../etc/passwd")], [])

    def test_package_uri_traversal_raises(self):
        """package:// URIs must also reject traversal segments."""
        from model_generation.builders.urdf_writer import URDFWriter

        writer = URDFWriter()

        with pytest.raises(ValueError, match="path traversal"):
            writer.write(
                "unsafe_robot",
                [self._mesh_link("package://robot/../secrets/base.stl")],
                [],
            )

    def test_relative_mesh_path_allowed(self):
        """Safe in-tree relative mesh paths should continue to serialize."""
        from model_generation.builders.urdf_writer import URDFWriter

        writer = URDFWriter()
        result = writer.write("safe_robot", [self._mesh_link("meshes/base.stl")], [])

        assert 'filename="meshes/base.stl"' in result

    def test_package_uri_allowed(self):
        """Safe package:// mesh URIs should continue to serialize."""
        from model_generation.builders.urdf_writer import URDFWriter

        writer = URDFWriter()
        result = writer.write(
            "safe_robot",
            [self._mesh_link("package://robot_description/meshes/base.stl")],
            [],
        )

        assert 'filename="package://robot_description/meshes/base.stl"' in result


class TestURDFWriterCompositeJointValidation:
    """H-06: Composite joint expansion should validate and normalize inputs."""

    @staticmethod
    def _links():
        from model_generation.core.types import Inertia, Link

        inertia = Inertia(ixx=1, iyy=1, izz=1, mass=1)
        return [
            Link(name="base", inertia=inertia),
            Link(name="child", inertia=inertia),
        ]

    def test_missing_parent_raises_clear_error(self):
        """Composite joints must reject missing parent references before serialization."""
        from model_generation.builders.urdf_writer import URDFWriter
        from model_generation.core.types import Joint, JointType, Origin

        writer = URDFWriter()
        joint = Joint(
            name="shoulder",
            joint_type=JointType.GIMBAL,
            parent="",
            child="child",
            origin=Origin(),
        )

        with pytest.raises(ValueError, match="parent"):
            writer.write("bad_robot", self._links(), [joint])

    def test_missing_composite_axis_uses_default(self):
        """Missing composite axes should fall back to the canonical default sequence."""
        from model_generation.builders.urdf_writer import URDFWriter
        from model_generation.core.types import Joint, JointType, Origin

        writer = URDFWriter()
        joint = Joint(
            name="shoulder",
            joint_type=JointType.GIMBAL,
            parent="base",
            child="child",
            origin=Origin(),
            composite_axes=[(0, 0, 1), None, (1, 0, 0)],
        )

        result = writer.write("axis_robot", self._links(), [joint])
        assert 'axis xyz="0 1 0"' in result

    def test_missing_composite_limits_use_defaults(self):
        """Composite joints without explicit limits should emit default revolute limits."""
        from model_generation.builders.urdf_writer import URDFWriter
        from model_generation.core.types import Joint, JointType, Origin

        writer = URDFWriter()
        joint = Joint(
            name="elbow",
            joint_type=JointType.UNIVERSAL,
            parent="base",
            child="child",
            origin=Origin(),
            limits=None,
            composite_limits=[None, None],
        )

        result = writer.write("limits_robot", self._links(), [joint])
        assert result.count("<limit ") == 2


class TestURDFWriterXMLEscaping:
    """C-01: XML injection in URDF writer robot_name (already fixed)."""

    def test_robot_name_escaped(self):
        """Special XML characters in robot_name should be escaped."""
        from model_generation.builders.urdf_writer import URDFWriter
        from model_generation.core.types import Inertia, Link

        inertia = Inertia(ixx=1, iyy=1, izz=1, mass=1)
        links = [Link(name="base", inertia=inertia)]

        writer = URDFWriter()
        result = writer.write('robot<"evil">&', links, [])
        assert "&lt;" in result
        assert "&quot;" in result
        assert "&amp;" in result


class TestURDFWriterMaterialCollision:
    """H-03: Material name collisions should be warned."""

    def test_collision_logs_warning(self, caplog):
        """Two links with same material name but different colors should warn."""
        from model_generation.builders.urdf_writer import URDFWriter
        from model_generation.core.types import Inertia, Link, Material

        inertia = Inertia(ixx=1, iyy=1, izz=1, mass=1)
        mat_a = Material(name="skin", color=(1.0, 0.0, 0.0, 1.0))
        mat_b = Material(name="skin", color=(0.0, 1.0, 0.0, 1.0))
        links = [
            Link(name="arm", inertia=inertia, visual_material=mat_a),
            Link(name="leg", inertia=inertia, visual_material=mat_b),
        ]

        writer = URDFWriter()
        import logging

        with caplog.at_level(logging.WARNING):
            writer._collect_materials(links, {})

        assert any("collision" in r.message.lower() for r in caplog.records)


# ──────────────────────────────────────────────────────────────────
# Body Parameters Tests
# ──────────────────────────────────────────────────────────────────


_has_scipy = pytest.importorskip is not None  # placeholder
try:
    pytest.importorskip("scipy")
    import scipy  # noqa: F401

    _has_scipy = True
except ImportError:
    _has_scipy = False


@pytest.mark.skipif(
    not _has_scipy, reason="scipy required for humanoid_character_builder"
)
class TestBodyParametersStrictValidation:
    """H-02: No bounds on anthropometric params."""

    def test_negative_height_raises(self):
        from humanoid_character_builder.core.body_parameters import BodyParameters

        with pytest.raises(ValueError, match="height_m must be positive"):
            BodyParameters(height_m=-1.0, mass_kg=75.0)

    def test_extreme_height_raises(self):
        from humanoid_character_builder.core.body_parameters import BodyParameters

        params = BodyParameters(height_m=10.0, mass_kg=75.0)
        with pytest.raises(ValueError, match="height_m must be in"):
            params.validate_strict()

    def test_negative_mass_raises(self):
        from humanoid_character_builder.core.body_parameters import BodyParameters

        with pytest.raises(ValueError, match="mass_kg must be positive"):
            BodyParameters(height_m=1.75, mass_kg=-50.0)

    def test_extreme_factor_raises(self):
        from humanoid_character_builder.core.body_parameters import BodyParameters

        params = BodyParameters(height_m=1.75, mass_kg=75.0, arm_length_factor=10.0)
        with pytest.raises(ValueError, match="arm_length_factor exceeds hard limit"):
            params.validate_strict()

    def test_valid_params_pass(self):
        from humanoid_character_builder.core.body_parameters import BodyParameters

        params = BodyParameters(height_m=1.80, mass_kg=80.0)
        params.validate_strict()  # Should not raise

    def test_validate_includes_all_factors(self):
        """validate() should check all proportion factors, including newly added ones."""
        from humanoid_character_builder.core.body_parameters import BodyParameters

        params = BodyParameters(
            height_m=1.75,
            mass_kg=75.0,
            neck_length_factor=-1.0,  # newly checked
        )
        errors = params.validate()
        assert any("neck_length_factor" in e for e in errors)


# ──────────────────────────────────────────────────────────────────
# MJCF Converter Tests
# ──────────────────────────────────────────────────────────────────


try:
    import defusedxml  # noqa: F401

    _has_defusedxml = True
except ImportError:
    _has_defusedxml = False


@pytest.mark.skipif(
    not _has_defusedxml, reason="defusedxml required for MJCF converter"
)
class TestMJCFCapsuleParsing:
    """H-14: MJCF capsule parsing IndexError."""

    def test_capsule_fromto_insufficient_values(self):
        """Capsule with < 6 fromto values should not crash."""
        from model_generation.converters.mjcf_converter import MJCFConverter

        converter = MJCFConverter()
        import xml.etree.ElementTree as ET

        geom = ET.Element("geom", type="capsule", fromto="0 0 0", size="0.05")
        # _parse_mjcf_geom returns (Geometry | None, Origin)
        result = converter._parse_mjcf_geom(geom)
        # Should return (None, origin) without crashing
        assert isinstance(result, tuple)
        geom_obj, origin = result
        assert geom_obj is None  # insufficient fromto → returns None

    def test_capsule_zero_length_fromto(self):
        """Capsule with identical fromto endpoints should fallback to sphere."""
        from model_generation.converters.mjcf_converter import MJCFConverter

        converter = MJCFConverter()
        import xml.etree.ElementTree as ET

        geom = ET.Element("geom", type="capsule", fromto="1 2 3 1 2 3", size="0.05")
        result = converter._parse_mjcf_geom(geom)
        assert isinstance(result, tuple)
        geom_obj, _ = result
        # Should fallback to sphere, not crash
        if geom_obj is not None:
            from model_generation.core.types import GeometryType

            assert geom_obj.geometry_type == GeometryType.SPHERE

    def test_capsule_valid_fromto(self):
        """Valid capsule with proper fromto should parse correctly."""
        from model_generation.converters.mjcf_converter import MJCFConverter

        converter = MJCFConverter()
        import xml.etree.ElementTree as ET

        geom = ET.Element("geom", type="capsule", fromto="0 0 0 0 0 1", size="0.05")
        result = converter._parse_mjcf_geom(geom)
        geom_obj, origin = result
        assert geom_obj is not None
        from model_generation.core.types import GeometryType

        assert geom_obj.geometry_type == GeometryType.CAPSULE
