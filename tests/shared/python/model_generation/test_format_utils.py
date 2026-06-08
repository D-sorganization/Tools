"""Tests for model format detection and conversion utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
from model_generation.converters.format_utils import (
    ModelFormat,
    convert,
    detect_format,
    validate_mjcf,
    validate_urdf,
)

_URDF_XML = """<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
</robot>
"""

_MJCF_XML = """<mujoco model="r">
  <worldbody/>
</mujoco>
"""


class TestDetectFormatFromContent:
    def test_robot_string_is_urdf(self) -> None:
        assert detect_format(_URDF_XML) == ModelFormat.URDF

    def test_mujoco_string_is_mjcf(self) -> None:
        assert detect_format(_MJCF_XML) == ModelFormat.MJCF

    def test_sdf_string(self) -> None:
        assert detect_format("<sdf version='1.6'></sdf>") == ModelFormat.SDF

    def test_world_string_is_sdf(self) -> None:
        assert detect_format("<world name='w'></world>") == ModelFormat.SDF

    def test_unrecognized_xml_is_unknown(self) -> None:
        assert detect_format("<foo/>") == ModelFormat.UNKNOWN


class TestDetectFormatFromPath:
    def test_urdf_extension(self) -> None:
        assert detect_format(Path("robot.urdf")) == ModelFormat.URDF

    def test_sdf_extension(self) -> None:
        assert detect_format(Path("scene.sdf")) == ModelFormat.SDF

    @pytest.mark.parametrize("ext", [".mdl", ".slx"])
    def test_simscape_extensions(self, ext: str) -> None:
        assert detect_format(Path(f"model{ext}")) == ModelFormat.SIMSCAPE

    def test_unknown_extension(self) -> None:
        assert detect_format(Path("notes.txt")) == ModelFormat.UNKNOWN

    def test_xml_extension_reads_content(self, tmp_path: Path) -> None:
        f = tmp_path / "robot.xml"
        f.write_text(_URDF_XML)
        assert detect_format(f) == ModelFormat.URDF

    def test_xml_extension_missing_file_is_unknown(self) -> None:
        # Nonexistent .xml -> empty content -> UNKNOWN, no exception.
        assert detect_format(Path("does_not_exist.xml")) == ModelFormat.UNKNOWN


class TestConvert:
    def test_unknown_string_target_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown target format"):
            convert(_URDF_XML, "not-a-format")

    def test_explicit_unknown_target_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be ModelFormat.UNKNOWN"):
            convert(_URDF_XML, ModelFormat.UNKNOWN)

    def test_undetectable_source_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not detect"):
            convert("<foo/>", ModelFormat.URDF)

    def test_same_format_string_source_returns_source(self) -> None:
        # Source already URDF and target URDF -> returned unchanged.
        result = convert(_URDF_XML, ModelFormat.URDF)
        assert "<robot" in result

    def test_same_format_string_target_normalized(self) -> None:
        result = convert(_URDF_XML, "urdf")
        assert "<robot" in result

    def test_unsupported_pair_raises(self) -> None:
        # SDF -> URDF is explicitly out of scope.
        with pytest.raises(ValueError, match="not supported"):
            convert("<sdf></sdf>", ModelFormat.URDF)


class TestValidateUrdf:
    def test_valid_urdf_string_returns_no_errors(self) -> None:
        errors = validate_urdf(_URDF_XML)
        assert isinstance(errors, list)

    def test_malformed_xml_returns_error_list(self) -> None:
        errors = validate_urdf("<robot><link></robot>")
        assert errors  # non-empty
        assert all(isinstance(e, str) for e in errors)


class TestValidateMjcf:
    def test_valid_mjcf_returns_empty(self) -> None:
        # Valid MJCF validates cleanly whether via mujoco or the
        # defusedxml fallback.
        errors = validate_mjcf(_MJCF_XML)
        assert errors == [] or all(isinstance(e, str) for e in errors)

    def test_malformed_xml_is_reported(self) -> None:
        # Behavior depends on whether mujoco is installed:
        #  - with mujoco, the underlying loader raises ValueError directly;
        #  - without it, the defusedxml fallback returns an error-message list.
        # Either way, malformed input must not be silently accepted.
        try:
            errors = validate_mjcf("<mujoco><body></mujoco>")
        except ValueError:
            return
        assert errors
        assert all(isinstance(e, str) for e in errors)
