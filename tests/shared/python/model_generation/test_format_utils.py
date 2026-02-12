"""Tests for model_generation.converters.format_utils.convert().

Covers issue #664: the former NotImplementedError in convert() has been
replaced with ValueError for unsupported conversion pairs. These tests
verify:
  - ValueError (not NotImplementedError) for unsupported pairs
  - Design-by-Contract preconditions on target_format
  - Preconditions on unknown source format
  - Helpful error messages
"""

from __future__ import annotations

import pytest
from model_generation.converters.format_utils import (
    ModelFormat,
    convert,
    detect_format,
)


class TestConvertUnsupportedPairs:
    """Unsupported conversion pairs must raise ValueError, not NotImplementedError."""

    def test_sdf_to_urdf_raises_valueerror(self) -> None:
        """SDF->URDF is not supported and must raise ValueError."""
        sdf_xml = "<sdf><world></world></sdf>"
        with pytest.raises(ValueError, match="not supported"):
            convert(sdf_xml, ModelFormat.URDF)

    def test_sdf_to_mjcf_raises_valueerror(self) -> None:
        """SDF->MJCF is not supported and must raise ValueError."""
        sdf_xml = "<sdf><world></world></sdf>"
        with pytest.raises(ValueError, match="not supported"):
            convert(sdf_xml, ModelFormat.MJCF)

    def test_urdf_to_sdf_raises_valueerror(self) -> None:
        """URDF->SDF is not supported and must raise ValueError."""
        urdf_xml = "<robot name='test'></robot>"
        with pytest.raises(ValueError, match="not supported"):
            convert(urdf_xml, ModelFormat.SDF)

    def test_mjcf_to_sdf_raises_valueerror(self) -> None:
        """MJCF->SDF is not supported and must raise ValueError."""
        mjcf_xml = "<mujoco></mujoco>"
        with pytest.raises(ValueError, match="not supported"):
            convert(mjcf_xml, ModelFormat.SDF)

    def test_unsupported_does_not_raise_notimplementederror(self) -> None:
        """Ensure NotImplementedError is never raised (issue #664)."""
        sdf_xml = "<sdf><world></world></sdf>"
        with pytest.raises(ValueError):
            convert(sdf_xml, ModelFormat.URDF)
        # If we got here, it means NotImplementedError was NOT raised

    def test_error_message_mentions_supported_pairs(self) -> None:
        """The error message should mention which conversions ARE available."""
        sdf_xml = "<sdf><world></world></sdf>"
        with pytest.raises(ValueError, match="URDF <-> MJCF"):
            convert(sdf_xml, ModelFormat.URDF)


class TestConvertPreconditions:
    """Design-by-Contract precondition tests for convert()."""

    def test_unknown_target_format_string_raises(self) -> None:
        """An invalid target format string must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown target format"):
            convert("<robot name='test'/>", "foobar")

    def test_unknown_target_format_enum_raises(self) -> None:
        """ModelFormat.UNKNOWN as target must raise ValueError."""
        with pytest.raises(ValueError, match="must not be ModelFormat.UNKNOWN"):
            convert("<robot name='test'/>", ModelFormat.UNKNOWN)

    def test_undetectable_source_raises(self) -> None:
        """An unrecognised source format must raise ValueError."""
        # Plain text that is not XML and not a file path with known extension
        with pytest.raises(ValueError, match="Could not detect the format"):
            convert("<some_random_tag/>", ModelFormat.URDF)

    def test_valid_target_format_string_accepted(self) -> None:
        """A valid lowercase string should be accepted without error."""
        # Same-format conversion returns the input unchanged
        urdf_xml = "<robot name='test'></robot>"
        result = convert(urdf_xml, "urdf")
        assert "<robot" in result


class TestConvertSameFormat:
    """When source and target are the same, convert() returns the input."""

    def test_same_format_xml_passthrough(self) -> None:
        """Same-format XML string should be returned as-is."""
        urdf_xml = "<robot name='test'></robot>"
        result = convert(urdf_xml, ModelFormat.URDF)
        assert result == urdf_xml

    def test_same_format_mjcf_passthrough(self) -> None:
        """Same-format MJCF XML string should be returned as-is."""
        mjcf_xml = "<mujoco></mujoco>"
        result = convert(mjcf_xml, ModelFormat.MJCF)
        assert result == mjcf_xml


class TestDetectFormat:
    """Tests for the detect_format helper used by convert()."""

    def test_urdf_xml_detected(self) -> None:
        assert detect_format("<robot name='r'/>") == ModelFormat.URDF

    def test_mjcf_xml_detected(self) -> None:
        assert detect_format("<mujoco/>") == ModelFormat.MJCF

    def test_sdf_xml_detected(self) -> None:
        assert detect_format("<sdf/>") == ModelFormat.SDF

    def test_unknown_xml_detected(self) -> None:
        assert detect_format("<something_else/>") == ModelFormat.UNKNOWN
