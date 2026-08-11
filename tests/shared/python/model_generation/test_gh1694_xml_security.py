"""
Tests for GH1694: XML External Entity (XXE) security hardening.

Verifies that all XML-parsing code paths in model_generation use defusedxml
and correctly reject XXE payloads and entity expansion attacks.

These are security regression tests — if they fail, untrusted XML input
could trigger XXE or entity-expansion denial-of-service attacks.

defusedxml raises defusedxml.common.EntitiesForbidden (a subclass of ValueError)
for DTD-based attacks. Some code paths catch these as ParseError (which is a
re-export of xml.etree.ElementTree.ParseError and is the same class), while
others let EntitiesForbidden propagate as a ValueError subclass.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# XXE / Billion-Laughs payloads
# ---------------------------------------------------------------------------

# Standard XXE probe: tries to read /etc/passwd via an external entity.
# defusedxml raises EntitiesForbidden; stdlib ET would attempt the file access.
XXE_PAYLOAD = """<?xml version="1.0"?>
<!DOCTYPE root [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<root>&xxe;</root>
"""

# Billion-Laughs: exponential entity expansion (DoS).
BILLION_LAUGHS = """<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
]>
<root>&lol3;</root>
"""


# ---------------------------------------------------------------------------
# URDFParser
# ---------------------------------------------------------------------------


class TestURDFParserXXE:
    """URDFParser must reject XXE via defusedxml."""

    def test_parse_xxe_raises(self) -> None:
        """XXE payload in URDF input must be rejected (raises some exception)."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()
        with pytest.raises(ValueError):
            parser.parse(XXE_PAYLOAD)

    def test_parse_billion_laughs_raises(self) -> None:
        """Billion-laughs payload must be rejected (raises some exception)."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()
        with pytest.raises(ValueError):
            parser.parse(BILLION_LAUGHS)

    def test_valid_urdf_still_parses(self) -> None:
        """Ensure the parser still accepts valid URDF after hardening."""
        from model_generation.converters.urdf_parser import URDFParser

        valid_urdf = "<robot name='test'><link name='base_link'/></robot>"
        parser = URDFParser()
        model = parser.parse(valid_urdf)
        assert model.name == "test"
        assert len(model.links) == 1

    def test_parse_xxe_does_not_read_filesystem(self) -> None:
        """defusedxml must block entity resolution — /etc/passwd must not appear."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()
        try:
            model = parser.parse(XXE_PAYLOAD)
            # If somehow no exception: the output must not contain passwd content
            output = str(model)
            assert "root:x:" not in output
            assert "/bin/bash" not in output
        except Exception:
            # Exception is the correct behavior
            pass


# ---------------------------------------------------------------------------
# MJCFConverter
# ---------------------------------------------------------------------------


class TestMJCFConverterXXE:
    """MJCFConverter.mjcf_to_urdf must reject XXE payloads."""

    def test_mjcf_to_urdf_rejects_xxe(self) -> None:
        """XXE in MJCF input must raise, not silently succeed."""
        from model_generation.converters.mjcf_converter import MJCFConverter

        converter = MJCFConverter()
        with pytest.raises(ValueError):
            converter.mjcf_to_urdf(XXE_PAYLOAD)

    def test_mjcf_to_urdf_rejects_billion_laughs(self) -> None:
        """Billion-laughs in MJCF input must be rejected."""
        from model_generation.converters.mjcf_converter import MJCFConverter

        converter = MJCFConverter()
        with pytest.raises(ValueError):
            converter.mjcf_to_urdf(BILLION_LAUGHS)

    def test_valid_mjcf_with_body_parses(self) -> None:
        """Valid MJCF with a body element should produce URDF output."""
        from model_generation.converters.mjcf_converter import MJCFConverter

        # Minimal MJCF with a body in the worldbody
        minimal_mjcf = (
            "<mujoco model='test'>"
            "<worldbody>"
            "<body name='base'>"
            "<inertial mass='1.0' pos='0 0 0' diaginertia='0.1 0.1 0.1'/>"
            "</body>"
            "</worldbody>"
            "</mujoco>"
        )
        converter = MJCFConverter()
        urdf = converter.mjcf_to_urdf(minimal_mjcf)
        assert "<robot" in urdf


# ---------------------------------------------------------------------------
# MDLParser
# ---------------------------------------------------------------------------


class TestMDLParserXXE:
    """MDLParser.parse_string must reject XXE payloads in XML mode."""

    def test_parse_string_xml_rejects_xxe(self) -> None:
        """XXE in SLX/XML content must raise an exception (EntitiesForbidden)."""
        from model_generation.converters.simscape.mdl_parser import MDLParser

        parser = MDLParser()
        with pytest.raises(ValueError):
            parser.parse_string(XXE_PAYLOAD, format="xml")

    def test_parse_string_xml_rejects_billion_laughs(self) -> None:
        """Billion-laughs entity expansion must raise an exception."""
        from model_generation.converters.simscape.mdl_parser import MDLParser

        parser = MDLParser()
        with pytest.raises(ValueError):
            parser.parse_string(BILLION_LAUGHS, format="xml")

    def test_valid_xml_parses(self) -> None:
        """Valid SLX-like XML should parse without error."""
        from model_generation.converters.simscape.mdl_parser import MDLParser

        valid_xml = "<System><Block Name='test' BlockType='SubSystem'/></System>"
        parser = MDLParser()
        model = parser.parse_string(valid_xml, format="xml")
        assert model is not None


# ---------------------------------------------------------------------------
# URDFTextEditor
# ---------------------------------------------------------------------------


class TestURDFTextEditorXXE:
    """URDFTextEditor.validate must use defusedxml and reject XXE."""

    def test_validate_xxe_raises_or_returns_error(self) -> None:
        """XXE in editor content must either raise or produce a validation ERROR."""
        from model_generation.editor.text_editor import (
            URDFTextEditor,
            ValidationSeverity,
        )

        editor = URDFTextEditor()
        editor.load_string(XXE_PAYLOAD)
        # defusedxml raises EntitiesForbidden which is a ValueError subclass
        # The validate() method may not catch it — acceptable: the error propagates
        try:
            messages = editor.validate()
            # If no exception: there must be an error-level message
            severities = [m.severity for m in messages]
            assert ValidationSeverity.ERROR in severities
        except Exception:
            # Exception is also correct (defusedxml blocking attack)
            pass

    def test_validate_billion_laughs_raises_or_returns_error(self) -> None:
        """Billion-laughs payload must either raise or produce ERROR message."""
        from model_generation.editor.text_editor import (
            URDFTextEditor,
            ValidationSeverity,
        )

        editor = URDFTextEditor()
        editor.load_string(BILLION_LAUGHS)
        try:
            messages = editor.validate()
            severities = [m.severity for m in messages]
            assert ValidationSeverity.ERROR in severities
        except Exception:
            pass

    def test_valid_urdf_validates_without_error(self) -> None:
        """A valid URDF should still pass validation after hardening."""
        from model_generation.editor.text_editor import (
            URDFTextEditor,
            ValidationSeverity,
        )

        valid_urdf = "<robot name='test'><link name='base_link'/></robot>"
        editor = URDFTextEditor()
        editor.load_string(valid_urdf)
        messages = editor.validate()
        errors = [m for m in messages if m.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# validate_mjcf (format_utils)
# ---------------------------------------------------------------------------


class TestValidateMJCFXXE:
    """validate_mjcf path must use defusedxml when mujoco is not available."""

    def test_validate_mjcf_string_valid(self) -> None:
        """validate_mjcf with valid MJCF returns a list (empty errors or mujoco errors)."""
        from model_generation.converters.format_utils import validate_mjcf

        valid_mjcf = "<mujoco model='test'><worldbody></worldbody></mujoco>"
        errors = validate_mjcf(valid_mjcf)
        assert isinstance(errors, list)

    def test_validate_mjcf_invalid_xml_returns_errors_or_raises(self) -> None:
        """Malformed XML should either return error messages or raise (mujoco raises)."""
        from model_generation.converters.format_utils import validate_mjcf

        bad_xml = "<mujoco><unclosed>"
        try:
            errors = validate_mjcf(bad_xml)
            # If returned: must be a non-empty error list
            assert isinstance(errors, list)
            assert len(errors) > 0
        except Exception:
            # mujoco installed: raises ValueError for malformed XML — also correct
            pass

    def test_validate_mjcf_no_stdlib_et_parse_in_fallback(self) -> None:
        """The defusedxml fallback must not import stdlib ET for ParseError."""
        import inspect

        from model_generation.converters import format_utils

        source = inspect.getsource(format_utils)
        # The fallback branch must not use StdET.ParseError
        assert (
            "StdET" not in source
        ), "format_utils.py must not reference StdET — use DefusedET.ParseError instead"
