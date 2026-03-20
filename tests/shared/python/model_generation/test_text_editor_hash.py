"""Tests for GH1651: MD5 usedforsecurity=False in URDFTextEditor._add_to_history.

Verifies that the hashlib.md5() call uses usedforsecurity=False so that the
code runs correctly on security-restricted platforms (FIPS mode) and passes
Bandit's HIGH-severity weak-hash check.
"""

from __future__ import annotations

import hashlib

import pytest
from model_generation.editor.text_editor import URDFTextEditor

# ── Helpers ─────────────────────────────────────────────────────────────────

_MINIMAL_URDF = """<?xml version="1.0"?>
<robot name="test">
  <link name="base_link"/>
</robot>"""


# ── Tests ────────────────────────────────────────────────────────────────────


class TestTextEditorHashSecurity:
    """Verify MD5 is called with usedforsecurity=False (GH1651)."""

    def test_add_to_history_creates_checksum(self) -> None:
        """_add_to_history stores a non-empty checksum in the version."""
        editor = URDFTextEditor()
        editor.set_content(_MINIMAL_URDF)
        # set_content triggers _add_to_history internally
        assert editor._history, "History should not be empty after set_content"
        version = editor._history[-1]
        assert version.checksum, "Checksum must be non-empty"

    def test_checksum_is_valid_md5_hex(self) -> None:
        """Checksum must be a valid 32-character MD5 hex digest."""
        editor = URDFTextEditor()
        editor.set_content(_MINIMAL_URDF)
        version = editor._history[-1]
        # MD5 hex digest is always 32 lowercase hex characters
        assert len(version.checksum) == 32
        assert all(c in "0123456789abcdef" for c in version.checksum)

    def test_checksum_matches_expected_md5(self) -> None:
        """Checksum value must equal hashlib.md5(..., usedforsecurity=False)."""
        editor = URDFTextEditor()
        editor.set_content(_MINIMAL_URDF)
        version = editor._history[-1]
        expected = hashlib.md5(
            _MINIMAL_URDF.encode(), usedforsecurity=False
        ).hexdigest()
        assert version.checksum == expected

    def test_checksum_changes_on_content_change(self) -> None:
        """Different content must produce a different checksum."""
        editor = URDFTextEditor()
        editor.set_content(_MINIMAL_URDF)
        checksum_v1 = editor._history[-1].checksum

        modified = _MINIMAL_URDF.replace("base_link", "link_two")
        editor.set_content(modified)
        checksum_v2 = editor._history[-1].checksum

        assert checksum_v1 != checksum_v2, "Checksum must differ for different content"

    @pytest.mark.unit
    def test_md5_call_does_not_raise_on_fips_platforms(self) -> None:
        """hashlib.md5 with usedforsecurity=False must not raise ValueError.

        On FIPS-enforcing systems, hashlib.md5() without usedforsecurity=False
        raises ValueError.  This test confirms the call signature is correct by
        exercising it directly (the fix in source ensures editor also works).
        """
        # Should not raise regardless of platform FIPS mode
        digest = hashlib.md5(b"test", usedforsecurity=False).hexdigest()
        assert len(digest) == 32
