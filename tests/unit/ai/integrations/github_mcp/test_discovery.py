"""Tests for ``is_github_mcp_available`` — pre-flight discovery."""

from __future__ import annotations

from unittest.mock import patch

from src.shared.python.ai.integrations.github_mcp import discovery


def _patch_env(token: str | None) -> dict[str, str]:
    """Build an environ dict optionally containing the token env var."""
    env: dict[str, str] = {}
    if token is not None:
        env["GITHUB_PERSONAL_ACCESS_TOKEN"] = token
    return env


def test_available_when_token_and_npx_present() -> None:
    with (
        patch.dict("os.environ", _patch_env("ghp_real"), clear=True),
        patch.object(discovery.shutil, "which", return_value="/usr/bin/npx"),
    ):
        ok, msg = discovery.is_github_mcp_available()
    assert ok is True
    assert "available" in msg.lower()


def test_unavailable_when_token_missing() -> None:
    with (
        patch.dict("os.environ", _patch_env(None), clear=True),
        patch.object(discovery.shutil, "which", return_value="/usr/bin/npx"),
    ):
        ok, msg = discovery.is_github_mcp_available()
    assert ok is False
    assert "GITHUB_PERSONAL_ACCESS_TOKEN" in msg


def test_unavailable_when_token_blank() -> None:
    with (
        patch.dict("os.environ", _patch_env("   "), clear=True),
        patch.object(discovery.shutil, "which", return_value="/usr/bin/npx"),
    ):
        ok, msg = discovery.is_github_mcp_available()
    assert ok is False
    assert "GITHUB_PERSONAL_ACCESS_TOKEN" in msg


def test_unavailable_when_npx_missing() -> None:
    with (
        patch.dict("os.environ", _patch_env("ghp_real"), clear=True),
        patch.object(discovery.shutil, "which", return_value=None),
    ):
        ok, msg = discovery.is_github_mcp_available()
    assert ok is False
    assert "npx" in msg.lower()


def test_unavailable_message_actionable() -> None:
    """Operator-facing messages must guide remediation, not just say ``no``."""
    with (
        patch.dict("os.environ", _patch_env(None), clear=True),
        patch.object(discovery.shutil, "which", return_value=None),
    ):
        ok, msg = discovery.is_github_mcp_available()
    assert ok is False
    # Whatever check fails first, the message must mention what to fix.
    assert msg.strip()
    assert len(msg) > 10


def test_check_token_helper_via_explicit_arg() -> None:
    """The helper accepts an explicit token override (used by Prefs UI)."""
    with (
        patch.dict("os.environ", _patch_env(None), clear=True),
        patch.object(discovery.shutil, "which", return_value="/usr/bin/npx"),
    ):
        ok, msg = discovery.is_github_mcp_available(token="ghp_supplied")
    assert ok is True, msg
