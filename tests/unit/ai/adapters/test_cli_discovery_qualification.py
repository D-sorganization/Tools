"""Tests for CLI adapter discovery probes and subscription qualification records."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from shared.python.ai.adapters.cli_qualification import (
    DEFAULT_CLI_DISCOVERY_SPECS,
    CliDiscoveryProbeResult,
    CliDiscoverySpec,
    CliQualificationRecord,
    detect_forbidden_api_key_env,
    probe_cli_provider_discovery,
    validate_qualification_record,
)


def _create_fake_cli_script(tmp_path: Path, name: str, script_body: str) -> str:
    """Create a cross-platform runnable fake CLI script on disk."""
    py_path = tmp_path / f"{name}_fake.py"
    py_path.write_text(script_body, encoding="utf-8")
    if os.name == "nt":
        cmd_path = tmp_path / f"{name}.cmd"
        cmd_path.write_text(
            f'@echo off\n"{sys.executable}" "{py_path}" %*\n', encoding="utf-8"
        )
        return str(cmd_path)
    sh_path = tmp_path / name
    sh_path.write_text(
        f'#!/usr/bin/env sh\nexec "{sys.executable}" "{py_path}" "$@"\n',
        encoding="utf-8",
    )
    sh_path.chmod(sh_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(sh_path)


def test_probe_missing_binary_reports_not_found(tmp_path: Path) -> None:
    """Missing CLI binary returns probe_ok=False with resolved_path=None."""
    missing = str(tmp_path / "does_not_exist_cli")
    result = probe_cli_provider_discovery(
        provider="agy",
        binary_path=missing,
        required_help_flags=("--skip-trust", "-p"),
    )
    assert isinstance(result, CliDiscoveryProbeResult)
    assert not result.probe_ok
    assert result.resolved_path is None
    assert not result.flags_accepted
    assert "not found" in result.detail.lower() or "missing" in result.detail.lower()


def test_probe_valid_cli_verifies_version_and_required_flags(tmp_path: Path) -> None:
    """Fake binary answering --version and --help flags passes Oracle 16."""
    exe = _create_fake_cli_script(
        tmp_path,
        "agy",
        (
            "import sys\n"
            "args = sys.argv[1:]\n"
            "if '--version' in args:\n"
            "    print('agy 2.4.1')\n"
            "elif '--help' in args:\n"
            "    print('Usage: agy --skip-trust -p <prompt> --model <name>')\n"
        ),
    )
    result = probe_cli_provider_discovery(
        provider="agy",
        binary_path=exe,
        required_help_flags=("--skip-trust", "-p", "--model"),
    )
    assert result.probe_ok
    assert result.flags_accepted
    assert result.resolved_path == exe
    assert "agy 2.4.1" in (result.version_output or "")
    assert result.missing_flags == ()


def test_probe_failing_version_probe_sets_probe_ok_false(tmp_path: Path) -> None:
    """Nonzero --version exit fails discovery with probe_ok and flags_accepted false."""
    exe = _create_fake_cli_script(
        tmp_path,
        "codex",
        (
            "import sys\n"
            "if '--version' in sys.argv[1:]:\n"
            "    sys.stderr.write('broken\\n')\n"
            "    sys.exit(17)\n"
            "print('Usage: codex exec -p')\n"
        ),
    )
    result = probe_cli_provider_discovery(
        provider="codex_cli",
        binary_path=exe,
        required_help_flags=("exec",),
    )
    assert not result.probe_ok
    assert not result.flags_accepted
    assert result.resolved_path == exe
    assert "17" in result.detail or "version" in result.detail.lower()


def test_probe_failing_help_probe_sets_probe_ok_false(tmp_path: Path) -> None:
    """Fake CLI exiting nonzero on --help fails discovery even when --version ok."""
    exe = _create_fake_cli_script(
        tmp_path,
        "claude",
        (
            "import sys\n"
            "args = sys.argv[1:]\n"
            "if '--version' in args:\n"
            "    print('claude 1.0.0')\n"
            "elif '--help' in args:\n"
            "    sys.exit(2)\n"
        ),
    )
    result = probe_cli_provider_discovery(
        provider="claude_code",
        binary_path=exe,
        required_help_flags=("-p",),
    )
    assert not result.probe_ok
    assert not result.flags_accepted
    assert result.resolved_path == exe
    assert "claude 1.0.0" in (result.version_output or "")
    assert "2" in result.detail or "help" in result.detail.lower()


def test_probe_detects_unsupported_required_flags(tmp_path: Path) -> None:
    """Fake executable omitting --skip-trust from --help fails discovery cleanly."""
    exe = _create_fake_cli_script(
        tmp_path,
        "agy",
        (
            "import sys\n"
            "args = sys.argv[1:]\n"
            "if '--version' in args:\n"
            "    print('agy 0.1.0')\n"
            "elif '--help' in args:\n"
            "    print('Usage: agy -p <prompt>')\n"
        ),
    )
    result = probe_cli_provider_discovery(
        provider="agy",
        binary_path=exe,
        required_help_flags=("--skip-trust", "-p"),
    )
    assert not result.probe_ok
    assert not result.flags_accepted
    assert result.missing_flags == ("--skip-trust",)
    assert "--skip-trust" in result.detail


def test_default_discovery_specs_cover_all_three_subscription_clis() -> None:
    """Discovery matrix defines specs for claude_code, codex_cli, and agy."""
    providers = {spec.provider for spec in DEFAULT_CLI_DISCOVERY_SPECS}
    assert {"claude_code", "codex_cli", "agy"} <= providers
    by_provider = {spec.provider: spec for spec in DEFAULT_CLI_DISCOVERY_SPECS}
    assert isinstance(by_provider["codex_cli"], CliDiscoverySpec)
    assert "--skip-git-repo-check" in by_provider["codex_cli"].required_help_flags
    assert "--skip-trust" in by_provider["agy"].required_help_flags


def test_detect_forbidden_api_key_env_and_qualification_validation() -> None:
    """Qualification records using an API key env var or invalid fields fail."""
    clean_env = {"PATH": "/usr/bin", "HOME": "/tmp"}
    assert detect_forbidden_api_key_env(clean_env) == ()

    dirty_env = {
        "ANTHROPIC_API_KEY": "sk-ant-not-allowed",  # pragma: allowlist secret
        "GEMINI_API_KEY": "   ",
    }
    assert detect_forbidden_api_key_env(dirty_env) == ("ANTHROPIC_API_KEY",)

    valid_record = CliQualificationRecord(
        provider="claude_code",
        cli_version="claude 1.0.50",
        os_name="Windows",
        exit_code=0,
        duration_seconds=3.2,
        transcript_sha256="a" * 64,
        used_api_key_env=False,
    )
    ok, reasons = validate_qualification_record(valid_record, env=clean_env)
    assert ok
    assert reasons == ()

    ok_dirty, reasons_dirty = validate_qualification_record(valid_record, env=dirty_env)
    assert not ok_dirty
    assert any("ANTHROPIC_API_KEY" in reason for reason in reasons_dirty)

    api_key_flagged = CliQualificationRecord(
        provider="claude_code",
        cli_version="claude 1.0.50",
        os_name="Windows",
        exit_code=0,
        duration_seconds=3.2,
        transcript_sha256="b" * 64,
        used_api_key_env=True,
    )
    ok_flagged, reasons_flagged = validate_qualification_record(
        api_key_flagged, env=clean_env
    )
    assert not ok_flagged
    assert any("used_api_key_env" in reason for reason in reasons_flagged)


def test_tier2_installed_cli_discovery_no_model_call() -> None:
    """Optional Tier-2 check verifies --version/--help on any locally installed CLI."""
    import shutil

    checked = 0
    for spec in DEFAULT_CLI_DISCOVERY_SPECS:
        found = shutil.which(spec.binary_name)
        if not found:
            continue
        checked += 1
        result = probe_cli_provider_discovery(
            provider=spec.provider,
            binary_path=found,
            required_help_flags=spec.required_help_flags,
        )
        assert isinstance(result, CliDiscoveryProbeResult)
        assert result.resolved_path == found
        assert result.probe_ok, f"{spec.provider} failed discovery: {result.detail}"
        assert result.flags_accepted, (
            f"{spec.provider} missing flags: {result.missing_flags}"
        )
    if checked == 0:
        pytest.skip("No live subscription CLI binaries on PATH in this environment")
