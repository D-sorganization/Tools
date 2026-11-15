"""Tier-2 CLI discovery probe and Tier-3 subscription qualification checks (#5727).

Provides deterministic discovery verification (resolved path, ``--version``,
and ``--help`` flag acceptance) via :func:`run_cli_subprocess` without invoking
any model turn, plus validation helpers that reject qualification records run
under pay-per-use API key environment variables.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from shared.python.ai.adapters.cli_subprocess_runner import (
    PromptDelivery,
    run_cli_subprocess,
)
from shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_DISCOVERY_TIMEOUT = 10.0
FORBIDDEN_API_KEY_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class CliDiscoverySpec:
    """Expected CLI binary and required adapter flags for one provider."""

    provider: str
    binary_name: str
    required_help_flags: tuple[str, ...]


DEFAULT_CLI_DISCOVERY_SPECS: tuple[CliDiscoverySpec, ...] = (
    CliDiscoverySpec(
        provider="claude_code",
        binary_name="claude",
        required_help_flags=("-p", "--model"),
    ),
    CliDiscoverySpec(
        provider="codex_cli",
        binary_name="codex",
        required_help_flags=("exec", "--skip-git-repo-check", "--model"),
    ),
    CliDiscoverySpec(
        provider="agy",
        binary_name="agy",
        required_help_flags=("--skip-trust", "-p", "--model"),
    ),
)


@dataclass(frozen=True)
class CliDiscoveryProbeResult:
    """Structured result of probing a subscription CLI binary (Oracle 16)."""

    provider: str
    os_name: str
    resolved_path: str | None
    version_output: str | None
    flags_accepted: bool
    missing_flags: tuple[str, ...]
    probe_ok: bool
    detail: str


@dataclass(frozen=True)
class CliQualificationRecord:
    """Recorded Tier-3 single-turn subscription qualification artifact (Oracle 17)."""

    provider: str
    cli_version: str
    os_name: str
    exit_code: int
    duration_seconds: float
    transcript_sha256: str
    used_api_key_env: bool = False


def detect_forbidden_api_key_env(
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return sorted names of non-empty API-key env vars in ``env``."""
    source = os.environ if env is None else env
    present = [
        name for name in FORBIDDEN_API_KEY_ENV_VARS if source.get(name, "").strip()
    ]
    return tuple(sorted(present))


def _resolve_target_binary(binary_path: str) -> str | None:
    if not binary_path or not binary_path.strip():
        raise ValueError("binary_path must be a non-empty string")
    candidate = Path(binary_path)
    if candidate.is_file():
        return str(candidate)
    if candidate.parent == Path("."):
        return shutil.which(binary_path)
    return None


def probe_cli_provider_discovery(
    provider: str,
    binary_path: str,
    required_help_flags: tuple[str, ...] = (),
    *,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
) -> CliDiscoveryProbeResult:
    """Probe ``binary_path`` with ``--version`` and ``--help`` only (no model call).

    Preconditions:
        - ``provider`` and ``binary_path`` are non-empty strings.
        - ``timeout`` is strictly positive.
    Postconditions:
        - Never invokes a live prompt turn.
        - ``probe_ok`` is True iff the binary exists, ``--version`` exited 0,
          and every entry in ``required_help_flags`` appears in ``--help`` output.
    """
    if not provider or not provider.strip():
        raise ValueError("provider must be a non-empty string")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    os_name = os.name
    resolved = _resolve_target_binary(binary_path)
    if resolved is None:
        return CliDiscoveryProbeResult(
            provider=provider,
            os_name=os_name,
            resolved_path=None,
            version_output=None,
            flags_accepted=False,
            missing_flags=required_help_flags,
            probe_ok=False,
            detail=f"CLI binary not found for {provider}: {binary_path}",
        )

    try:
        ver_res = run_cli_subprocess(
            [resolved, "--version"],
            prompt=None,
            prompt_delivery=PromptDelivery.STDIN,
            timeout=timeout,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001 — surface discovery failures without crashing
        logger.warning("Discovery --version probe failed for %s: %s", provider, exc)
        return CliDiscoveryProbeResult(
            provider=provider,
            os_name=os_name,
            resolved_path=resolved,
            version_output=None,
            flags_accepted=False,
            missing_flags=required_help_flags,
            probe_ok=False,
            detail=f"--version probe failed for {provider}: {exc}",
        )

    version_text = (ver_res.stdout or ver_res.stderr or "").strip() or None
    if ver_res.returncode != 0:
        return CliDiscoveryProbeResult(
            provider=provider,
            os_name=os_name,
            resolved_path=resolved,
            version_output=version_text,
            flags_accepted=False,
            missing_flags=required_help_flags,
            probe_ok=False,
            detail=f"--version exited {ver_res.returncode} for {provider}",
        )

    try:
        help_res = run_cli_subprocess(
            [resolved, "--help"],
            prompt=None,
            prompt_delivery=PromptDelivery.STDIN,
            timeout=timeout,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001 — surface discovery failures without crashing
        logger.warning("Discovery --help probe failed for %s: %s", provider, exc)
        return CliDiscoveryProbeResult(
            provider=provider,
            os_name=os_name,
            resolved_path=resolved,
            version_output=version_text,
            flags_accepted=False,
            missing_flags=required_help_flags,
            probe_ok=False,
            detail=f"--help probe failed for {provider}: {exc}",
        )

    help_text = f"{help_res.stdout}\n{help_res.stderr}"
    missing = tuple(flag for flag in required_help_flags if flag not in help_text)
    flags_ok = help_res.returncode == 0 and not missing
    probe_ok = flags_ok
    if probe_ok:
        detail = f"Discovery verified for {provider} ({version_text or resolved})"
    elif help_res.returncode != 0:
        detail = f"--help exited {help_res.returncode} for {provider}"
    else:
        detail = (
            f"Missing expected flags in --help for {provider}: {', '.join(missing)}"
        )

    return CliDiscoveryProbeResult(
        provider=provider,
        os_name=os_name,
        resolved_path=resolved,
        version_output=version_text,
        flags_accepted=flags_ok,
        missing_flags=missing,
        probe_ok=probe_ok,
        detail=detail,
    )


def validate_qualification_record(
    record: CliQualificationRecord,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Validate a Tier-3 CLI subscription qualification record (Oracle 17).

    Returns ``(is_valid, reasons)``. Any run performed with a pay-per-use API key
    environment variable set is invalid by definition.
    """
    reasons: list[str] = []
    if not record.provider.strip():
        reasons.append("provider must be non-empty")
    if not record.cli_version.strip():
        reasons.append("cli_version must be non-empty")
    if not record.os_name.strip():
        reasons.append("os_name must be non-empty")
    if record.exit_code != 0:
        reasons.append(f"exit_code must be 0, got {record.exit_code}")
    if record.duration_seconds <= 0:
        reasons.append("duration_seconds must be positive")
    if not _SHA256_RE.match(record.transcript_sha256):
        reasons.append("transcript_sha256 must be a 64-character hex digest")
    if record.used_api_key_env:
        reasons.append(
            "used_api_key_env=True: subscription-only runs cannot use API keys"
        )
    forbidden = detect_forbidden_api_key_env(env)
    if forbidden:
        reasons.append(
            "Forbidden API-key environment variable(s) set: " + ", ".join(forbidden)
        )
    return not reasons, tuple(reasons)


__all__ = [
    "DEFAULT_CLI_DISCOVERY_SPECS",
    "DEFAULT_DISCOVERY_TIMEOUT",
    "FORBIDDEN_API_KEY_ENV_VARS",
    "CliDiscoveryProbeResult",
    "CliDiscoverySpec",
    "CliQualificationRecord",
    "detect_forbidden_api_key_env",
    "probe_cli_provider_discovery",
    "validate_qualification_record",
]
