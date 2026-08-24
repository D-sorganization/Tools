"""Trusted Git executable discovery.

This module avoids PATH-based executable resolution for subprocess git calls.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

TRUSTED_GIT_ENV_VAR = "D_SORG_TRUSTED_GIT"


def _default_git_candidates(env: Mapping[str, str]) -> tuple[Path, ...]:
    if os.name == "nt":
        return tuple(
            Path(root) / suffix
            for root in (
                env.get("ProgramW6432"),
                env.get("ProgramFiles"),
                env.get("ProgramFiles(x86)"),
                env.get("LocalAppData"),
            )
            if root
            for suffix in (
                Path("Git") / "cmd" / "git.exe",
                Path("Git") / "bin" / "git.exe",
                Path("Programs") / "Git" / "cmd" / "git.exe",
                Path("Programs") / "Git" / "bin" / "git.exe",
            )
        )
    return (
        Path("/usr/bin/git"),
        Path("/usr/local/bin/git"),
        Path("/opt/homebrew/bin/git"),
        Path("/opt/local/bin/git"),
    )


def _resolve_absolute_candidate(candidate: str | os.PathLike[str]) -> str | None:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        return None
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_file():
        return None
    return str(resolved)


def resolve_trusted_git_executable(
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Return a trusted absolute Git executable path, or ``None``."""
    environment = os.environ if env is None else env

    override = environment.get(TRUSTED_GIT_ENV_VAR)
    if override:
        resolved = _resolve_absolute_candidate(override)
        if resolved is not None:
            return resolved

    for candidate in _default_git_candidates(environment):
        resolved = _resolve_absolute_candidate(candidate)
        if resolved is not None:
            return resolved

    return None
