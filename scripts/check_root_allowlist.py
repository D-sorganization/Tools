#!/usr/bin/env python3
"""Fail when a tracked top-level repository entry is not on the explicit allowlist.

Root hygiene gate for the Fleet Readiness Program (Repository_Management#1505,
Tools#4917). Agent scratch files, generated reports, machine-specific symlinks
and one-off prototypes tend to accumulate at the repository root because
nothing rejects them. This check lists the first path component of every
tracked path (``git ls-files``) and fails, naming the offenders, when any of
them is missing from ``ROOT_ALLOWLIST``.

Adding a new top-level entry is a deliberate act: extend ``ROOT_ALLOWLIST``
in the same change and say why in the PR.

Usage::

    python scripts/check_root_allowlist.py            # gate the current repo
    python scripts/check_root_allowlist.py --list     # print tracked entries
    python scripts/check_root_allowlist.py --root X   # gate another checkout
"""

from __future__ import annotations

import argparse
import shutil
import subprocess  # nosec B404 - fixed Git executable and argv only.
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every tracked top-level entry that is allowed to exist. Keep sorted.
ROOT_ALLOWLIST: frozenset[str] = frozenset(
    {
        ".agent",  # agent skill definitions (.agent/skills/*)
        ".benchmarks",  # performance SLA docs, not pytest-benchmark output
        ".cargo",  # cargo-audit configuration for rust_core
        ".claude",
        ".cursor",  # Cursor IDE rules
        ".docker",  # container entrypoint
        ".dockerignore",
        ".editorconfig",
        ".env.example",
        ".gaai",  # GAAI agent knowledge base (tracked docs)
        ".gitattributes",
        ".github",
        ".gitignore",
        ".jules",  # Jules agent prompts and completist data
        ".mypy_cache",  # only a .gitkeep so the cache dir exists on runners
        ".pre-commit-config.yaml",
        ".prettierignore",
        ".secrets.baseline",  # detect-secrets baseline
        "AGENTS.md",
        "AGENT_HANDOFF.md",
        "CHANGELOG.md",
        "CLAUDE.md",
        "CONTRIBUTING.md",
        "Cargo.toml",
        "Chaotic_Pendulum",  # legacy top-level tool; relocation is Phase 1 work
        "Dockerfile",
        "Dockerfile.dev",
        "Dockerfile.prod",
        "LICENSE",
        "MANIFEST.in",
        "Makefile",
        "README.md",
        "SECURITY.md",
        "SPEC.md",
        "UnifiedToolsLauncher.py",  # launcher entry point (desktop shortcuts)
        "VERSION",
        "_bootstrap.py",  # sys.path bootstrap imported by root entry points
        "agent_templates",
        "assessments",  # dated repository assessments
        "assets",
        "benchmarks",
        "build_hooks.py",  # hatch build hooks referenced from pyproject.toml
        "commit_screensaver.py",  # root tool entry point
        "config",
        "conftest.py",
        "convert_tools_icon.py",  # generates assets/tools_icon*.ico
        "create_launcher_shortcut.ps1",
        "create_launcher_shortcut_png.ps1",
        "create_requested_shortcuts.ps1",
        "deploy",
        "docker-compose.yml",
        "docs",
        "drafts",  # Jules-Code-Quality-Reviewer.yml is cited by docs/assessments
        "examples",
        "generate_real_assessments.py",
        "helm",
        "launch.py",
        "launch_signal_toolkit.py",
        "manuals",  # Tools design manual sources and generated artifacts
        "matlab",  # legacy MATLAB tools; relocation is Phase 1 work
        "model_pack.yaml",
        "mypy.ini",
        "output",  # only a .gitkeep; tool output directory
        "package-lock.json",
        "package.json",
        "pyproject.toml",
        "requirements-lock.txt",
        "requirements-rate-pyqt.txt",
        "requirements.txt",
        "ruff.toml",
        "run_impact_explorer.bat",
        "run_tile_launcher.py",
        "rust-toolchain.toml",
        "rust_core",
        "schema",  # pid_spec JSON schema (singular, legacy location)
        "schemas",  # mocap JSON schemas
        "scripts",
        "setup.py",
        "setup_dev.py",
        "shared_scripts",  # fleet_hooks.py shared with downstream repos
        "src",
        "start-gaai-daemon.sh",
        "tests",
        "tool_surface_contract.json",
        "tools.json",  # generated tool registry (scripts/generate_tools_json.py)
        "uv.lock",
        "verify_launcher.py",
        "visual-baseline-candidates",  # only a .gitkeep; CI evidence upload dir
        "wave_solver.py",
    }
)


def tracked_top_level_entries(root: Path) -> tuple[str, ...]:
    """Return the sorted, de-duplicated first path components of tracked files."""

    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to list tracked top-level entries")
    result = subprocess.run(  # nosec B603 - fixed Git argv, no shell.
        [git, "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    entries = {
        record.split("/", 1)[0]
        for record in result.stdout.decode("utf-8").split(chr(0))
        if record
    }
    return tuple(sorted(entries))


def disallowed_entries(
    entries: Iterable[str], allowlist: frozenset[str] = ROOT_ALLOWLIST
) -> tuple[str, ...]:
    """Return the entries that are not on the allowlist, sorted."""

    return tuple(sorted(entry for entry in entries if entry not in allowlist))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository checkout to inspect (default: this repository)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the tracked top-level entries and exit 0",
    )
    args = parser.parse_args(argv)

    entries = tracked_top_level_entries(args.root.resolve())
    if args.list:
        for entry in entries:
            sys.stdout.write(f"{entry}\n")
        return 0

    offenders = disallowed_entries(entries)
    if offenders:
        sys.stderr.write(
            "Root allowlist check failed. Tracked top-level entries not in "
            "scripts/check_root_allowlist.py ROOT_ALLOWLIST:\n"
        )
        for entry in offenders:
            sys.stderr.write(f"- {entry}\n")
        sys.stderr.write(
            "Move the entry under an existing directory, or add it to the "
            "allowlist deliberately in the same change.\n"
        )
        return 1

    sys.stdout.write(
        f"Root allowlist check passed ({len(entries)} tracked top-level entries).\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
