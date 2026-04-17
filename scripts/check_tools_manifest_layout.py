#!/usr/bin/env python3
"""Validate high-confidence tool manifest layout contracts.

This guard intentionally checks only launcher-backed tool directories. A
directory with a launch script is a discoverable tool surface and must have a
sibling gui_registration.py so generated manifests and contract tests can see
it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

LAUNCHER_NAMES = frozenset({"launch_pyqt6.py", "launch_web.py", "launch_gui.py"})


@dataclass(frozen=True)
class ManifestLayoutIssue:
    path: str
    message: str


def _repo_relative(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def launcher_tool_dirs(repo_root: Path) -> list[Path]:
    """Return directories below src/ that expose a known tool launcher."""
    src_dir = repo_root / "src"
    if not src_dir.is_dir():
        return []

    tool_dirs = {
        path.parent
        for path in src_dir.rglob("*")
        if path.is_file() and path.name in LAUNCHER_NAMES
    }
    return sorted(tool_dirs)


def check_manifest_layout(repo_root: Path) -> list[ManifestLayoutIssue]:
    """Find launcher-backed tool directories that cannot enter manifests."""
    issues: list[ManifestLayoutIssue] = []
    for tool_dir in launcher_tool_dirs(repo_root):
        registration = tool_dir / "gui_registration.py"
        if registration.exists():
            continue
        launchers = sorted(path.name for path in tool_dir.iterdir() if path.name in LAUNCHER_NAMES)
        launcher_list = ", ".join(launchers)
        issues.append(
            ManifestLayoutIssue(
                path=_repo_relative(repo_root, tool_dir),
                message=(
                    "launcher-backed tool directory is missing "
                    f"gui_registration.py ({launcher_list})"
                ),
            )
        )
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check launcher-backed tool directories for manifest registration."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to inspect.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable issue data.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    issues = check_manifest_layout(repo_root)
    if args.json:
        sys.stdout.write(json.dumps([asdict(issue) for issue in issues], indent=2))
        sys.stdout.write("\n")
    elif issues:
        for issue in issues:
            sys.stderr.write(f"{issue.path}: {issue.message}\n")

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
