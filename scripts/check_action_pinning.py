#!/usr/bin/env python3
"""Enforce immutable pinning of third-party GitHub Actions (issue #3255).

A mutable ``uses:`` reference (``owner/repo@v1``, ``@main``, ``@stable``) lets a
tag move or a compromised release silently change privileged CI behaviour with
no repository diff. This linter requires every *third-party* action to be pinned
to a full 40-hex-character commit SHA.

Policy
------
- Local actions (``./path`` or ``docker://...``) are ignored.
- Actions whose owner is in ``TRUSTED_TAG_OWNERS`` (GitHub's own first-party
  ``actions``/``github`` orgs) may use a version tag; they are maintained by
  GitHub and are the conventional allowlist boundary.
- Every other (third-party) action MUST be pinned to a full commit SHA. A
  trailing ``# vX.Y.Z`` comment is encouraged for readability and is allowed.

Run ``python scripts/check_action_pinning.py`` (optionally with explicit paths).
Exit code 0 = all good; 1 = violations found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# First-party GitHub orgs whose actions may be referenced by tag. These are
# maintained by GitHub itself and are the standard allowlist boundary for
# SHA-pinning policies. Everything else must be SHA-pinned.
TRUSTED_TAG_OWNERS = frozenset({"actions", "github"})

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# Matches:  uses: owner/repo@ref            (optionally followed by a comment)
#           uses: owner/repo/subdir@ref
_USES_RE = re.compile(
    r"""^\s*-?\s*uses:\s*["']?(?P<ref>[^"'\s#]+)["']?""",
    re.IGNORECASE,
)


def _iter_workflow_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.yml")))
            files.extend(sorted(path.glob("*.yaml")))
        elif path.suffix in {".yml", ".yaml"}:
            files.append(path)
    return files


def violations_for_reference(ref: str) -> str | None:
    """Return a violation message for a ``uses:`` reference, or None if pinned."""
    # Local or docker actions are not tag/SHA pinnable in the same sense.
    if ref.startswith("./") or ref.startswith("../") or ref.startswith("docker://"):
        return None
    if "@" not in ref:
        return f"{ref!r} has no version/SHA pin"

    repo, _, version = ref.partition("@")
    owner = repo.split("/", 1)[0]
    if owner in TRUSTED_TAG_OWNERS:
        return None
    if _SHA_RE.match(version):
        return None
    return (
        f"{ref!r} third-party action is not pinned to a 40-char commit SHA "
        f"(found {version!r}); pin it to a full SHA"
    )


def check_file(path: Path) -> list[str]:
    problems: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = _USES_RE.match(line)
        if not match:
            continue
        message = violations_for_reference(match.group("ref"))
        if message is not None:
            problems.append(f"{path}:{lineno}: {message}")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        targets = [Path(a) for a in args]
    else:
        repo_root = Path(__file__).resolve().parents[1]
        targets = [repo_root / ".github" / "workflows"]

    problems: list[str] = []
    for workflow in _iter_workflow_files(targets):
        problems.extend(check_file(workflow))

    if problems:
        sys.stderr.write("Unpinned third-party GitHub Actions found:\n")
        for problem in problems:
            sys.stderr.write(f"- {problem}\n")
        sys.stderr.write(
            "\nThird-party actions must be pinned to a full commit SHA. "
            "First-party actions/* and github/* may use a tag. See issue #3255.\n"
        )
        return 1

    sys.stdout.write("All third-party GitHub Actions are SHA-pinned.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
