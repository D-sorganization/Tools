#!/usr/bin/env python3
"""Decide the release bump and write the delta changelog entry (release.yml).

Replaces the inline ``analyse-commits`` Python in ``.github/workflows/release.yml``
so the two behaviours that produced ten minor bumps in ten days and a 2 MB
``CHANGELOG.md`` (Tools #4910, RM #1507) are testable:

1. **Gating.** On a ``push`` to ``main`` the bump is decided by the HEAD commit
   subject only: ``feat`` -> minor, ``fix``/``perf`` -> patch, ``!``/``BREAKING``
   -> major. ``chore``/``docs``/``ci``/``test``/``refactor``/... and pushes by
   bots (dependabot, ``*[bot]``) never bump. On ``workflow_dispatch`` the
   operator's ``force_bump`` wins, else the bump is auto-detected over the
   whole delta.
2. **Delta only.** The changelog entry lists the commits since the previous
   release marker -- the newest ``v*`` tag, or, while the repository has no
   tags yet, the newest ``chore(release): bump version to vX.Y.Z`` commit --
   never the whole history again. Squash-merge commits whose subject carries
   no text (``@ (#4851)``) are resolved to the PR title through ``gh api``;
   if that fails they are rendered as ``PR #4851``. No row is ever ``- @``.

Outputs (``--github-output`` / ``--out``):
    bump, new_version, current_version  and the entry markdown file.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

BREAKING_RE = re.compile(r"BREAKING[ -]CHANGE|^[a-z]+(\([^)]*\))?!:")
CONVENTIONAL_RE = re.compile(
    r"^(?P<type>[a-z]+)(\((?P<scope>[^)]*)\))?(?P<bang>!)?:\s*(?P<desc>.*)$"
)
RELEASE_BUMP_RE = re.compile(r"^chore\(release\): bump version to v\d+\.\d+\.\d+")
PR_SUFFIX_RE = re.compile(r"\s*\(#(?P<pr>\d+)\)\s*$")

MINOR_TYPES = frozenset({"feat"})
PATCH_TYPES = frozenset({"fix", "perf"})
#: Types that never trigger a release on their own.
NO_BUMP_TYPES = frozenset(
    {"chore", "docs", "ci", "test", "refactor", "style", "build", "revert", "cleanup"}
)
BOT_ACTOR_RE = re.compile(r"(\[bot\]$|^dependabot|^github-actions|^renovate)", re.I)

SECTION_TITLES = (
    ("major", "Breaking changes"),
    ("feat", "Features"),
    ("fix", "Fixes"),
    ("perf", "Performance"),
    ("other", "Other"),
)


@dataclass(frozen=True)
class Commit:
    sha: str
    subject: str


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False, cwd=cwd
    ).stdout.strip()


# --------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------
def is_bot_actor(actor: str | None) -> bool:
    """True for dependabot / GitHub App / *[bot] actors."""
    if not actor:
        return False
    return BOT_ACTOR_RE.search(actor.strip()) is not None


def bump_for_subject(subject: str) -> str:
    """``major`` / ``minor`` / ``patch`` / ``none`` for one commit subject."""
    if RELEASE_BUMP_RE.match(subject):
        return "none"
    if BREAKING_RE.search(subject):
        return "major"
    match = CONVENTIONAL_RE.match(subject)
    if match is None:
        return "none"
    kind = match.group("type")
    if kind in MINOR_TYPES:
        return "minor"
    if kind in PATCH_TYPES:
        return "patch"
    return "none"


def highest_bump(bumps: list[str]) -> str:
    order = {"none": 0, "patch": 1, "minor": 2, "major": 3}
    return max(bumps, key=lambda b: order.get(b, 0), default="none")


def decide_bump(
    *,
    event: str,
    head_subject: str,
    actor: str | None,
    force_bump: str,
    delta_subjects: list[str],
) -> str:
    """Apply the gating rules from the module docstring."""
    force = (force_bump or "").strip().lower()
    if event == "workflow_dispatch":
        if force in ("major", "minor", "patch"):
            return force
        return highest_bump([bump_for_subject(s) for s in delta_subjects])
    # push: only the merge commit at HEAD decides, and never a bot's push.
    if is_bot_actor(actor):
        return "none"
    return bump_for_subject(head_subject)


def next_version(current: str, bump: str) -> str:
    major, minor, patch = (int(part) for part in current.split("."))
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    return current


def split_pr_suffix(subject: str) -> tuple[str, int | None]:
    """``"feat: x (#12)"`` -> ``("feat: x", 12)``."""
    match = PR_SUFFIX_RE.search(subject)
    if match is None:
        return subject.strip(), None
    return subject[: match.start()].strip(), int(match.group("pr"))


def subject_has_text(subject_body: str) -> bool:
    """False for ``@``, ``-``, empty -- anything without a letter or digit."""
    return re.search(r"[A-Za-z0-9]", subject_body) is not None


def render_entry_line(
    subject: str, resolve_pr_title: Callable[[int], str | None]
) -> str | None:
    """Return the ``- ...`` row for a commit, or None to drop it.

    Release-bump commits are dropped (they are not changes). A subject with no
    text is replaced by the PR title (``resolve_pr_title``) or ``PR #N``.
    """
    if RELEASE_BUMP_RE.match(subject):
        return None
    body, pr = split_pr_suffix(subject)
    if not subject_has_text(body):
        if pr is None:
            return None
        title = resolve_pr_title(pr)
        body = title.strip() if title and subject_has_text(title) else f"PR #{pr}"
    suffix = f" (#{pr})" if pr is not None else ""
    return f"- {body}{suffix}"


def section_for(line: str) -> str:
    body = line[2:] if line.startswith("- ") else line
    if BREAKING_RE.search(body):
        return "major"
    match = CONVENTIONAL_RE.match(body)
    if match is None:
        return "other"
    kind = match.group("type")
    if kind in ("feat", "fix", "perf"):
        return kind
    return "other"


def render_entry(lines: list[str]) -> str:
    """Group rows under Features / Fixes / Performance / Other; dedupe."""
    seen: set[str] = set()
    grouped: dict[str, list[str]] = {key: [] for key, _ in SECTION_TITLES}
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        grouped[section_for(line)].append(line)
    parts: list[str] = []
    for key, title in SECTION_TITLES:
        if grouped[key]:
            parts.append(f"#### {title}\n\n" + "\n".join(grouped[key]))
    if not parts:
        return "- No changes since the previous release marker.\n"
    return "\n\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# Git / GitHub access
# --------------------------------------------------------------------------
def previous_release_marker(cwd: Path | None = None) -> str | None:
    """Newest ``v*`` tag, else the newest release-bump commit sha, else None."""
    tags = [
        t.strip()
        for t in _git(
            "tag", "--sort=-version:refname", "--list", "v[0-9]*", cwd=cwd
        ).splitlines()
        if t.strip()
    ]
    if tags:
        return tags[0]
    sha = _git(
        "log",
        "-1",
        "--format=%H",
        "--grep=^chore(release): bump version to v",
        cwd=cwd,
    )
    return sha or None


def delta_commits(marker: str | None, cwd: Path | None = None) -> list[Commit]:
    rev_range = f"{marker}..HEAD" if marker else "HEAD"
    out = _git("log", rev_range, "--format=%H%x00%s", "--no-merges", cwd=cwd)
    commits: list[Commit] = []
    for row in out.splitlines():
        sha, _, subject = row.partition("\x00")
        if sha:
            commits.append(Commit(sha=sha, subject=subject.strip()))
    return commits


def gh_pr_title(repo: str, number: int) -> str | None:
    if not repo:
        return None
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/pulls/{number}", "--jq", ".title"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--current-version", required=True)
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", "push"))
    parser.add_argument("--actor", default=os.environ.get("GITHUB_ACTOR", ""))
    parser.add_argument("--force-bump", default=os.environ.get("FORCE_BUMP", ""))
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--head-subject", default=None, help="Defaults to git log -1")
    parser.add_argument("--out", default="release-changelog-entry.md")
    parser.add_argument(
        "--github-output",
        default=os.environ.get("GITHUB_OUTPUT"),
        help="File to append bump/new_version/current_version to",
    )
    args = parser.parse_args(argv)

    head_subject = args.head_subject or _git("log", "-1", "--format=%s")
    marker = previous_release_marker()
    commits = delta_commits(marker)
    subjects = [c.subject for c in commits]

    bump = decide_bump(
        event=args.event,
        head_subject=head_subject,
        actor=args.actor,
        force_bump=args.force_bump,
        delta_subjects=subjects,
    )
    new_version = next_version(args.current_version, bump)

    lines = [
        row
        for row in (
            render_entry_line(s, lambda n: gh_pr_title(args.repo, n)) for s in subjects
        )
        if row is not None
    ]
    entry = render_entry(lines)
    Path(args.out).write_text(entry, encoding="utf-8", newline="\n")

    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as fh:
            fh.write(f"bump={bump}\n")
            fh.write(f"new_version={new_version}\n")
            fh.write(f"current_version={args.current_version}\n")

    sys.stdout.write(
        f"event     : {args.event}\n"
        f"head      : {head_subject}\n"
        f"actor     : {args.actor}\n"
        f"marker    : {marker or '<none: whole history>'}\n"
        f"delta     : {len(commits)} commits, {len(lines)} rows\n"
        f"Bump type : {bump}\n"
        f"Current   : {args.current_version}\n"
        f"New       : {new_version}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
