"""anti_phantom_merge_guard.py — Python implementation of the anti-phantom-merge guard.

This script is the authoritative Python reference implementation of the guard
logic that is also expressed as a shell script inside
``.github/workflows/anti-phantom-merge.yml``.  It can be run locally for
debugging and is tested by ``tests/ops/test_anti_phantom_merge_guard.py``.

Race-condition fix (issue #2949)
---------------------------------
The original shell script resolved ``MERGE_BASE`` dynamically from
``origin/<base_branch>`` which could still be propagating at workflow
startup.  This Python version accepts ``base_sha`` and ``head_sha``
as **explicit parameters** sourced from ``github.event.pull_request.base.sha``
and ``github.event.pull_request.head.sha`` — values that GitHub guarantees
are stable at the moment the event fires.

Usage
-----
    python scripts/anti_phantom_merge_guard.py \\
        --base-sha <BASE_SHA> \\
        --head-sha <HEAD_SHA> \\
        --pr-title "feat: ..." \\
        --pr-body "Closes #123" \\
        [--repo D-sorganization/Tools] \\
        [--pr-number 999]

Exit codes
----------
0 — all rules passed
1 — one or more rules failed (failures printed to stdout)
2 — usage / precondition error
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Feature-claim pattern (matches Rule 2)
# ---------------------------------------------------------------------------

_FEATURE_CLAIM_RE = re.compile(
    r"^feat[:\(]|Implement|System|Engine|Framework",
    re.IGNORECASE,
)

# Prefixes that classify a path as "implementation" for Rule 2
_IMPL_PREFIXES = ("src/", "rust_core/", "api/")


# ---------------------------------------------------------------------------
# Pure helpers (all I/O isolated for testability)
# ---------------------------------------------------------------------------


def count_changed_files(diff_output: str) -> int:
    """Return the number of changed files in a ``git diff --name-only`` output.

    Precondition: diff_output is a string (may be empty).
    Postcondition: returns a non-negative integer.
    """
    assert isinstance(diff_output, str), "diff_output must be a str"
    lines = [ln for ln in diff_output.splitlines() if ln.strip()]
    return len(lines)


def get_diff_via_git(*, base_sha: str, head_sha: str) -> str:
    """Return the raw ``git diff --name-only`` output for the given SHAs.

    This function uses *pinned* SHAs from the GitHub event payload so that
    the result is not affected by concurrent pushes or slow ref propagation
    (the race described in issue #2949).

    Preconditions:
    - base_sha must be non-empty
    - head_sha must be non-empty
    """
    assert base_sha, "base_sha must be a non-empty string"
    assert head_sha, "head_sha must be a non-empty string"

    # Compute the merge base between the two pinned SHAs.
    # Using explicit SHAs (not branch names) avoids the race where the ref
    # for e.g. origin/main has not yet propagated.
    try:
        merge_base = (
            subprocess.check_output(
                ["git", "merge-base", base_sha, head_sha],
                text=False,
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        # Fallback: use base_sha directly if merge-base fails
        merge_base = base_sha

    raw = subprocess.check_output(
        ["git", "diff", "--name-only", f"{merge_base}...{head_sha}"],
        text=False,
    )
    return raw.decode()


def _has_impl_files(diff_output: str) -> bool:
    """Return True if the diff touches at least one implementation file."""
    for line in diff_output.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(p) for p in _IMPL_PREFIXES):
            return True
    return False


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------


def check_rule_1(*, num_changed: int, pr_title: str) -> str | None:
    """Rule 1 — empty diff.

    Returns an error message string if the rule is violated, None otherwise.

    Preconditions:
    - num_changed >= 0
    - pr_title is a string
    """
    assert num_changed >= 0, "num_changed must be >= 0"
    assert isinstance(pr_title, str), "pr_title must be a str"

    if num_changed == 0 and pr_title != "chore: empty PR":
        return (
            "Rule 1 (empty diff): This PR has 0 changed files against the base "
            "branch. Empty PRs are not mergeable. If you intentionally want an "
            "empty PR, set the title to exactly `chore: empty PR`."
        )
    return None


def check_rule_2(
    *,
    pr_title: str,
    changed_files_output: str,
    num_changed: int,
) -> str | None:
    """Rule 2 — feature claim with no implementation files.

    Returns an error message string if the rule is violated, None otherwise.

    Preconditions:
    - pr_title is a string
    - changed_files_output is a string
    - num_changed >= 0
    """
    assert isinstance(pr_title, str), "pr_title must be a str"
    assert isinstance(changed_files_output, str), "changed_files_output must be a str"
    assert num_changed >= 0, "num_changed must be >= 0"

    # Skip Rule 2 when Rule 1 already covers the empty-diff case.
    if num_changed == 0:
        return None

    if not _FEATURE_CLAIM_RE.search(pr_title):
        return None

    if not _has_impl_files(changed_files_output):
        return (
            f"Rule 2 (feature claim, no implementation): PR title `{pr_title}` "
            "claims a feature/system/engine/framework, but the diff touches "
            "no files under `src/`, `rust_core/`, or `api/`. "
            "Either retitle the PR or add the implementation."
        )
    return None


def check_rule_4(*, commits: list[str]) -> str | None:
    """Rule 4 — bot-only commits.

    *commits* is a list of ``<sha>|<author>|<subject>`` strings, newest first.

    Returns an error message string if the rule is violated, None otherwise.

    Preconditions:
    - commits is a list of strings
    """
    assert isinstance(commits, list), "commits must be a list"

    if not commits:
        return None

    last_line = commits[0]
    parts = last_line.split("|", 2)
    if len(parts) < 3:  # noqa: PLR2004
        return None

    last_author = parts[1]
    last_subject = parts[2]

    if last_author != "github-actions[bot]":
        return None
    if not last_subject.startswith("Merge branch 'main'"):
        return None

    # Check if there are any non-bot commits
    human_commits = [c for c in commits if "|github-actions[bot]|" not in c]
    if not human_commits:
        return (
            "Rule 4 (bot-only commits): The only commits between this branch and "
            "`main` are auto-merge-bot commits. The branch appears to have been "
            "absorbed into main via a sibling PR — there is no original work left "
            "to merge. Close this PR."
        )
    return None


# ---------------------------------------------------------------------------
# Top-level coordinator
# ---------------------------------------------------------------------------


def run_all_checks(
    *,
    base_sha: str,
    head_sha: str,
    pr_title: str,
    pr_body: str,
    diff_output: str,
    commits: list[str],
) -> list[str]:
    """Run all phantom-guard rules and return a list of failure messages.

    An empty list means all rules passed.

    This function accepts pre-computed ``diff_output`` and ``commits`` so that
    the pure logic can be tested without network or git access.

    Preconditions:
    - base_sha and head_sha are non-empty strings
    - pr_title is a string
    - pr_body is a string
    - diff_output is a string
    - commits is a list of strings
    """
    assert base_sha, "base_sha must be non-empty"
    assert head_sha, "head_sha must be non-empty"
    assert isinstance(pr_title, str), "pr_title must be a str"
    assert isinstance(pr_body, str), "pr_body must be a str"
    assert isinstance(diff_output, str), "diff_output must be a str"
    assert isinstance(commits, list), "commits must be a list"

    num_changed = count_changed_files(diff_output)
    failures: list[str] = []

    r1 = check_rule_1(num_changed=num_changed, pr_title=pr_title)
    if r1 is not None:
        failures.append(r1)

    r2 = check_rule_2(
        pr_title=pr_title,
        changed_files_output=diff_output,
        num_changed=num_changed,
    )
    if r2 is not None:
        failures.append(r2)

    r4 = check_rule_4(commits=commits)
    if r4 is not None:
        failures.append(r4)

    return failures


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-sha", required=True, help="Base branch commit SHA")
    parser.add_argument("--head-sha", required=True, help="PR head commit SHA")
    parser.add_argument("--pr-title", required=True, help="PR title string")
    parser.add_argument("--pr-body", default="", help="PR body / description")
    parser.add_argument("--repo", default="", help="owner/repo (for gh CLI calls)")
    parser.add_argument("--pr-number", default="", help="PR number (for gh CLI calls)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.  Returns exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    base_sha: str = args.base_sha
    head_sha: str = args.head_sha

    if not base_sha or not head_sha:
        print("ERROR: --base-sha and --head-sha are required.", file=sys.stderr)
        return 2

    print(
        f"[anti_phantom_merge_guard] base_sha={base_sha[:12]} head_sha={head_sha[:12]}"
    )

    diff_output = get_diff_via_git(base_sha=base_sha, head_sha=head_sha)
    num_changed = count_changed_files(diff_output)
    print(f"[anti_phantom_merge_guard] changed_files={num_changed}")

    # Collect commits
    try:
        raw_commits = subprocess.check_output(
            ["git", "log", "--format=%H|%an|%s", f"{base_sha}..{head_sha}"],
            text=False,
        ).decode()
        commits = [ln for ln in raw_commits.splitlines() if ln.strip()]
    except subprocess.CalledProcessError:
        commits = []

    failures = run_all_checks(
        base_sha=base_sha,
        head_sha=head_sha,
        pr_title=args.pr_title,
        pr_body=args.pr_body,
        diff_output=diff_output,
        commits=commits,
    )

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}")
        return 1

    print("[anti_phantom_merge_guard] All phantom-guard rules passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
