"""bump_vendor_pin.py — Vendor-pin bump helper for Tools consumers.

This script automates the process of opening bump PRs in consumer repos
(UpstreamDrift, Gasification_Model) when a new Tools release tag is
published.

Usage
-----
    python scripts/bump_vendor_pin.py [--tag TAG] [--dry-run]

Options
-------
--tag TAG       The Tools release tag to pin (default: resolve latest tag
                via ``git describe --tags --abbrev=0``).
--dry-run       Print what would happen without opening any PRs.
--consumers     Comma-separated subset of consumers (default: all).

Design
------
* DbC preconditions guard every public function (see assert statements).
* All subprocess calls are isolated so tests can mock them cheaply.
* No global mutable state — configuration is passed explicitly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Known consumer repositories that receive vendor-pin bump PRs.
CONSUMER_REPOS: list[str] = [
    "D-sorganization/UpstreamDrift",
    "D-sorganization/Gasification_Model",
]

#: Branch prefix used for bump PRs in consumer repos.
BUMP_BRANCH_PREFIX = "chore/bump-tools-vendor-pin"

#: PR title template.
_PR_TITLE_TEMPLATE = "chore: bump Tools vendor pin to {tag}"

#: PR body template.
_PR_BODY_TEMPLATE = """\
## Automated vendor-pin bump

| Field | Value |
|-------|-------|
| Tools tag | `{tag}` |
| Commit SHA | `{sha}` |
| Consumer repo | `{consumer_repo}` |

This PR was opened automatically by `scripts/bump_vendor_pin.py` in the
Tools repository.  It updates the vendored Tools dependency to the latest
release so that `{consumer_repo}` stays in sync with upstream.

### What to check

- [ ] Consumer smoke tests pass (triggered automatically by CI)
- [ ] No breaking API changes in the diff between the previous pin and `{tag}`
- [ ] If smoke tests **fail**, leave this PR open with a comment explaining
      the blocker — do **not** close or force-merge

### How to update the pin manually

```bash
# In the consumer repo
git checkout -b chore/bump-tools-vendor-pin-{tag}
# Update whichever pin file the repo uses:
#   UpstreamDrift:       src/shared/python/ (subtree or submodule SHA)
#   Gasification_Model:  vendor/ud-tools/   (subtree or submodule SHA)
echo "{sha}" > .vendor-tools-sha   # or equivalent pin file
git commit -am "chore: bump Tools vendor pin to {tag}"
git push origin chore/bump-tools-vendor-pin-{tag}
```

*Filed by `scripts/bump_vendor_pin.py` — see `docs/ops/vendor_pins.md`
for the full cadence and process documentation.*
"""


# ---------------------------------------------------------------------------
# Validation helpers (DbC preconditions)
# ---------------------------------------------------------------------------


def validate_consumer(consumer_repo: str) -> None:
    """Raise *ValueError* if *consumer_repo* is not in the known allowlist.

    Precondition: consumer_repo is a non-empty string.
    Postcondition: no exception means the repo is safe to target.
    """
    assert (
        isinstance(consumer_repo, str) and consumer_repo
    ), "consumer_repo must be a non-empty string"
    if consumer_repo not in CONSUMER_REPOS:
        raise ValueError(
            f"Unknown consumer repo {consumer_repo!r}.  Allowed: {CONSUMER_REPOS}"
        )


# ---------------------------------------------------------------------------
# PR body builder
# ---------------------------------------------------------------------------


def build_pr_body(*, tag: str, sha: str, consumer_repo: str) -> str:
    """Return a Markdown PR body for a vendor-pin bump.

    Preconditions:
    - tag must be non-empty
    - sha must be non-empty
    - consumer_repo must be in CONSUMER_REPOS
    """
    assert tag, "tag must be a non-empty string"
    assert sha, "sha must be a non-empty string"
    validate_consumer(consumer_repo)  # raises ValueError for unknown repos

    return _PR_BODY_TEMPLATE.format(
        tag=tag,
        sha=sha,
        consumer_repo=consumer_repo,
    )


# ---------------------------------------------------------------------------
# Tag / SHA resolution
# ---------------------------------------------------------------------------


def resolve_latest_tag() -> str:
    """Return the latest git tag reachable from HEAD.

    Uses ``git describe --tags --abbrev=0``.

    Postcondition: returns a non-empty string.
    Raises: RuntimeError if no tags are found.
    """
    raw = subprocess.check_output(
        ["git", "describe", "--tags", "--abbrev=0"],
        text=False,
    )
    tag = raw.decode().strip()
    if not tag:
        raise RuntimeError(
            "No tags found in this repository.  "
            "Create a release tag before running bump_vendor_pin."
        )
    return tag


def resolve_sha_for_tag(tag: str) -> str:
    """Return the full commit SHA for *tag*.

    Precondition: tag must be a non-empty string.
    Postcondition: returns a non-empty hex string.
    """
    assert tag, "tag must be a non-empty string"

    raw = subprocess.check_output(
        ["git", "rev-list", "-n", "1", tag],
        text=False,
    )
    return raw.decode().strip()


# ---------------------------------------------------------------------------
# Core bump logic
# ---------------------------------------------------------------------------


def bump_consumer(
    *,
    consumer_repo: str,
    tag: str,
    sha: str,
    dry_run: bool = False,
) -> None:
    """Open a bump PR in *consumer_repo* (or print the plan if *dry_run*).

    Preconditions:
    - consumer_repo in CONSUMER_REPOS
    - tag is non-empty
    - sha is non-empty
    """
    validate_consumer(consumer_repo)
    assert tag, "tag must be non-empty"
    assert sha, "sha must be non-empty"

    branch = f"{BUMP_BRANCH_PREFIX}-{tag}"
    title = _PR_TITLE_TEMPLATE.format(tag=tag)
    body = build_pr_body(tag=tag, sha=sha, consumer_repo=consumer_repo)

    print(f"[bump_vendor_pin] Consumer: {consumer_repo}")
    print(f"[bump_vendor_pin] Branch:   {branch}")
    print(f"[bump_vendor_pin] Title:    {title}")

    if dry_run:
        print("[bump_vendor_pin] DRY RUN — no PR opened.")
        print("--- PR body ---")
        print(body)
        print("--- end ---")
        return

    # Open the PR via gh CLI.
    # NOTE: The consumer repo is expected to have its own CI that runs smoke
    # tests.  If the smoke tests fail the PR is left open.  The gh CLI cannot
    # create the branch in the consumer repo directly from here — a workflow
    # in that repo must be triggered via repository_dispatch or the branch
    # must already exist.  For now this command opens the PR against the
    # consumer's default branch.
    cmd = [
        "gh",
        "pr",
        "create",
        "--repo",
        consumer_repo,
        "--title",
        title,
        "--body",
        body,
        "--base",
        "main",
        "--head",
        branch,
    ]
    try:
        subprocess.check_call(cmd)
        print(f"[bump_vendor_pin] PR opened in {consumer_repo}.")
    except subprocess.CalledProcessError as exc:
        print(
            f"[bump_vendor_pin] WARNING: gh pr create failed for {consumer_repo}: {exc}",
            file=sys.stderr,
        )
        print(
            "[bump_vendor_pin] The PR was NOT opened.  "
            "Fix the error and re-run, or open it manually.",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns exit code (0 = success)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Tools release tag to pin (default: latest tag from git describe)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without opening any PRs",
    )
    parser.add_argument(
        "--consumers",
        default=None,
        help="Comma-separated consumer repos (default: all known consumers)",
    )
    args = parser.parse_args(argv)

    tag = args.tag or resolve_latest_tag()
    sha = resolve_sha_for_tag(tag)

    if args.consumers:
        consumers = [c.strip() for c in args.consumers.split(",")]
        for repo in consumers:
            validate_consumer(repo)
    else:
        consumers = list(CONSUMER_REPOS)

    print(f"[bump_vendor_pin] Tag: {tag}  SHA: {sha}")
    print(f"[bump_vendor_pin] Consumers: {consumers}")

    for repo in consumers:
        bump_consumer(
            consumer_repo=repo,
            tag=tag,
            sha=sha,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
