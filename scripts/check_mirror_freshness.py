"""Freshness check for the rate-of-closure-explorer public mirror.

The public Rate of Closure Impact Explorer
(https://d-sorganization.github.io/rate-of-closure-explorer/) is a Pages
mirror of the canonical ``src/rate_of_closure/web`` tree in this repo.  The
sync is a manual ``scripts/sync-from-tools.ps1`` run in the mirror repo, so
nothing detects drift when canonical ``web/`` moves after the last sync.
This script computes that drift and reports it (issue #4624, WS4 of
UpstreamDrift EPIC #8965).

Primary drift signal
    The last-change committer date of the canonical ``web/`` subtree
    (``git log -1 --format=%cI -- <subdir>``) versus the committer date of
    the mirror repo's most recent commit.  If canonical changed after the
    last mirror sync, the mirror has drifted.  When a sync commit records a
    canonical Tools SHA (per the release process), that recorded SHA is
    compared against the canonical HEAD commit instead -- an exact signal.

Deep signal (``--deep``)
    Git blob-SHA comparison of every canonical-tracked file under ``web/``
    against the mirror's root tree (the sync copies ``web/*`` to the mirror
    root).  Mirror-only scaffolding (LICENSE, README, scripts/, .github/)
    is ignored; only canonical paths are compared.

Contracts (DbC)
    Preconditions
        * ``CanonicalState``/``MirrorState`` commit ids are 7-40 char hex.
        * All datetimes are timezone-aware (naive datetimes are rejected).
        * ``files`` mappings, when provided, map POSIX-style relative paths
          to git blob SHAs.
    Postconditions
        * ``assess_freshness`` always returns a ``FreshnessReport`` whose
          ``fresh`` flag is consistent with ``drifted_files`` when a deep
          comparison was performed (fresh iff no drifted files).
        * The CLI exits 0 when fresh, 1 when drifted, 2 on any error.
    Invariant (of the channel, checked here, enforced by the mirror CI)
        * Mirror content is a pure function of a recorded canonical Tools
          commit; the mirror's own parity tests gate every deploy.

No network access happens in the pure core (``assess_freshness`` and the
parsing helpers); the CLI wrapper shells out to ``git`` and the
authenticated ``gh`` CLI so no new Python dependencies are needed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_MIRROR_REPO = "D-sorganization/rate-of-closure-explorer"
DEFAULT_WEB_SUBDIR = "src/rate_of_closure/web"

EXIT_FRESH = 0
EXIT_DRIFTED = 1
EXIT_ERROR = 2

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_RECORDED_SHA_RE = re.compile(
    r"(?:tools|canonical)"
    r"(?:[\s_-]*(?:commit|sha|rev(?:ision)?)\s*[:=@]?|\s*[:=@])"
    r"\s*([0-9a-f]{7,40})\b",
    re.IGNORECASE,
)


class ContractViolation(ValueError):
    """Raised when a DbC precondition on input data is violated."""


def _require_sha(value: str, name: str) -> str:
    if not _SHA_RE.match(value):
        raise ContractViolation(f"{name} must be a 7-40 char hex sha, got {value!r}")
    return value


def _require_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ContractViolation(f"{name} must be timezone-aware, got {value!r}")
    return value


@dataclass(frozen=True)
class CanonicalState:
    """State of the canonical ``web/`` subtree in this repo.

    Precondition: ``commit`` is hex; ``last_change`` is timezone-aware.
    """

    commit: str
    last_change: datetime
    files: dict[str, str] | None = None

    def __post_init__(self) -> None:
        _require_sha(self.commit, "canonical commit")
        _require_aware(self.last_change, "canonical last_change")


@dataclass(frozen=True)
class MirrorState:
    """State of the public mirror repo.

    Precondition: ``last_sync_commit`` is hex; ``last_sync_date`` is
    timezone-aware; ``recorded_canonical_commit`` is hex when present.
    """

    last_sync_commit: str
    last_sync_date: datetime
    recorded_canonical_commit: str | None = None
    files: dict[str, str] | None = None

    def __post_init__(self) -> None:
        _require_sha(self.last_sync_commit, "mirror last_sync_commit")
        _require_aware(self.last_sync_date, "mirror last_sync_date")
        if self.recorded_canonical_commit is not None:
            _require_sha(self.recorded_canonical_commit, "recorded_canonical_commit")


@dataclass(frozen=True)
class FreshnessReport:
    """Machine-readable drift verdict.

    Postcondition: when ``deep`` is True, ``fresh == (not drifted_files)``.
    """

    fresh: bool
    canonical_commit: str
    canonical_last_change: str
    mirror_last_sync: str
    mirror_last_sync_date: str
    signal: str
    reason: str
    deep: bool = False
    drifted_files: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    def summary(self) -> str:
        verdict = "FRESH" if self.fresh else "DRIFTED"
        lines = [
            f"mirror freshness: {verdict}",
            f"  canonical commit:      {self.canonical_commit}",
            f"  canonical last change: {self.canonical_last_change}",
            f"  mirror last sync:      {self.mirror_last_sync}"
            f" ({self.mirror_last_sync_date})",
            f"  signal:                {self.signal}",
            f"  reason:                {self.reason}",
        ]
        if self.deep:
            lines.append(f"  drifted files:         {len(self.drifted_files)}")
            lines.extend(f"    {path}" for path in self.drifted_files[:50])
            if len(self.drifted_files) > 50:
                lines.append(f"    ... and {len(self.drifted_files) - 50} more")
        return "\n".join(lines)


def parse_recorded_canonical_commit(message: str) -> str | None:
    """Extract a recorded canonical Tools SHA from a sync commit message.

    Returns None when the message records no SHA (the current sync commits
    do not; the release process asks future syncs to).
    """
    match = _RECORDED_SHA_RE.search(message)
    return match.group(1).lower() if match else None


def _shas_match(recorded: str, canonical: str) -> bool:
    short = min(len(recorded), len(canonical))
    return recorded[:short] == canonical[:short]


def compare_trees(
    canonical_files: dict[str, str], mirror_files: dict[str, str]
) -> list[str]:
    """Return canonical paths missing from or differing in the mirror.

    Only canonical-tracked paths are compared; mirror-only scaffolding is
    ignored by construction.  Identical content has identical git blob
    SHAs, so equality of SHAs is equality of content.
    """
    drifted = [
        path for path, blob in canonical_files.items() if mirror_files.get(path) != blob
    ]
    return sorted(drifted)


def assess_freshness(canonical: CanonicalState, mirror: MirrorState) -> FreshnessReport:
    """Pure drift assessment from two pre-gathered states.

    Signal priority: deep tree comparison (when both states carry file
    listings) > recorded canonical SHA in the sync commit > timestamp
    comparison (canonical last-change vs mirror last-sync date).
    """
    base = {
        "canonical_commit": canonical.commit,
        "canonical_last_change": canonical.last_change.isoformat(),
        "mirror_last_sync": mirror.last_sync_commit,
        "mirror_last_sync_date": mirror.last_sync_date.isoformat(),
    }
    if canonical.files is not None and mirror.files is not None:
        drifted = compare_trees(canonical.files, mirror.files)
        return FreshnessReport(
            fresh=not drifted,
            signal="tree",
            reason=(
                "all canonical web/ files present in mirror with identical blobs"
                if not drifted
                else f"{len(drifted)} canonical file(s) missing or differing"
            ),
            deep=True,
            drifted_files=drifted,
            **base,
        )
    if mirror.recorded_canonical_commit is not None:
        fresh = _shas_match(mirror.recorded_canonical_commit, canonical.commit)
        return FreshnessReport(
            fresh=fresh,
            signal="recorded-sha",
            reason=(
                f"mirror sync records canonical {mirror.recorded_canonical_commit}"
                + ("" if fresh else f", canonical HEAD is {canonical.commit}")
            ),
            **base,
        )
    fresh = mirror.last_sync_date >= canonical.last_change
    return FreshnessReport(
        fresh=fresh,
        signal="timestamp",
        reason=(
            "mirror synced at or after last canonical web/ change"
            if fresh
            else "canonical web/ changed after the last mirror sync"
        ),
        **base,
    )


# ---------------------------------------------------------------------------
# CLI wrapper: the only part that touches git / the network (via gh).
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=120, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {result.stderr.strip()}")
    return result.stdout


def gather_canonical_state(
    repo_root: Path, web_subdir: str, deep: bool
) -> CanonicalState:
    out = _run(
        ["git", "log", "-1", "--format=%H %cI", "--", web_subdir], cwd=repo_root
    ).strip()
    if not out:
        raise RuntimeError(f"no commits touch {web_subdir} in {repo_root}")
    sha, iso = out.split(" ", 1)
    files: dict[str, str] | None = None
    if deep:
        listing = _run(
            ["git", "ls-tree", "-r", "HEAD", "--", web_subdir], cwd=repo_root
        )
        prefix = web_subdir.rstrip("/") + "/"
        files = {}
        for line in listing.splitlines():
            meta, path = line.split("\t", 1)
            files[path.removeprefix(prefix)] = meta.split()[2]
    return CanonicalState(
        commit=sha, last_change=datetime.fromisoformat(iso), files=files
    )


def gather_mirror_state(mirror_repo: str, deep: bool) -> MirrorState:
    payload = json.loads(_run(["gh", "api", f"repos/{mirror_repo}/commits?per_page=1"]))
    if not payload:
        raise RuntimeError(f"mirror {mirror_repo} has no commits")
    head = payload[0]
    files: dict[str, str] | None = None
    if deep:
        tree = json.loads(
            _run(["gh", "api", f"repos/{mirror_repo}/git/trees/HEAD?recursive=1"])
        )
        if tree.get("truncated"):
            raise RuntimeError("mirror tree listing truncated; cannot deep-compare")
        files = {
            entry["path"]: entry["sha"]
            for entry in tree["tree"]
            if entry["type"] == "blob"
        }
    return MirrorState(
        last_sync_commit=head["sha"],
        last_sync_date=datetime.fromisoformat(
            head["commit"]["committer"]["date"].replace("Z", "+00:00")
        ),
        recorded_canonical_commit=parse_recorded_canonical_commit(
            head["commit"]["message"]
        ),
        files=files,
    )


def exit_code_for(report: FreshnessReport) -> int:
    """Map a report to the CLI exit code (0 fresh, 1 drifted)."""
    return EXIT_FRESH if report.fresh else EXIT_DRIFTED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report drift between canonical web/ and the public mirror."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Tools repo root (default: this checkout)",
    )
    parser.add_argument("--mirror-repo", default=DEFAULT_MIRROR_REPO)
    parser.add_argument("--web-subdir", default=DEFAULT_WEB_SUBDIR)
    parser.add_argument(
        "--deep",
        action="store_true",
        help="compare per-file git blob SHAs instead of relying on dates",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the JSON report on stdout"
    )
    args = parser.parse_args(argv)
    try:
        canonical = gather_canonical_state(
            args.repo_root, args.web_subdir, deep=args.deep
        )
        mirror = gather_mirror_state(args.mirror_repo, deep=args.deep)
        report = assess_freshness(canonical, mirror)
    except (RuntimeError, ContractViolation, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(report.to_json() if args.json else report.summary())
    return exit_code_for(report)


if __name__ == "__main__":
    sys.exit(main())
