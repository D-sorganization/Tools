#!/usr/bin/env python3
"""Set the release-derived ``Spec Version`` field in SPEC.md.

Before Repository_Management#1520 every pull request bumped this field, which
made it a global counter two concurrent pull requests always fought over — for
no benefit, because nothing consumed the number. From #1520 onward the field is
**release-derived**: it is set when a release is cut, from the release tag or
the newest ``CHANGELOG.md`` heading, by this script. Individual pull requests
must not touch it.

Usage::

    python scripts/bump_spec_version.py --version 1.3.0
    python scripts/bump_spec_version.py --from-changelog
    python scripts/bump_spec_version.py --from-tag v1.3.0
    python scripts/bump_spec_version.py --check     # CI: is the field consistent?

``--check`` exits non-zero only when the field is missing or malformed. It
deliberately does **not** require the field to equal the newest change-log row:
that equality was the second half of the serial treadmill.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _field_pattern(label: str) -> re.Pattern[str]:
    """Match one Identity-table row, capturing its padded value cell."""
    return re.compile(
        rf"^(?P<prefix>\|\s*\*\*{label}\*\*\s*\|\s*)"
        r"(?P<value>[^|]*?)(?P<suffix>\s*\|)\s*$",
        re.MULTILINE,
    )


SPEC_VERSION_RE = _field_pattern("Spec Version")
LAST_UPDATE_RE = _field_pattern("Last Spec Update")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
CHANGELOG_HEADING_RE = re.compile(
    r"^##\s*\[?v?(?P<version>\d+\.\d+\.\d+)\]?", re.MULTILINE
)


class BumpError(RuntimeError):
    """Raised when the spec version cannot be read or resolved."""


def read_version(text: str) -> str:
    match = SPEC_VERSION_RE.search(text)
    if match is None:
        raise BumpError("SPEC.md has no '**Spec Version**' row in the Identity table")
    return match.group("value").strip()


def version_from_changelog(changelog: str) -> str:
    match = CHANGELOG_HEADING_RE.search(changelog)
    if match is None:
        raise BumpError("CHANGELOG.md has no '## <version>' heading to read")
    return match.group("version")


def version_from_tag(tag: str) -> str:
    candidate = tag.lstrip("vV")
    if not SEMVER_RE.match(candidate):
        raise BumpError(f"tag {tag!r} is not a semantic version")
    return candidate


def _pad_to(width: int, value: str) -> str:
    return value.ljust(width) if width > len(value) else value


def set_version(text: str, version: str, *, today: str) -> str:
    if not SEMVER_RE.match(version):
        raise BumpError(f"{version!r} is not a semantic version (X.Y.Z)")

    def replace_version(match: re.Match[str]) -> str:
        width = len(match.group("value"))
        return (
            f"{match.group('prefix')}{_pad_to(width, version)}{match.group('suffix')}"
        )

    def replace_date(match: re.Match[str]) -> str:
        width = len(match.group("value"))
        return f"{match.group('prefix')}{_pad_to(width, today)}{match.group('suffix')}"

    if SPEC_VERSION_RE.search(text) is None:
        raise BumpError("SPEC.md has no '**Spec Version**' row in the Identity table")
    text = SPEC_VERSION_RE.sub(replace_version, text, count=1)
    return LAST_UPDATE_RE.sub(replace_date, text, count=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(ROOT / "SPEC.md"))
    parser.add_argument("--changelog", default=str(ROOT / "CHANGELOG.md"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--version", help="explicit X.Y.Z")
    source.add_argument("--from-changelog", action="store_true")
    source.add_argument("--from-tag", help="release tag, e.g. v1.3.0")
    source.add_argument(
        "--check",
        action="store_true",
        help="verify the field exists and is a semantic version; change nothing",
    )
    parser.add_argument("--today", default=dt.date.today().isoformat())
    args = parser.parse_args(argv)

    spec_path = Path(args.spec)
    text = spec_path.read_text(encoding="utf-8")

    try:
        if args.check:
            current = read_version(text)
            if not SEMVER_RE.match(current):
                print(f"ERROR: Spec Version {current!r} is not a semantic version")
                return 1
            print(f"Spec Version {current} (release-derived; not bumped per PR).")
            return 0

        if args.version:
            version = args.version
        elif args.from_tag:
            version = version_from_tag(args.from_tag)
        else:
            version = version_from_changelog(
                Path(args.changelog).read_text(encoding="utf-8")
            )
        updated = set_version(text, version, today=args.today)
    except (BumpError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1

    if updated == text:
        print(f"Spec Version already {version}; nothing to do.")
        return 0
    spec_path.write_text(updated, encoding="utf-8", newline="\n")
    print(f"Spec Version set to {version} (Last Spec Update {args.today}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
