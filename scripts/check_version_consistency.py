#!/usr/bin/env python3
"""Check (or set) the release version across every file that declares it.

Sources (Tools #4910):
    VERSION                     plain text
    pyproject.toml              [project] version
    package.json                "version" (workspace root)
    helm/*/Chart.yaml           appVersion  (only if a chart exists)

``--set X.Y.Z`` rewrites all of them; the default mode exits 1 and names every
file that disagrees with ``VERSION``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
PYPROJECT_RE = re.compile(
    r'^(?P<prefix>version\s*=\s*")(?P<version>[^"]+)(?P<suffix>")', re.M
)
PACKAGE_RE = re.compile(
    r'^(?P<prefix>\s*"version"\s*:\s*")(?P<version>[^"]+)(?P<suffix>")', re.M
)
CHART_RE = re.compile(
    r'^(?P<prefix>appVersion:\s*"?)(?P<version>[^"\s]+)(?P<suffix>"?)\s*$', re.M
)


def _chart_files(root: Path) -> list[Path]:
    helm = root / "helm"
    return sorted(helm.glob("*/Chart.yaml")) if helm.is_dir() else []


def read_versions(root: Path) -> dict[str, str | None]:
    """Map of relative file -> declared version (None when undeclared)."""
    out: dict[str, str | None] = {}
    version_file = root / "VERSION"
    out["VERSION"] = (
        version_file.read_text(encoding="utf-8").strip()
        if version_file.exists()
        else None
    )

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        match = PYPROJECT_RE.search(pyproject.read_text(encoding="utf-8"))
        out["pyproject.toml"] = match.group("version") if match else None

    package = root / "package.json"
    if package.exists():
        try:
            data = json.loads(package.read_text(encoding="utf-8"))
            out["package.json"] = (
                str(data.get("version")) if "version" in data else None
            )
        except json.JSONDecodeError:
            out["package.json"] = None

    for chart in _chart_files(root):
        match = CHART_RE.search(chart.read_text(encoding="utf-8"))
        out[chart.relative_to(root).as_posix()] = (
            match.group("version") if match else None
        )
    return out


def mismatches(versions: dict[str, str | None]) -> list[str]:
    """Files whose version differs from ``VERSION`` (or is missing)."""
    expected = versions.get("VERSION")
    if expected is None:
        return ["VERSION (missing)"]
    bad = []
    for name, found in versions.items():
        if name == "VERSION":
            continue
        if found != expected:
            bad.append(f"{name} ({found or 'undeclared'} != {expected})")
    return bad


def _sub(pattern: re.Pattern[str], text: str, version: str) -> str:
    return pattern.sub(
        lambda m: f"{m.group('prefix')}{version}{m.group('suffix')}", text, count=1
    )


def set_version(root: Path, version: str) -> list[str]:
    """Write ``version`` into every source; return the files touched."""
    if not SEMVER_RE.match(version):
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {version!r}")
    touched: list[str] = []
    (root / "VERSION").write_text(version + "\n", encoding="utf-8", newline="\n")
    touched.append("VERSION")

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8")
        pyproject.write_text(
            _sub(PYPROJECT_RE, text, version), encoding="utf-8", newline="\n"
        )
        touched.append("pyproject.toml")

    package = root / "package.json"
    if package.exists():
        text = package.read_text(encoding="utf-8")
        package.write_text(
            _sub(PACKAGE_RE, text, version), encoding="utf-8", newline="\n"
        )
        touched.append("package.json")

    for chart in _chart_files(root):
        text = chart.read_text(encoding="utf-8")
        chart.write_text(_sub(CHART_RE, text, version), encoding="utf-8", newline="\n")
        touched.append(chart.relative_to(root).as_posix())
    return touched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--set", dest="set_version", metavar="X.Y.Z", default=None)
    args = parser.parse_args(argv)

    if args.set_version:
        try:
            touched = set_version(args.root, args.set_version)
        except ValueError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        sys.stdout.write(f"set {args.set_version} in: {', '.join(touched)}\n")

    versions = read_versions(args.root)
    for name, found in versions.items():
        sys.stdout.write(f"{name}: {found or '<undeclared>'}\n")
    bad = mismatches(versions)
    if bad:
        sys.stderr.write("version mismatch:\n- " + "\n- ".join(bad) + "\n")
        return 1
    sys.stdout.write("versions consistent\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
