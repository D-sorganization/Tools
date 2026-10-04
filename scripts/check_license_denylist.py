"""ADR-008 dependency and license deny-list check for the MIT core (#5417).

Two checks, driven by ``config/license_denylist.json``:

* static: every requirement declared in ``pyproject.toml`` (core and every
  extra) and the root requirements files must not name a denied package;
* installed (``--installed``): the transitive closure of the core
  ``[project].dependencies`` must not contain a denied package or a
  distribution whose license metadata names a denied SPDX identifier.

Exit codes: 0 clean, 1 on any violation or unusable configuration.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import re
import sys
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

logger = logging.getLogger(__name__)

CONFIG_RELPATH = Path("config") / "license_denylist.json"
REQUIREMENTS_FILES = (
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-rate-pyqt.txt",
)
AGPL_CLASSIFIER = "GNU Affero General Public License v3"


class DenylistConfigError(Exception):
    """The deny-list configuration is missing or malformed (fail closed)."""


@dataclass(frozen=True)
class Violation:
    """A deny-list finding, rendered as ``package -> reason``."""

    package: str
    reason: str

    def __str__(self) -> str:
        return f"{self.package} -> {self.reason}"


@dataclass(frozen=True)
class Denylist:
    packages: frozenset[str]
    license_ids: tuple[str, ...]


def normalize_name(name: str) -> str:
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def load_denylist(root: Path) -> Denylist:
    """Load the deny-list; raise DenylistConfigError if unusable."""
    path = root / CONFIG_RELPATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        packages = data["denied_packages"]
        license_ids = data["denied_license_ids"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DenylistConfigError(f"cannot load {path}: {exc!r}") from exc
    if not (
        isinstance(packages, list)
        and isinstance(license_ids, list)
        and all(isinstance(x, str) for x in (*packages, *license_ids))
    ):
        raise DenylistConfigError(f"{path}: denied lists must be lists of strings")
    return Denylist(
        frozenset(normalize_name(p) for p in packages),
        tuple(i.lower() for i in license_ids),
    )


def _parse(spec: str, source: str) -> Requirement | None:
    try:
        return Requirement(spec)
    except InvalidRequirement:
        logger.warning("%s: skipping unparseable requirement %r", source, spec)
        return None


def _pyproject_specs(root: Path) -> list[tuple[str, str]]:
    path = root / "pyproject.toml"
    project = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
    specs = [(s, f"{path.name}:dependencies") for s in project.get("dependencies", [])]
    for extra, items in project.get("optional-dependencies", {}).items():
        specs.extend((s, f"{path.name}:extra[{extra}]") for s in items)
    return specs


def _requirements_file_specs(path: Path) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = re.sub(r"(^|\s)#.*$", "", line).strip()
        if text and not text.startswith("-"):
            specs.append((text, path.name))
    return specs


def check_static(root: Path, denylist: Denylist) -> list[Violation]:
    """Check declared requirements for denied package names."""
    specs = _pyproject_specs(root)
    for name in REQUIREMENTS_FILES:
        path = root / name
        if path.is_file():
            specs.extend(_requirements_file_specs(path))
    violations = []
    for spec, source in specs:
        req = _parse(spec, source)
        if req is None:
            continue
        normalized = normalize_name(req.name)
        if normalized in denylist.packages:
            violations.append(
                Violation(normalized, f"denied package declared in {source}")
            )
    return violations


def _license_violation(
    dist: importlib.metadata.Distribution, denylist: Denylist
) -> str | None:
    meta = dist.metadata
    for value in (meta.get("License-Expression"), meta.get("License")):
        lowered = (value or "").lower()
        for spdx in denylist.license_ids:
            if spdx in lowered:
                return f"denied license {spdx!r} in {value!r}"
    for classifier in meta.get_all("Classifier") or []:
        if AGPL_CLASSIFIER in classifier:
            return f"denied license classifier {classifier!r}"
    return None


def _core_requirements(root: Path) -> list[Requirement]:
    path = root / "pyproject.toml"
    deps = tomllib.loads(path.read_text(encoding="utf-8"))["project"]["dependencies"]
    parsed = (_parse(s, path.name) for s in deps)
    return [r for r in parsed if r is not None]


def check_installed(
    root: Path, denylist: Denylist
) -> tuple[list[Violation], list[Violation]]:
    """Walk the installed closure of core deps; return (violations, warnings)."""
    violations: list[Violation] = []
    warnings: list[Violation] = []
    seen: set[str] = set()
    queue = [normalize_name(r.name) for r in _core_requirements(root)]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in denylist.packages:
            violations.append(Violation(name, "denied package in installed closure"))
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            warnings.append(Violation(name, "not installed"))
            continue
        reason = _license_violation(dist, denylist)
        if reason:
            violations.append(Violation(name, reason))
        for spec in dist.requires or []:
            req = _parse(spec, name)
            if req is None:
                continue
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            queue.append(normalize_name(req.name))
    return violations, warnings


def _report(violations: Iterable[Violation], warnings: Iterable[Violation]) -> None:
    for warning in warnings:
        logger.warning("%s", warning)
    for violation in violations:
        logger.error("%s", violation)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--installed", action="store_true", help="also walk the installed environment"
    )
    args = parser.parse_args(argv)
    try:
        denylist = load_denylist(args.root)
        violations = check_static(args.root, denylist)
        warnings: list[Violation] = []
        if args.installed:
            more, warnings = check_installed(args.root, denylist)
            violations.extend(more)
    except (DenylistConfigError, OSError, tomllib.TOMLDecodeError, KeyError) as exc:
        logger.error("license deny-list check could not run: %s", exc)
        return 1
    _report(violations, warnings)
    if violations:
        return 1
    logger.info("license deny-list check passed")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
