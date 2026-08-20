"""Fail closed when the trusted Rate PyQt binary stack is inconsistent."""

from __future__ import annotations

import argparse
import importlib
import logging
from importlib.metadata import version as metadata_version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

LOGGER = logging.getLogger(__name__)
REQUIRED_DISTRIBUTIONS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pyqt6": "PyQt6",
}


def read_expected_versions(constraints_path: Path) -> dict[str, str]:
    """Read exact binary-stack pins from a pip constraints file.

    Args:
        constraints_path: Existing UTF-8 pip constraints file.

    Returns:
        Canonical distribution names mapped to exact expected versions.

    Raises:
        ValueError: If a required distribution is missing, duplicated, or not
            pinned to one concrete version.
    """
    expected: dict[str, str] = {}
    for raw_line in constraints_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        if name not in REQUIRED_DISTRIBUTIONS:
            continue
        if name in expected:
            raise ValueError(f"duplicate required constraint: {requirement.name}")
        specifiers = list(requirement.specifier)
        if (
            len(specifiers) != 1
            or specifiers[0].operator != "=="
            or "*" in specifiers[0].version
        ):
            raise ValueError(f"{requirement.name} must be exactly pinned")
        expected[name] = specifiers[0].version

    missing = sorted(set(REQUIRED_DISTRIBUTIONS) - set(expected))
    if missing:
        raise ValueError(f"missing required constraints: {', '.join(missing)}")
    return expected


def _import_runtime() -> None:
    """Import the ABI-sensitive modules used by the rendered Rate window."""
    for module_name in (
        "numpy",
        "scipy.integrate",
        "scipy.sparse.linalg",
        "PyQt6.QtWidgets",
    ):
        importlib.import_module(module_name)


def verify_runtime(constraints_path: Path) -> dict[str, str]:
    """Verify exact installed versions, then import the binary runtime."""
    expected = read_expected_versions(constraints_path)
    installed = {
        name: metadata_version(distribution)
        for name, distribution in REQUIRED_DISTRIBUTIONS.items()
    }
    mismatches = [
        f"{name} {installed[name]} != constrained {expected[name]}"
        for name in REQUIRED_DISTRIBUTIONS
        if installed[name] != expected[name]
    ]
    if mismatches:
        raise RuntimeError("; ".join(mismatches))
    _import_runtime()
    LOGGER.info("Verified isolated Rate PyQt runtime: %s", installed)
    return installed


def main() -> int:
    """Run the trusted Rate PyQt environment check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    verify_runtime(args.constraints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
